//! An end-to-end CDDA-like simulation test: a 5×5 grid world where a survivor
//! crafts a weapon from materials scattered between their clothing pockets and
//! the ground, planned entirely by the HTN planner.
//!
//! # The world (idiomatic Bevy, data-driven)
//!
//! - **Grid + spatial index**: ground items are entities with a [`Pos`]; a
//!   [`SpatialIndex`] resource (rebuilt each tick) maps positions to entities
//!   so the pickup system resolves targets in O(1).
//! - **Clothing/pocket inventory via Relationships**: the survivor *wears*
//!   clothing (`WornBy` relationship → `Wearing` target) and items sit *in
//!   pockets* (`InPocket` relationship → `Pockets` target). There is no
//!   monolithic inventory struct — contents are the relationship graph, and a
//!   sync system derives the planner-facing [`PocketContents`] summary from
//!   it every tick.
//! - **Crafting is data**: a [`RecipeBook`] component holds [`Recipe`]s
//!   (output ← inputs); the HTN domain is generic over them. Nothing in the
//!   domain mentions "spear" — the goal is a [`CraftGoal`] component.
//!
//! # The HTN layer
//!
//! The planner owns *decisions*; game systems own *the world*. HTN task
//! actions dispatch intent markers (`PickupRequest` / `CraftRequest`) that
//! dedicated systems realize against the spatial index and relationship
//! graph; movement is ordered through a [`Travel`] component the movement
//! system steps one tile per tick. The planner simulates all of this on the
//! scratchpad (effects), the driver re-validates each step against the real
//! world every tick, and world drift (the survivor not having arrived yet)
//! drops the plan and replans — the CDDA replan loop, end to end.
//!
//! The scenario: the survivor at (0,0) wears a jacket holding a stick; a rag
//! lies at (4,4). Goal: craft a spear (stick + rag). The planner must send
//! them across the map, pick the rag up, and craft.

use std::collections::HashMap;

use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
use bevy_bhtn::tasks::TaskBuilder;
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;
use bevy_ecs::system::EntityCommands;

// ---------------------------------------------------------------------------
// World model — components
// ---------------------------------------------------------------------------

/// Every item kind in the game. Data, not code: recipes reference these.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
enum ItemKind {
    #[default]
    Stick,
    Rag,
    Spear,
}

/// A tile position on the 5×5 grid.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
struct Pos {
    x: i32,
    y: i32,
}

impl Pos {
    const WORLD_SIZE: i32 = 5;
    fn stepped_toward(self, target: Pos) -> Pos {
        let mut p = self;
        if p.x != target.x {
            p.x += (target.x - p.x).signum();
        } else if p.y != target.y {
            p.y += (target.y - p.y).signum();
        }
        p.x = p.x.clamp(0, Pos::WORLD_SIZE - 1);
        p.y = p.y.clamp(0, Pos::WORLD_SIZE - 1);
        p
    }
}

/// A movement order the HTN issues: walk to `target`. The movement system
/// steps one tile per tick while the survivor hasn't arrived. `Position` is
/// *never* an HTN effect — the world moves at walking speed, not planning
/// speed.
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
struct Travel {
    target: Pos,
}

/// Whether the survivor has arrived at their [`Travel`] target. Owned by the
/// movement system in reality (recomputed from the real position every tick);
/// the planner *simulates* it as an effect of `move_to_item` so the plan can
/// validate its later steps. The driver's re-validation drops the plan every
/// tick until the walk has actually completed — the CDDA drift-replan loop.
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
struct Arrived(pub bool);

/// The item the survivor is currently acquiring (set by the planner from the
/// recipe data).
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
struct Focus(ItemKind);

/// What the survivor wants to craft.
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
struct CraftGoal(ItemKind);

/// The planner-facing summary of everything in the survivor's pockets,
/// derived from the relationship graph by [`sync_pockets`] each tick.
#[derive(Component, Clone, Debug, Default)]
struct PocketContents(HashMap<ItemKind, u32>);

impl PocketContents {
    fn count(&self, kind: ItemKind) -> u32 {
        self.0.get(&kind).copied().unwrap_or(0)
    }
    fn add(&mut self, kind: ItemKind) {
        *self.0.entry(kind).or_insert(0) += 1;
    }
    fn remove(&mut self, kind: ItemKind) {
        if let Some(n) = self.0.get_mut(&kind) {
            *n = n.saturating_sub(1);
        }
    }
}

/// What the survivor can see lying on the ground (refreshed each tick by
/// [`perception`]). This is the agent's knowledge of the world — the only
/// part of the world the planner can simulate.
#[derive(Component, Clone, Debug, Default)]
struct GroundKnowledge(Vec<(ItemKind, Pos)>);

impl GroundKnowledge {
    fn pos_of(&self, kind: ItemKind) -> Option<Pos> {
        self.0.iter().find(|(k, _)| *k == kind).map(|(_, p)| *p)
    }
    fn contains(&self, kind: ItemKind) -> bool {
        self.pos_of(kind).is_some()
    }
    fn remove(&mut self, kind: ItemKind) {
        self.0.retain(|(k, _)| *k != kind);
    }
}

/// Crafting knowledge — pure data the domain reads.
#[derive(Component, Clone, Debug, Default)]
struct RecipeBook(Vec<Recipe>);

#[derive(Clone, Debug)]
struct Recipe {
    output: ItemKind,
    inputs: Vec<ItemKind>,
}

impl RecipeBook {
    fn recipe_for(&self, output: ItemKind) -> Option<&Recipe> {
        self.0.iter().find(|r| r.output == output)
    }
    fn inputs_satisfied(&self, output: ItemKind, pockets: &PocketContents) -> bool {
        self.recipe_for(output)
            .is_some_and(|r| r.inputs.iter().all(|i| pockets.count(*i) > 0))
    }
    fn first_missing(&self, output: ItemKind, pockets: &PocketContents) -> Option<ItemKind> {
        self.recipe_for(output)?
            .inputs
            .iter()
            .copied()
            .find(|i| pockets.count(*i) == 0)
    }
}

// ---------------------------------------------------------------------------
// World model — relationships (the pocket system)
// ---------------------------------------------------------------------------

/// The survivor wears this clothing.
#[derive(Component)]
#[relationship(relationship_target = Wearing)]
struct WornBy(#[entities] Entity);

/// Relationship target on the survivor: everything worn.
#[derive(Component)]
#[relationship_target(relationship = WornBy)]
struct Wearing(Vec<Entity>);

/// An item sits in this clothing entity's pocket.
#[derive(Component)]
#[relationship(relationship_target = Pockets)]
struct InPocket(#[entities] Entity);

/// Relationship target on clothing: the items in its pockets.
#[derive(Component)]
#[relationship_target(relationship = InPocket)]
struct Pockets(Vec<Entity>);

#[derive(Component)]
struct Jacket;
#[derive(Component)]
struct OnGround;

// ---------------------------------------------------------------------------
// World model — resources and intent markers
// ---------------------------------------------------------------------------

/// The 2D spatial index: ground-item positions → entities, rebuilt each tick.
#[derive(Resource, Default)]
struct SpatialIndex(HashMap<Pos, Entity>);

/// Intent dispatched by the HTN `pick_up` action; realized by [`pick_up_items`].
#[derive(Component)]
struct PickupRequest;
/// Intent dispatched by the HTN `do_craft` action; realized by [`craft_items`].
#[derive(Component)]
struct CraftRequest;

// ---------------------------------------------------------------------------
// Game systems (the world's own logic — no HTN knowledge)
// ---------------------------------------------------------------------------

/// Refresh every survivor's ground knowledge from the real world.
fn perception(
    ground: Query<(&Pos, &ItemKind), With<OnGround>>,
    mut survivors: Query<&mut GroundKnowledge>,
) {
    let items: Vec<(ItemKind, Pos)> = ground.iter().map(|(p, k)| (*k, *p)).collect();
    for mut knowledge in &mut survivors {
        knowledge.0 = items.clone();
    }
}

/// Derive the planner-facing pocket summary from the relationship graph.
/// The graph is the truth; this summary is what the planner simulates.
fn sync_pockets(
    mut survivors: Query<(&Wearing, &mut PocketContents)>,
    clothing: Query<&Pockets>,
    items: Query<&ItemKind>,
) {
    for (wearing, mut contents) in &mut survivors {
        let mut map = HashMap::new();
        for &cloth in &wearing.0 {
            let Ok(pockets) = clothing.get(cloth) else {
                continue;
            };
            for &item in &pockets.0 {
                if let Ok(kind) = items.get(item) {
                    *map.entry(*kind).or_insert(0) += 1;
                }
            }
        }
        contents.0 = map;
    }
}

/// Execute movement orders one tile per tick (CDDA-style grid walking), and
/// recompute arrival from the real position — the planner's simulated
/// `Arrived` is corrected here every tick.
fn movement(mut survivors: Query<(&mut Pos, &Travel, &mut Arrived)>) {
    for (mut pos, travel, mut arrived) in &mut survivors {
        if *pos != travel.target {
            let next = pos.stepped_toward(travel.target);
            pos.x = next.x;
            pos.y = next.y;
        }
        arrived.0 = *pos == travel.target;
    }
}

/// Realize pickup intents against the spatial index and the pocket
/// relationships: the ground item entity despawns and a new item entity
/// appears in the survivor's first pocket.
fn pick_up_items(
    mut commands: Commands,
    survivors: Query<(Entity, &Pos, &Focus, &Wearing), With<PickupRequest>>,
    index: Res<SpatialIndex>,
) {
    for (survivor, pos, focus, wearing) in &survivors {
        if let Some(&item) = index.0.get(pos) {
            commands.entity(item).despawn();
            if let Some(&pocket) = wearing.0.first() {
                commands.spawn((focus.0, InPocket(pocket)));
            }
        }
        commands.entity(survivor).remove::<PickupRequest>();
    }
}

/// Realize craft intents: consume one input item entity per recipe input
/// from the pockets (via the relationship graph) and spawn the output into
/// the first pocket.
fn craft_items(
    mut commands: Commands,
    survivors: Query<(Entity, &CraftGoal, &RecipeBook, &Wearing), With<CraftRequest>>,
    clothing: Query<&Pockets>,
    items: Query<&ItemKind>,
) {
    for (survivor, goal, book, wearing) in &survivors {
        if let Some(recipe) = book.recipe_for(goal.0) {
            let mut needed: HashMap<ItemKind, u32> =
                recipe
                    .inputs
                    .iter()
                    .copied()
                    .fold(HashMap::new(), |mut m, i| {
                        *m.entry(i).or_insert(0) += 1;
                        m
                    });
            let mut consumed = Vec::new();
            'outer: for &cloth in &wearing.0 {
                let Ok(pockets) = clothing.get(cloth) else {
                    continue;
                };
                for &item in &pockets.0 {
                    let Ok(kind) = items.get(item) else {
                        continue;
                    };
                    if needed.get(kind).copied().unwrap_or(0) > 0 {
                        *needed.get_mut(kind).expect("checked above") -= 1;
                        consumed.push(item);
                        if needed.values().all(|n| *n == 0) {
                            break 'outer;
                        }
                    }
                }
            }
            if needed.values().all(|n| *n == 0) {
                for item in consumed {
                    commands.entity(item).despawn();
                }
                if let Some(&pocket) = wearing.0.first() {
                    commands.spawn((goal.0, InPocket(pocket)));
                }
            }
        }
        commands.entity(survivor).remove::<CraftRequest>();
    }
}

/// Rebuild the spatial index from the ground items.
fn update_spatial_index(
    ground: Query<(Entity, &Pos), With<OnGround>>,
    mut index: ResMut<SpatialIndex>,
) {
    index.0.clear();
    for (entity, pos) in &ground {
        index.0.insert(*pos, entity);
    }
}

// ---------------------------------------------------------------------------
// The HTN domain — generic over the recipe data
// ---------------------------------------------------------------------------

/// Root: loop until the craft goal sits in a pocket.
fn behave(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|pockets: &PocketContents, goal: &CraftGoal| pockets.count(goal.0) > 0); // terminal: the goal is crafted — done
    task.branch()
        .then(ensure_inputs)
        .then(do_craft)
        .then(behave);
}

/// Loop until every input of the goal recipe is in a pocket.
fn ensure_inputs(task: &mut TaskBuilder) {
    task.branch().precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    ); // terminal: inputs ready
    task.branch().then(acquire_missing).then(ensure_inputs);
}

/// Loop until the currently-missing input is acquired.
fn acquire_missing(task: &mut TaskBuilder) {
    task.branch().precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    ); // terminal: nothing missing
    task.branch()
        .then(select_missing)
        .then(acquire_selected)
        .then(acquire_missing);
}

/// Pick the first recipe input missing from the pockets (data-driven: the
/// recipe comes from the component, not the domain).
fn select_missing(task: &mut TaskBuilder) {
    task.effect(
        |focus: &mut Focus,
         pockets: &mut PocketContents,
         goal: &mut CraftGoal,
         book: &mut RecipeBook| {
            if let Some(missing) = book.first_missing(goal.0, pockets) {
                focus.0 = missing;
            }
        },
    );
}

/// Acquire the focused item: terminal if carried, otherwise walk to it and
/// pick it up.
fn acquire_selected(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|focus: &Focus, pockets: &PocketContents| pockets.count(focus.0) > 0); // terminal: already carrying it
    task.branch()
        .precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .then(move_to_item)
        .then(pick_up);
}

/// Order the walk and *simulate arrival* so the plan's later steps validate.
/// The agent's `Position` is deliberately never an effect — the world moves
/// at walking speed, and the driver's re-validation drops the plan every
/// tick until the movement system has actually arrived.
fn move_to_item(task: &mut TaskBuilder) {
    task.precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .effect(
            |travel: &mut Travel,
             arrived: &mut Arrived,
             ground: &mut GroundKnowledge,
             focus: &mut Focus| {
                if let Some(target) = ground.pos_of(focus.0) {
                    travel.target = target;
                    arrived.0 = true;
                }
            },
        );
}

/// Pick the focused item up once the walk has finished. The action dispatches
/// an intent; the pickup system realizes it against the spatial index and
/// pocket relationships.
fn pick_up(task: &mut TaskBuilder) {
    task.precondition(|arrived: &Arrived| arrived.0)
        .precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .effect(
            |pockets: &mut PocketContents, ground: &mut GroundKnowledge, focus: &mut Focus| {
                pockets.add(focus.0);
                ground.remove(focus.0);
            },
        )
        .action(|cmds: &mut EntityCommands| {
            cmds.insert(PickupRequest);
        });
}

/// Craft the goal from the pocket contents (simulated; the crafting system
/// realizes it on the relationship graph).
fn do_craft(task: &mut TaskBuilder) {
    task.precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    )
    .effect(
        |pockets: &mut PocketContents, goal: &mut CraftGoal, book: &mut RecipeBook| {
            if let Some(recipe) = book.recipe_for(goal.0) {
                for input in &recipe.inputs {
                    pockets.remove(*input);
                }
                pockets.add(goal.0);
            }
        },
    )
    .action(|cmds: &mut EntityCommands| {
        cmds.insert(CraftRequest);
    });
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

/// The survivor's worn clothing (the jacket) contains an item of `kind`.
fn pocket_holds(world: &mut World, survivor: Entity, kind: ItemKind) -> Option<Entity> {
    let mut items = world.query::<(Entity, &ItemKind, &InPocket)>();
    for (entity, item_kind, in_pocket) in items.iter(world) {
        if *item_kind == kind {
            let clothing = in_pocket.0;
            if world
                .get::<WornBy>(clothing)
                .is_some_and(|w| w.0 == survivor)
            {
                return Some(entity);
            }
        }
    }
    None
}

#[test]
fn survivor_walks_picks_up_and_crafts_a_spear() {
    let mut world = World::new();

    // --- Data-driven game content ------------------------------------------
    let recipes = RecipeBook(vec![Recipe {
        output: ItemKind::Spear,
        inputs: vec![ItemKind::Stick, ItemKind::Rag],
    }]);

    // --- Domain + driver config ---------------------------------------------
    world.insert_resource(HtnConfig::new(
        HtnDomain::from_root(behave)
            .build()
            .expect("well-formed domain"),
    ));

    // --- The survivor: wears a jacket holding a stick -----------------------
    let survivor = world
        .spawn((
            Pos { x: 0, y: 0 },
            Travel::default(),
            Arrived::default(),
            Focus::default(),
            CraftGoal(ItemKind::Spear),
            PocketContents::default(),
            GroundKnowledge::default(),
            recipes,
            HtnAgent::default(),
        ))
        .id();
    let jacket = world.spawn((Jacket, WornBy(survivor))).id();
    world.spawn((ItemKind::Stick, InPocket(jacket)));

    // --- The rag lies on the ground across the map ---------------------------
    world.spawn((ItemKind::Rag, Pos { x: 4, y: 4 }, OnGround));

    world.insert_resource(SpatialIndex::default());

    // --- The tick: perceive → sync → plan/execute → world systems -----------
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            perception,
            sync_pockets,
            htn_ai_system, // exclusive: plans, validates, executes one step
            movement,
            pick_up_items,
            craft_items,
            update_spatial_index,
        )
            .chain(),
    );

    // --- Run until the spear is crafted into a pocket ------------------------
    let mut ticks = 0;
    while pocket_holds(&mut world, survivor, ItemKind::Spear).is_none() {
        ticks += 1;
        assert!(
            ticks <= 100,
            "the survivor never crafted a spear (100 ticks)"
        );
        schedule.run(&mut world);
    }
    // One more tick so the sync/perception systems settle the summaries.
    schedule.run(&mut world);

    // --- The spear exists, in a pocket of the survivor's clothing ------------
    let spear = pocket_holds(&mut world, survivor, ItemKind::Spear)
        .expect("the spear was crafted into a pocket");
    assert!(
        world.get::<ItemKind>(spear).is_some(),
        "the spear is a real item entity"
    );

    // --- The inputs were consumed --------------------------------------------
    let mut remaining = world.query::<(Entity, &ItemKind)>();
    let kinds: Vec<ItemKind> = remaining.iter(&world).map(|(_, k)| *k).collect();
    assert_eq!(
        kinds.iter().filter(|k| **k == ItemKind::Stick).count(),
        0,
        "the stick was consumed by crafting"
    );
    assert_eq!(
        kinds.iter().filter(|k| **k == ItemKind::Rag).count(),
        0,
        "the rag was consumed by crafting"
    );

    // --- The rag is off the ground and out of the spatial index --------------
    let index = world.resource::<SpatialIndex>();
    assert!(
        !index
            .0
            .values()
            .any(|e| world.get::<ItemKind>(*e) == Some(&ItemKind::Rag)),
        "the rag no longer occupies the spatial index"
    );

    // --- The survivor actually walked to the rag ------------------------------
    assert_eq!(
        world.get::<Pos>(survivor),
        Some(&Pos { x: 4, y: 4 }),
        "the survivor walked across the map to the rag"
    );

    // --- The derived pocket summary agrees with the relationship graph -------
    let contents = world.get::<PocketContents>(survivor).unwrap();
    assert_eq!(contents.count(ItemKind::Spear), 1);
    assert_eq!(contents.count(ItemKind::Stick), 0);
    assert_eq!(contents.count(ItemKind::Rag), 0);

    // --- The agent is idle: the goal branch is terminal -----------------------
    let agent = world.get::<HtnAgent>(survivor).unwrap();
    assert!(
        agent.plan.is_none(),
        "the goal is satisfied; no further plan"
    );

    // Silence the unused-import lint for Duration in case the tick budget
    // grows a wall-clock guard.
    let _ = std::time::Duration::from_secs(0);
}
