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
use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::TaskBuilder;
use bevy_bhtn::{DomainBuilder, HtnDomain, HtnError, TaskFn};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;
use bevy_ecs::system::EntityCommands;

// ---------------------------------------------------------------------------
// World model — components
// ---------------------------------------------------------------------------

/// Every item kind in the game. Data, not code: recipes reference these.
/// `Scrap`, `Berry` and `Bottle` are irrelevant to the test's goal — they
/// exist to prove the planner stays focused on the goal recipe.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
enum ItemKind {
    #[default]
    Stick,
    Rag,
    Spear,
    Scrap,
    Berry,
    Bottle,
    Bandage,
    Torch,
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

/// Whether the survivor has slung their pack. The GTN-sharing scenario's
/// once-only preparation: referenced once per fetch trip by the task graph,
/// executed exactly once per plan (and, thanks to the gate, exactly once per
/// episode across the driver's replans).
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
struct Packed(pub bool);

/// Whether the survivor's hands are free. The GTN-insertion scenario's
/// unmodeled precondition: picking items up fills the hands, and crafting
/// needs a free hand — the domain author never models putting things down;
/// the insertion compilation weaves `free_hands` into the plan gap.
#[derive(Component, Clone, Copy, Debug, PartialEq, Default)]
struct HandsFree(pub bool);

/// The crafts the survivor wants, **in order, with multiplicity**: `[Spear,
/// Torch]` crafts two different items; `[Bandage, Bandage]` crafts two of
/// one. The queue scenarios' planner-facing goal — a multiset, so the
/// terminal is "every kind crafted its listed number of times".
#[derive(Component, Clone, Debug, Default)]
struct CraftQueue(pub Vec<ItemKind>);

impl CraftQueue {
    /// How many times `kind` is listed.
    fn occurrences(&self, kind: ItemKind) -> u32 {
        self.0.iter().filter(|k| **k == kind).count() as u32
    }
    /// Whether every listed craft has been crafted.
    fn all_crafted(&self, pockets: &PocketContents) -> bool {
        self.0
            .iter()
            .all(|k| pockets.count(*k) >= self.occurrences(*k))
    }
    /// Whether the pockets hold the inputs for **every remaining craft** —
    /// the count-aware fetch gate: a queue of two bandages is only "fetched"
    /// once two scrap AND two berry are carried.
    fn fetch_done(&self, pockets: &PocketContents, book: &RecipeBook) -> bool {
        self.0.iter().all(|k| {
            let remaining = self.occurrences(*k) - pockets.count(*k);
            remaining == 0 || book.inputs_satisfied_times(*k, pockets, remaining)
        })
    }
}

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
        // One entry per pickup: duplicates of a kind are independent items,
        // and the agent's knowledge keeps the ones still lying on the ground.
        if let Some(i) = self.0.iter().position(|(k, _)| *k == kind) {
            self.0.remove(i);
        }
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
    /// Whether the pockets hold enough of every input to craft `output`
    /// `times` more times (the queue scenarios' count-aware fetch gate).
    fn inputs_satisfied_times(
        &self,
        output: ItemKind,
        pockets: &PocketContents,
        times: u32,
    ) -> bool {
        self.recipe_for(output)
            .is_some_and(|r| r.inputs.iter().all(|i| pockets.count(*i) >= times))
    }
    /// The first input the pockets hold fewer than `times` of.
    fn first_missing_times(
        &self,
        output: ItemKind,
        pockets: &PocketContents,
        times: u32,
    ) -> Option<ItemKind> {
        self.recipe_for(output)?
            .inputs
            .iter()
            .copied()
            .find(|i| pockets.count(*i) < times)
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
///
/// The first branch is an **empty terminal branch** — `.branch().precondition(…)`
/// with no `.then` — the idiom for a "done" method: when its precondition
/// holds, the task decomposes to nothing and the loop stops.
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
        |focus: &mut Focus, pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
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
             ground: &GroundKnowledge,
             focus: &Focus| {
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
            |pockets: &mut PocketContents, ground: &mut GroundKnowledge, focus: &Focus| {
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
        |pockets: &mut PocketContents, goal: &CraftGoal, book: &RecipeBook| {
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
// GTN scenarios — task sharing + task insertion, end to end
// ---------------------------------------------------------------------------

/// The once-only preparation for every fetch trip: sling the pack. Gated by
/// `!Packed` — the "done once" pattern the sharing contract requires. The
/// task graph references it once per trip; the sharing compilation makes the
/// plan execute it once per episode.
fn sling_pack(task: &mut TaskBuilder) {
    task.precondition(|p: &Packed| !p.0)
        .effect(|p: &mut Packed| p.0 = true);
}

/// The sharing scenario's loop chain: identical shape to the plain `behave`
/// chain, but every fetch trip references [`sling_pack`]. Without the
/// compilation the second trip's reference is a **failed branch** (the gate
/// has closed and there is no no-op path); with it, the wrapper's `done`
/// method degrades the repeat reference to zero steps.
fn behave_shared(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|pockets: &PocketContents, goal: &CraftGoal| pockets.count(goal.0) > 0);
    task.branch()
        .then(ensure_shared)
        .then(do_craft)
        .then(behave_shared);
}
fn ensure_shared(task: &mut TaskBuilder) {
    task.branch().precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    );
    task.branch().then(acquire_shared).then(ensure_shared);
}
fn acquire_shared(task: &mut TaskBuilder) {
    task.branch().precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    );
    task.branch()
        .then(select_missing)
        .then(acquire_shared_selected)
        .then(acquire_shared);
}
fn acquire_shared_selected(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|focus: &Focus, pockets: &PocketContents| pockets.count(focus.0) > 0);
    task.branch()
        .precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .then(sling_pack)
        .then(move_to_item)
        .then(pick_up);
}

/// Pick up, but the hands are now full — the side effect the task graph
/// deliberately does not model. The insertion compilation must weave the
/// unmodeled repair ([`free_hands`]) into the plan gap.
fn pick_up_hands(task: &mut TaskBuilder) {
    task.precondition(|arrived: &Arrived| arrived.0)
        .precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .effect(
            |pockets: &mut PocketContents,
             ground: &mut GroundKnowledge,
             hands: &mut HandsFree,
             focus: &Focus| {
                pockets.add(focus.0);
                ground.remove(focus.0);
                hands.0 = false;
            },
        )
        .action(|cmds: &mut EntityCommands| {
            cmds.insert(PickupRequest);
        });
}

/// Crafting needs a free hand — and [`pick_up_hands`] just took one.
fn do_craft_free(task: &mut TaskBuilder) {
    task.precondition(|hands: &HandsFree| hands.0)
        .precondition(
            |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
                book.inputs_satisfied(goal.0, pockets)
            },
        )
        .effect(
            |pockets: &mut PocketContents, goal: &CraftGoal, book: &RecipeBook| {
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

/// The unmodeled repair action: reachable only through insertion.
fn free_hands(task: &mut TaskBuilder) {
    task.precondition(|h: &HandsFree| !h.0)
        .effect(|h: &mut HandsFree| h.0 = true);
}

/// The insertion scenario's loop chain: [`pick_up_hands`] fills the hands,
/// [`do_craft_free`] demands them free, and no body ever mentions putting
/// things down.
fn behave_repair(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|pockets: &PocketContents, goal: &CraftGoal| pockets.count(goal.0) > 0);
    task.branch()
        .then(ensure_repair)
        .then(do_craft_free)
        .then(behave_repair);
}
fn ensure_repair(task: &mut TaskBuilder) {
    task.branch().precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    );
    task.branch().then(acquire_repair).then(ensure_repair);
}
fn acquire_repair(task: &mut TaskBuilder) {
    task.branch().precondition(
        |pockets: &PocketContents, goal: &CraftGoal, book: &RecipeBook| {
            book.inputs_satisfied(goal.0, pockets)
        },
    );
    task.branch()
        .then(select_missing)
        .then(acquire_repair_selected)
        .then(acquire_repair);
}
fn acquire_repair_selected(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|focus: &Focus, pockets: &PocketContents| pockets.count(focus.0) > 0);
    task.branch()
        .precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .then(move_to_item)
        .then(pick_up_hands);
}

// ---------------------------------------------------------------------------
// Queue scenarios — multiple crafts, proactive fetching
// ---------------------------------------------------------------------------

/// The queue root: fetch **everything still missing across the whole queue**
/// (proactive — one plan covers both crafts' pickups), then craft each queue
/// item in order, then loop until the whole multiset is crafted.
fn queue_behave(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|queue: &CraftQueue, pockets: &PocketContents| queue.all_crafted(pockets));
    task.branch()
        .then(queue_fetch_loop)
        .then(queue_craft_loop)
        .then(queue_behave);
}

/// Fetch until the pockets hold the inputs for **every remaining craft** in
/// the queue (count-aware: two bandages means two scrap and two berry). The
/// scan is queue-order/recipe-order — deterministic, not distance-optimized.
fn queue_fetch_loop(task: &mut TaskBuilder) {
    task.branch().precondition(
        |queue: &CraftQueue, pockets: &PocketContents, book: &RecipeBook| {
            queue.fetch_done(pockets, book)
        },
    );
    task.branch()
        .then(queue_fetch_next)
        .then(queue_acquire_selected)
        .then(queue_fetch_loop);
}

/// Focus the first missing input of the first queue item whose remaining
/// crafts aren't fully stocked (queue order, then recipe-input order). The
/// precondition keeps backtracked retries honest: once everything is stocked,
/// re-entering this branch fails here instead of committing endless no-op
/// selects.
fn queue_fetch_next(task: &mut TaskBuilder) {
    task.precondition(
        |queue: &CraftQueue, pockets: &PocketContents, book: &RecipeBook| {
            !queue.fetch_done(pockets, book)
        },
    )
    .effect(
        |focus: &mut Focus, queue: &CraftQueue, pockets: &PocketContents, book: &RecipeBook| {
            for kind in &queue.0 {
                let remaining = queue.occurrences(*kind) - pockets.count(*kind);
                if remaining > 0 {
                    if let Some(missing) = book.first_missing_times(*kind, pockets, remaining) {
                        focus.0 = missing;
                        return;
                    }
                }
            }
        },
    );
}

/// Acquire the focused input, count-aware: terminal only when the pockets
/// hold enough of it for **every remaining craft that needs it** (two
/// bandages need two scrap — one carried is not done). Otherwise walk to a
/// ground copy and pick it up.
fn queue_acquire_selected(task: &mut TaskBuilder) {
    task.branch().precondition(
        |focus: &Focus, queue: &CraftQueue, pockets: &PocketContents, book: &RecipeBook| {
            let need = queue
                .0
                .iter()
                .filter_map(|k| {
                    let rem = queue.occurrences(*k) - pockets.count(*k);
                    (rem > 0
                        && book
                            .recipe_for(*k)
                            .is_some_and(|r| r.inputs.contains(&focus.0)))
                    .then_some(rem)
                })
                .max()
                .unwrap_or(0);
            pockets.count(focus.0) >= need
        },
    );
    task.branch()
        .precondition(|focus: &Focus, ground: &GroundKnowledge| ground.contains(focus.0))
        .then(move_to_item)
        .then(pick_up);
}

/// Craft each queue item in order, once per listing.
fn queue_craft_loop(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|queue: &CraftQueue, pockets: &PocketContents| queue.all_crafted(pockets));
    task.branch()
        .then(queue_craft_next)
        .then(do_craft)
        .then(queue_craft_loop);
}

/// Focus the first queue item that is crafted fewer times than listed *and*
/// whose inputs are currently satisfied. The precondition is what keeps the
/// plan clean: once a craft consumes its inputs and no other queue item is
/// craftable, this branch fails *here* instead of committing a no-op select
/// followed by a failing craft — the plan ends after the last doable craft
/// and the driver replans for the next wave.
fn queue_craft_next(task: &mut TaskBuilder) {
    task.precondition(
        |queue: &CraftQueue, pockets: &PocketContents, book: &RecipeBook| {
            queue.0.iter().any(|k| {
                pockets.count(*k) < queue.occurrences(*k) && book.inputs_satisfied(*k, pockets)
            })
        },
    )
    .effect(
        |goal: &mut CraftGoal, queue: &CraftQueue, pockets: &PocketContents, book: &RecipeBook| {
            for kind in &queue.0 {
                let needed = queue.occurrences(*kind);
                if pockets.count(*kind) < needed && book.inputs_satisfied(*kind, pockets) {
                    goal.0 = *kind;
                    return;
                }
            }
        },
    );
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

// ---------------------------------------------------------------------------
// The shared world harness
// ---------------------------------------------------------------------------

/// The shared CDDA-like world: a survivor at (0,0) wearing a jacket that
/// holds a stick and a bottle; rag/scrap/berry/bottle scattered on the
/// ground; the full recipe book; the tick schedule
/// (perceive → sync → plan/execute → movement → pickup → craft → index).
struct CddaWorld {
    world: World,
    schedule: Schedule,
    survivor: Entity,
    jacket: Entity,
}

/// Where every ground item lies (shared by all scenarios).
const GROUND_ITEMS: [(ItemKind, Pos); 4] = [
    (ItemKind::Rag, Pos { x: 4, y: 4 }),
    (ItemKind::Scrap, Pos { x: 2, y: 2 }),
    (ItemKind::Berry, Pos { x: 0, y: 3 }),
    (ItemKind::Bottle, Pos { x: 3, y: 1 }),
];

impl CddaWorld {
    fn new(goal: ItemKind) -> Self {
        Self::new_with(goal, vec![goal], GROUND_ITEMS.as_slice(), behave, |b| b)
    }

    /// Build the world with a different root, craft queue, ground layout,
    /// and customized domain (the GTN scenarios add `share_task` /
    /// `with_insertion` / `insertable`).
    fn new_with<R: TaskFn>(
        goal: ItemKind,
        queue: Vec<ItemKind>,
        ground: &[(ItemKind, Pos)],
        root: R,
        domain: impl FnOnce(DomainBuilder) -> DomainBuilder,
    ) -> Self {
        let mut world = World::new();

        // Data-driven game content: the full recipe book. Whichever recipe
        // the goal names is the only one the planner may pursue.
        let recipes = RecipeBook(vec![
            Recipe {
                output: ItemKind::Spear,
                inputs: vec![ItemKind::Stick, ItemKind::Rag],
            },
            Recipe {
                output: ItemKind::Bandage,
                inputs: vec![ItemKind::Scrap, ItemKind::Berry],
            },
            Recipe {
                output: ItemKind::Torch,
                inputs: vec![ItemKind::Bottle, ItemKind::Scrap],
            },
        ]);

        world.insert_resource(HtnConfig::new(
            domain(HtnDomain::from_root(root))
                .build()
                .expect("well-formed domain"),
        ));

        let survivor = world
            .spawn((
                Pos { x: 0, y: 0 },
                Travel::default(),
                Arrived::default(),
                Focus::default(),
                CraftGoal(goal),
                CraftQueue(queue),
                PocketContents::default(),
                GroundKnowledge::default(),
                Packed::default(),
                HandsFree(true),
                recipes,
                HtnAgent::default(),
            ))
            .id();
        let jacket = world.spawn((Jacket, WornBy(survivor))).id();
        // Carried from the start: a spear input (stick) and an irrelevant
        // bottle the planner must leave alone unless the goal needs it.
        world.spawn((ItemKind::Stick, InPocket(jacket)));
        world.spawn((ItemKind::Bottle, InPocket(jacket)));

        for (kind, pos) in ground {
            world.spawn((*kind, *pos, OnGround));
        }

        world.insert_resource(SpatialIndex::default());

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

        Self {
            world,
            schedule,
            survivor,
            jacket,
        }
    }

    /// Run ticks until `kind` sits in a pocket of the survivor's clothing;
    /// returns the tick count. The simulation is fully deterministic (no
    /// randomness, no hash-iteration-order dependence), so tests pin the
    /// exact number.
    fn run_until_crafted(&mut self, kind: ItemKind) -> usize {
        let mut ticks = 0;
        while pocket_holds(&mut self.world, self.survivor, kind).is_none() {
            ticks += 1;
            assert!(
                ticks <= 100,
                "the survivor never crafted {kind:?} (100 ticks)"
            );
            self.schedule.run(&mut self.world);
        }
        // One more tick so the sync/perception systems settle the summaries.
        self.schedule.run(&mut self.world);
        ticks
    }

    /// Every item kind still present in the world (pockets + ground).
    fn kinds(&mut self) -> Vec<ItemKind> {
        let mut q = self.world.query::<&ItemKind>();
        q.iter(&self.world).copied().collect()
    }

    fn ground_kind_at(&self, pos: Pos) -> Option<ItemKind> {
        self.world
            .resource::<SpatialIndex>()
            .0
            .get(&pos)
            .copied()
            .map(|e| {
                *self
                    .world
                    .get::<ItemKind>(e)
                    .expect("indexed entity has a kind")
            })
    }

    fn position(&self) -> Pos {
        *self
            .world
            .get::<Pos>(self.survivor)
            .expect("survivor alive")
    }

    fn pocket_count(&self, kind: ItemKind) -> u32 {
        self.world
            .get::<PocketContents>(self.survivor)
            .expect("survivor alive")
            .count(kind)
    }

    fn idle(&self) -> bool {
        self.world
            .get::<HtnAgent>(self.survivor)
            .expect("survivor alive")
            .plan
            .is_none()
    }

    /// The current plan's full step-program names (exact execution ordering).
    fn plan_names(&self) -> Vec<String> {
        self.world
            .get::<HtnAgent>(self.survivor)
            .and_then(|a| a.plan.as_ref())
            .map(|p| p.task_names().iter().map(|u| u.to_string()).collect())
            .unwrap_or_default()
    }

    /// Run exactly `n` ticks (for checkpointed ordering assertions).
    fn tick_n(&mut self, n: usize) {
        for _ in 0..n {
            self.schedule.run(&mut self.world);
        }
    }

    fn component<T: Component + Clone + Copy>(&self) -> T {
        *self.world.get::<T>(self.survivor).expect("survivor alive")
    }
}

// ---------------------------------------------------------------------------
// The scenarios
// ---------------------------------------------------------------------------

/// The survivor walks to the rag, picks it up, and crafts the spear. The
/// simulation is deterministic, so the tick count is pinned exactly. Verified
/// against a per-tick trace; the arithmetic:
///
/// 1 (plan + select_missing) + 8 (walk — one tile per tick, the movement
/// system runs every tick while the driver interleaves move-ticks with
/// drift-fail and replan ticks) + 1 (replan + select) + 1 (move re-issued,
/// no-op walk) + 1 (pick_up: arrival confirmed, rag pocketed) + 1 (do_craft)
/// = 13.
#[test]
fn survivor_walks_picks_up_and_crafts_a_spear() {
    let mut sim = CddaWorld::new(ItemKind::Spear);
    let ticks = sim.run_until_crafted(ItemKind::Spear);
    assert_eq!(ticks, 13, "deterministic tick count for the spear run");

    // The spear exists, in a pocket of the survivor's jacket (the first
    // worn clothing — the pocket the pickup/craft systems target).
    let spear_entity =
        pocket_holds(&mut sim.world, sim.survivor, ItemKind::Spear).expect("spear in a pocket");
    assert!(sim
        .world
        .get::<Pockets>(sim.jacket)
        .is_some_and(|p| p.0.contains(&spear_entity)));

    // The inputs were consumed.
    let kinds = sim.kinds();
    assert_eq!(kinds.iter().filter(|k| **k == ItemKind::Stick).count(), 0);
    assert_eq!(kinds.iter().filter(|k| **k == ItemKind::Rag).count(), 0);

    // The rag is off the ground; every distractor is exactly where it was.
    assert_eq!(sim.ground_kind_at(Pos { x: 4, y: 4 }), None);
    for (kind, pos) in GROUND_ITEMS.iter().skip(1) {
        assert_eq!(
            sim.ground_kind_at(*pos),
            Some(*kind),
            "{kind:?} still lies where it was placed"
        );
    }
    // No irrelevant recipe was ever crafted.
    assert_eq!(
        kinds
            .iter()
            .filter(|k| matches!(**k, ItemKind::Bandage | ItemKind::Torch))
            .count(),
        0,
        "no irrelevant recipe was crafted"
    );

    // The survivor walked across the map to the rag.
    assert_eq!(sim.position(), Pos { x: 4, y: 4 });

    // The pocket summary agrees with the relationship graph: spear crafted,
    // stick consumed, the irrelevant bottle untouched.
    assert_eq!(sim.pocket_count(ItemKind::Spear), 1);
    assert_eq!(sim.pocket_count(ItemKind::Stick), 0);
    assert_eq!(sim.pocket_count(ItemKind::Rag), 0);
    assert_eq!(sim.pocket_count(ItemKind::Bottle), 1);
    assert_eq!(sim.pocket_count(ItemKind::Scrap), 0);
    assert_eq!(sim.pocket_count(ItemKind::Berry), 0);

    // The agent is idle: the goal branch is terminal.
    assert!(sim.idle());
}

/// Crafting a bandage needs two ground items in sequence (scrap, then berry
/// — recipe-input order): two full walk-and-fetch trips, still pinned
/// exactly, and still no distraction toward the other recipes. Verified
/// against a per-tick trace; the arithmetic:
///
/// trip 1 (scrap, 4 tiles): 1 (plan + select) + 4 (walk) + 1 (pick_up —
/// arrival lands on a move-tick, so pickup validates directly) = 6;
/// trip 2 (berry, 3 tiles): 1 (plan + select) + 3 (walk) + 1 (move re-issued)
/// + 1 (pick_up) + 1 (do_craft) = 7. Total 13.
#[test]
fn survivor_crafts_a_bandage_from_two_ground_items() {
    let mut sim = CddaWorld::new(ItemKind::Bandage);
    let ticks = sim.run_until_crafted(ItemKind::Bandage);
    assert_eq!(ticks, 13, "deterministic tick count for the bandage run");

    assert!(pocket_holds(&mut sim.world, sim.survivor, ItemKind::Bandage).is_some());

    // Both inputs were consumed; the spear path is untouched.
    let kinds = sim.kinds();
    assert_eq!(kinds.iter().filter(|k| **k == ItemKind::Scrap).count(), 0);
    assert_eq!(kinds.iter().filter(|k| **k == ItemKind::Berry).count(), 0);
    assert_eq!(kinds.iter().filter(|k| **k == ItemKind::Rag).count(), 1);
    assert_eq!(
        kinds.iter().filter(|k| **k == ItemKind::Spear).count(),
        0,
        "the spear recipe was never pursued"
    );

    // The survivor stands where the last fetch happened (the berry).
    assert_eq!(sim.position(), Pos { x: 0, y: 3 });

    assert_eq!(sim.pocket_count(ItemKind::Bandage), 1);
    assert_eq!(sim.pocket_count(ItemKind::Scrap), 0);
    assert_eq!(sim.pocket_count(ItemKind::Berry), 0);
    // The stick and bottle are irrelevant to a bandage and still carried.
    assert_eq!(sim.pocket_count(ItemKind::Stick), 1);
    assert_eq!(sim.pocket_count(ItemKind::Bottle), 1);

    assert!(sim.idle());
}

/// Crafting a torch needs a bottle and scrap — and the survivor already
/// carries a bottle: the planner must use the carried one and leave the
/// ground bottle lying at (3,1) untouched. One fetch only — the shortest
/// scenario. Verified against a per-tick trace; the arithmetic:
/// 1 (plan + select) + 4 (walk to the scrap) + 1 (pick_up) + 1 (do_craft) = 7.
#[test]
fn survivor_crafts_a_torch_using_the_carried_bottle() {
    let mut sim = CddaWorld::new(ItemKind::Torch);
    let ticks = sim.run_until_crafted(ItemKind::Torch);
    assert_eq!(ticks, 7, "deterministic tick count for the torch run");

    assert!(pocket_holds(&mut sim.world, sim.survivor, ItemKind::Torch).is_some());

    // The carried bottle was consumed; the ground bottle is untouched.
    assert_eq!(sim.pocket_count(ItemKind::Bottle), 0);
    assert_eq!(
        sim.ground_kind_at(Pos { x: 3, y: 1 }),
        Some(ItemKind::Bottle),
        "the ground bottle was never fetched — the carried one was used"
    );
    assert_eq!(kinds_count(&mut sim, ItemKind::Bottle), 1);

    // The scrap was fetched and consumed; the rag is irrelevant here.
    assert_eq!(kinds_count(&mut sim, ItemKind::Scrap), 0);
    assert_eq!(kinds_count(&mut sim, ItemKind::Rag), 1);

    // The survivor stands where the scrap was.
    assert_eq!(sim.position(), Pos { x: 2, y: 2 });

    assert_eq!(sim.pocket_count(ItemKind::Torch), 1);
    assert_eq!(sim.pocket_count(ItemKind::Stick), 1);

    assert!(sim.idle());
}

fn kinds_count(sim: &mut CddaWorld, kind: ItemKind) -> usize {
    sim.kinds().iter().filter(|k| **k == kind).count()
}

// ---------------------------------------------------------------------------
// GTN scenarios — sharing and insertion against the live world
// ---------------------------------------------------------------------------

/// **Task sharing (Theorem 4.8), end to end.** A bandage needs two fetch
/// trips, and every trip's plan references [`sling_pack`]. Without the
/// compilation the second trip is unplannable: the gate has closed and the
/// reference is a failed branch, not a no-op. With it, the wrapper's `done`
/// method degrades the repeat reference to zero steps — `sling_pack`
/// executes exactly once per episode, and the driver's drift-replans take
/// `done` too (the committed `Packed` gate persists; only the marker
/// resets).
///
/// Verified against a per-tick trace; the arithmetic is the plain bandage
/// run (13) plus the one `sling_pack` step executed on the first trip only:
/// **14**.
#[test]
fn shared_pack_slings_once_across_two_fetch_trips() {
    let mut sim = CddaWorld::new_with(
        ItemKind::Bandage,
        vec![ItemKind::Bandage],
        GROUND_ITEMS.as_slice(),
        behave_shared,
        |b| b.share_task(sling_pack),
    );

    // Tick 1: plan + select_missing. The plan is pinned EXACTLY — both trips
    // in one total order, with `sling_pack` appearing once (the second
    // reference took the wrapper's empty `done` method).
    sim.tick_n(1);
    assert_eq!(
        sim.plan_names(),
        [
            "select_missing",
            "sling_pack",
            "move_to_item",
            "pick_up",
            "select_missing",
            "move_to_item",
            "pick_up",
            "do_craft",
        ],
        "the exact compiled ordering: one sling, both trips, then the craft"
    );
    // The pack is still on the survivor's back... i.e. not yet slung: the
    // select step ran, the sling step is next.
    assert!(!sim.component::<Packed>().0);

    // Tick 2 executes sling_pack; the world commits it.
    sim.tick_n(1);
    assert!(
        sim.component::<Packed>().0,
        "the pack was slung exactly here"
    );

    // Continue the episode, counting the two checkpoint ticks toward the
    // pinned total.
    let mut ticks = 2;
    while pocket_holds(&mut sim.world, sim.survivor, ItemKind::Bandage).is_none() {
        ticks += 1;
        assert!(ticks <= 100, "the survivor never crafted the bandage");
        sim.tick_n(1);
    }
    sim.tick_n(1); // settle tick, mirroring `run_until_crafted`
    assert_eq!(ticks, 14, "13-tick bandage run + the one shared sling step");

    assert!(pocket_holds(&mut sim.world, sim.survivor, ItemKind::Bandage).is_some());
    assert_eq!(sim.position(), Pos { x: 0, y: 3 });
    // Both inputs consumed, both fetched in recipe order (scrap first).
    assert_eq!(sim.pocket_count(ItemKind::Scrap), 0);
    assert_eq!(sim.pocket_count(ItemKind::Berry), 0);
    assert_eq!(sim.pocket_count(ItemKind::Bandage), 1);
    assert!(sim.idle());
}

/// **The sharing contract, negative control.** The same shared-loop domain
/// *without* the compilation: the second trip's `sling_pack` reference hits
/// the closed gate and the branch fails — the bandage is unplannable. This
/// is the failure mode the `done` method exists to absorb.

#[test]
fn without_sharing_the_second_trip_reference_fails_the_branch() {
    let domain = HtnDomain::from_root(behave_shared).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner
        .plan(behave_shared, &state)
        .expect_err("without sharing the chain is unsolvable");
    assert!(
        matches!(plan, HtnError::NoPlan),
        "the gated second trip kills the whole chain — and is reported as \
         NoPlan, not an empty Complete plan"
    );
}

/// **Task insertion, end to end.** [`pick_up_hands`] fills the hands and
/// [`do_craft_free`] demands a free one — but no method body ever mentions
/// putting things down. Without insertion the chain is unsatisfiable (the
/// spear is unplannable). With it, the search backtracks into the plan gap
/// and weaves [`free_hands`] — reachable *only* through insertion — exactly
/// between the pickup and the craft.
///
/// Verified against a per-tick trace; the arithmetic is the plain spear run
/// (13) plus the one inserted repair step: **14**.
#[test]
fn insertion_weaves_the_unmodeled_repair_step() {
    let mut sim = CddaWorld::new_with(
        ItemKind::Spear,
        vec![ItemKind::Spear],
        GROUND_ITEMS.as_slice(),
        behave_repair,
        |b| b.insertable(free_hands).with_insertion(),
    );

    // Tick 1: plan + select_missing. The repair sits EXACTLY between the
    // pickup and the craft — the defined insertion semantics.
    sim.tick_n(1);
    assert_eq!(
        sim.plan_names(),
        [
            "select_missing",
            "move_to_item",
            "pick_up_hands",
            "free_hands",
            "do_craft_free",
        ],
        "free_hands woven into the gap between pickup and craft"
    );
    // The hands are still free: the select step ran; the repair has not.
    assert!(sim.component::<HandsFree>().0);

    // Continue the episode, counting the checkpoint tick toward the pinned
    // total.
    let mut ticks = 1;
    while pocket_holds(&mut sim.world, sim.survivor, ItemKind::Spear).is_none() {
        ticks += 1;
        assert!(ticks <= 100, "the survivor never crafted the spear");
        sim.tick_n(1);
    }
    sim.tick_n(1); // settle tick, mirroring `run_until_crafted`
    assert_eq!(
        ticks, 14,
        "13-tick spear run + the one inserted repair step"
    );

    assert!(pocket_holds(&mut sim.world, sim.survivor, ItemKind::Spear).is_some());
    // The repair executed: hands freed again, and only the rag left the ground.
    assert!(sim.component::<HandsFree>().0);
    assert_eq!(sim.ground_kind_at(Pos { x: 4, y: 4 }), None);
    for (kind, pos) in GROUND_ITEMS.iter().skip(1) {
        assert_eq!(sim.ground_kind_at(*pos), Some(*kind));
    }
    assert_eq!(sim.pocket_count(ItemKind::Spear), 1);
    assert!(sim.idle());
}

/// **The insertion contract, negative control.** The same repair domain
/// *without* the compilation: the hands fill and nothing ever frees them —
/// the spear is unplannable (empty final plan, search exhausted).
#[test]
fn without_insertion_the_unmodeled_precondition_kills_the_chain() {
    let domain = HtnDomain::from_root(behave_repair).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner
        .plan(behave_repair, &state)
        .expect_err("without insertion nothing plans");
    assert!(
        matches!(plan, HtnError::NoPlan),
        "nothing frees the hands; nothing plans — reported as NoPlan, not an \
         empty Complete plan"
    );
}

/// **Two different crafts, one plan.** The queue is `[Spear, Torch]`; the
/// stick and bottle are carried, so the missing inputs are the rag (4,4) and
/// the scrap (2,2). The proactive fetch phase gathers **both** crafts'
/// inputs in one plan — in queue/recipe order, not distance order — and the
/// craft phase then crafts spear, torch. Exact plan ordering pinned at tick
/// 1; the survivor's final position pins the fetch order (the last fetch
/// was the scrap).
#[test]
fn queue_fetches_for_both_crafts_proactively_then_crafts_in_order() {
    let mut sim = CddaWorld::new_with(
        ItemKind::Spear,
        vec![ItemKind::Spear, ItemKind::Torch],
        GROUND_ITEMS.as_slice(),
        queue_behave,
        |b| b,
    );

    sim.tick_n(1);
    assert_eq!(
        sim.plan_names(),
        [
            "queue_fetch_next",
            "move_to_item",
            "pick_up",
            "queue_fetch_next",
            "move_to_item",
            "pick_up",
            "queue_craft_next",
            "do_craft",
            "queue_craft_next",
            "do_craft",
        ],
        "both crafts' fetches batched first, then both crafts in queue order"
    );

    let ticks = sim.run_until_crafted(ItemKind::Torch);
    assert_eq!(
        ticks, 21,
        "deterministic tick count for the two-craft queue run"
    );

    // Both crafts exist, in pockets of the jacket.
    assert!(pocket_holds(&mut sim.world, sim.survivor, ItemKind::Spear).is_some());
    assert!(pocket_holds(&mut sim.world, sim.survivor, ItemKind::Torch).is_some());
    // All four inputs consumed (stick and bottle were carried; the ground
    // bottle was never fetched — the carried one satisfied the torch).
    for kind in [ItemKind::Stick, ItemKind::Rag, ItemKind::Scrap] {
        assert_eq!(kinds_count(&mut sim, kind), 0, "{kind:?} consumed");
    }
    assert_eq!(
        kinds_count(&mut sim, ItemKind::Bottle),
        1,
        "the ground bottle was never fetched — the carried one was used"
    );
    assert_eq!(
        sim.ground_kind_at(Pos { x: 3, y: 1 }),
        Some(ItemKind::Bottle),
    );
    // The fetch order was rag-then-scrap (queue order): the survivor ends
    // where the LAST fetched item (scrap) lay.
    assert_eq!(sim.position(), Pos { x: 2, y: 2 });
    assert_eq!(sim.pocket_count(ItemKind::Spear), 1);
    assert_eq!(sim.pocket_count(ItemKind::Torch), 1);
    assert!(sim.idle());
}

/// **Two of the same item.** The queue is `[Bandage, Bandage]`; the fetch
/// phase is count-aware — it gathers **two scrap and two berry** proactively
/// (the carried stick/bottle prove carried stock counts) — and the craft
/// phase crafts both. The whole episode is one plan, pinned exactly.
#[test]
fn queue_crafts_two_of_an_item_in_one_count_aware_plan() {
    // Two bandages need two scrap and two berry: the queue scenario's own
    // ground layout doubles them up (the shared GROUND_ITEMS holds one of
    // each — there a second bandage is genuinely uncraftable, and the
    // exhausted search returns an empty plan).
    let ground = [
        (ItemKind::Rag, Pos { x: 4, y: 4 }),
        (ItemKind::Scrap, Pos { x: 2, y: 2 }),
        (ItemKind::Berry, Pos { x: 0, y: 3 }),
        (ItemKind::Bottle, Pos { x: 3, y: 1 }),
        (ItemKind::Scrap, Pos { x: 0, y: 1 }),
        (ItemKind::Berry, Pos { x: 4, y: 0 }),
    ];
    let mut sim = CddaWorld::new_with(
        ItemKind::Bandage,
        vec![ItemKind::Bandage, ItemKind::Bandage],
        &ground,
        queue_behave,
        |b| b,
    );

    sim.tick_n(1);
    assert_eq!(
        sim.plan_names(),
        [
            "queue_fetch_next",
            "move_to_item",
            "pick_up",
            "queue_fetch_next",
            "move_to_item",
            "pick_up",
            "queue_fetch_next",
            "move_to_item",
            "pick_up",
            "queue_fetch_next",
            "move_to_item",
            "pick_up",
            "queue_craft_next",
            "do_craft",
            "queue_craft_next",
            "do_craft",
        ],
        "four fetches (2 scrap + 2 berry), then both crafts, in one plan"
    );

    // Run until BOTH bandages exist (run_until_crafted stops at the first).
    let mut ticks = 1;
    while sim.pocket_count(ItemKind::Bandage) < 2 {
        ticks += 1;
        assert!(ticks <= 200, "the survivor never crafted two bandages");
        sim.tick_n(1);
    }
    sim.tick_n(1); // settle tick, mirroring `run_until_crafted`
    assert_eq!(
        ticks, 34,
        "deterministic tick count for the doubled-craft queue run"
    );

    // Two bandages: every input fetched twice and consumed twice.
    assert_eq!(sim.pocket_count(ItemKind::Bandage), 2);
    assert_eq!(kinds_count(&mut sim, ItemKind::Scrap), 0);
    assert_eq!(kinds_count(&mut sim, ItemKind::Berry), 0);
    assert!(sim.idle());
}
