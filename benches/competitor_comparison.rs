//! Like-for-like comparison of `bevy_bhtn` against the two other HTN planners
//! in the Bevy ecosystem:
//!
//! - [`bevy_bae`](https://crates.io/crates/bevy_bae) 0.1 (Behavior As
//!   Entities) — entity-tree domains, string-keyed props, operators are
//!   systems (bevy_ecs 0.18, the same version this crate uses).
//! - [`bevy_htnp`](https://github.com/QueenOfSquiggles/bevy_htnp) 0.1 —
//!   string-keyed `WorldState`, `TaskRegistry`, declarative goals, time-sliced
//!   planning (vendored under `third_party/` with a repaired manifest; see
//!   there for why).
//!
//! # The problem (identical in all three)
//!
//! htnp's own example domain, expressed with boolean state so every library
//! encodes it the same way: the agent is in room A, the door is closed, and
//! the item is on the ground in room B. Tasks: `pickup_item` (needs room B),
//! `goto_b` (needs the door open), `open_door`, plus the example's two red
//! herrings `goto_a` / `close_door`. The only valid plan is
//! `open_door → goto_b → pickup_item`.
//!
//! # What is measured
//!
//! - **`fetch_item_single_actor`** — one complete AI episode per iteration:
//!   plan from the initial state and carry the plan through to the goal, via
//!   each library's native planning/execution machinery (bevy_bhtn: one plan +
//!   full scratchpad execution; BAE/htnp: their driver schedules run until the
//!   goal prop flips).
//! - **`fetch_item_planning_frame_{n}`** — the frame in which a population of
//!   `n` agents plans, through each library's native driver. bevy_bhtn plans
//!   on an immutable scratchpad (no reset needed); BAE/htnp require their plan
//!   state to be cleared first (component resets, included in the measurement
//!   — clearing plan state is part of their replan cost). BAE's frame also
//!   dispatches each agent's first operator (its planner and executor share
//!   one system pair); htnp's frame is pure planning.
//!
//! Populations are capped at 50k because BAE's per-agent entity-tree domains
//! multiply entity count ~10x.
//!
//! # The deep case (`deep_chain_*`)
//!
//! The same problem shape with the plan size increased ~1.5 orders of
//! magnitude: a corridor of [`DEPTH`] rooms — `step_i` requires
//! `progress == i` and sets `progress = i + 1`, then `pickup_item` — so the
//! only valid plan is `step_0 … step_99, pickup_item` (**101 steps** vs the
//! shallow domain's 3). Same state encoding in all three libraries (a numeric
//! progress counter + a boolean goal flag). Frame populations are 100 / 1k:
//! htnp's tree generator explores one node per plan step with a cloned
//! `HashMap` world per node, which makes deeper frames at 10k+ agents
//! impractical to sample.

mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// bevy_bhtn — typed components, closure conditions/effects
// ---------------------------------------------------------------------------

mod bhtn_side {
    use bevy_bhtn::planner::HtnPlanner;
    use bevy_bhtn::state::PlanState;
    use bevy_bhtn::tasks::TaskBuilder;
    use bevy_bhtn::HtnDomain;
    use bevy_ecs::prelude::*;
    use bevy_ecs::schedule::Schedule;
    use std::hint::black_box;

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct InRoomB(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct DoorOpen(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct ItemPickedUp(pub bool);

    pub fn pickup_item(task: &mut TaskBuilder) {
        task.precondition(|r: &InRoomB, i: &ItemPickedUp| r.0 && !i.0)
            .effect(|i: &mut ItemPickedUp| i.0 = true);
    }
    pub fn goto_b(task: &mut TaskBuilder) {
        task.precondition(|d: &DoorOpen, r: &InRoomB| d.0 && !r.0)
            .effect(|r: &mut InRoomB| r.0 = true);
    }
    pub fn open_door(task: &mut TaskBuilder) {
        task.precondition(|d: &DoorOpen| !d.0)
            .effect(|d: &mut DoorOpen| d.0 = true);
    }
    // The example's red herrings: reachable declarations, never chosen under
    // FirstMatch because a working branch always matches first.
    pub fn goto_a(task: &mut TaskBuilder) {
        task.precondition(|d: &DoorOpen, r: &InRoomB| d.0 && r.0)
            .effect(|r: &mut InRoomB| r.0 = false);
    }
    pub fn close_door(task: &mut TaskBuilder) {
        task.precondition(|d: &DoorOpen| !d.0)
            .effect(|d: &mut DoorOpen| d.0 = false);
    }

    /// Root: loop until the item is picked up. The red-herring branches sit
    /// last so they only ever match if every working branch failed.
    pub fn get_item(task: &mut TaskBuilder) {
        task.branch().precondition(|i: &ItemPickedUp| i.0); // terminal: done
        task.branch()
            .precondition(|r: &InRoomB, i: &ItemPickedUp| r.0 && !i.0)
            .then(pickup_item)
            .then(get_item);
        task.branch()
            .precondition(|d: &DoorOpen, r: &InRoomB| d.0 && !r.0)
            .then(goto_b)
            .then(get_item);
        task.branch()
            .precondition(|d: &DoorOpen| !d.0)
            .then(open_door)
            .then(get_item);
        task.branch()
            .precondition(|d: &DoorOpen, r: &InRoomB| d.0 && r.0)
            .then(goto_a)
            .then(get_item);
        task.branch()
            .precondition(|d: &DoorOpen| !d.0)
            .then(close_door)
            .then(get_item);
    }

    pub fn domain() -> HtnDomain {
        HtnDomain::from_root(get_item).build().expect("well-formed")
    }

    pub fn initial_state(domain: &HtnDomain) -> PlanState {
        PlanState::build(&domain.components).finish() // all defaults: false
    }

    /// Single-actor episode: one plan, then execute every step.
    pub fn single_actor_episode(domain: &HtnDomain, state: &PlanState) -> usize {
        let mut state = state.clone();
        let mut planner = HtnPlanner::new(domain);
        let plan = planner.plan("get_item", &state);
        let steps = plan.task_names().len();
        crate::common::execute_plan(domain, &mut state, &plan);
        steps
    }

    /// The per-agent scratchpad component for the frame benchmark.
    #[derive(Component, Default)]
    pub struct Scratch(pub PlanState);

    /// Planning-only AI frame: every agent plans from its scratchpad. The
    /// planner works on a clone, so no reset pass is needed between frames.
    pub fn run_ai(domain: Res<HtnRes>, mut q: Query<&mut Scratch>) {
        let root = domain.0.root;
        q.par_iter_mut().for_each(|scratch| {
            let mut planner = HtnPlanner::new(&domain.0);
            let plan = planner.plan_index(root, &scratch.0);
            black_box(plan.task_names().len());
        });
    }

    #[derive(Resource)]
    pub struct HtnRes(pub HtnDomain);

    pub fn frame_world(n: usize) -> (World, Schedule) {
        let domain = domain();
        let state = initial_state(&domain);
        let mut world = World::new();
        world
            .spawn_batch((0..n).map(|_| Scratch(PlanState::clone(&state))))
            .count();
        world.insert_resource(HtnRes(domain));
        let mut schedule = Schedule::default();
        schedule.add_systems(run_ai);
        (world, schedule)
    }

    // -- deep chain: 100 rooms, a 101-step plan ------------------------------

    pub const DEPTH: i32 = 100;

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct Progress(pub i32);

    /// Generates `DEPTH` distinct step tasks (`step_i` requires
    /// `progress == i`, sets `progress = i + 1`), the selector compound that
    /// picks the step matching the current progress, and the recursive root.
    /// Distinct fn items per step keep the graph identity mechanism honest —
    /// the domain really has 101 tasks, not one recycled task.
    macro_rules! deep_chain {
        ($(($step:ident, $idx:literal)),* $(,)?) => {
            $(
                fn $step(task: &mut TaskBuilder) {
                    task.precondition(move |p: &Progress| p.0 == $idx)
                        .effect(move |p: &mut Progress| p.0 = $idx + 1);
                }
            )*
            fn deep_pickup(task: &mut TaskBuilder) {
                task.precondition(|p: &Progress, i: &ItemPickedUp| p.0 == DEPTH && !i.0)
                    .effect(|i: &mut ItemPickedUp| i.0 = true);
            }
            fn next_step(task: &mut TaskBuilder) {
                $(
                    task.branch()
                        .precondition(move |p: &Progress| p.0 == $idx)
                        .then($step);
                )*
            }
            fn deep_root(task: &mut TaskBuilder) {
                task.branch().precondition(|i: &ItemPickedUp| i.0); // terminal
                task.branch()
                    .precondition(|p: &Progress| p.0 < DEPTH)
                    .then(next_step)
                    .then(deep_root);
                task.branch()
                    .precondition(|p: &Progress| p.0 == DEPTH)
                    .then(deep_pickup)
                    .then(deep_root);
            }
        };
    }

    deep_chain!(
        (s0, 0),
        (s1, 1),
        (s2, 2),
        (s3, 3),
        (s4, 4),
        (s5, 5),
        (s6, 6),
        (s7, 7),
        (s8, 8),
        (s9, 9),
        (s10, 10),
        (s11, 11),
        (s12, 12),
        (s13, 13),
        (s14, 14),
        (s15, 15),
        (s16, 16),
        (s17, 17),
        (s18, 18),
        (s19, 19),
        (s20, 20),
        (s21, 21),
        (s22, 22),
        (s23, 23),
        (s24, 24),
        (s25, 25),
        (s26, 26),
        (s27, 27),
        (s28, 28),
        (s29, 29),
        (s30, 30),
        (s31, 31),
        (s32, 32),
        (s33, 33),
        (s34, 34),
        (s35, 35),
        (s36, 36),
        (s37, 37),
        (s38, 38),
        (s39, 39),
        (s40, 40),
        (s41, 41),
        (s42, 42),
        (s43, 43),
        (s44, 44),
        (s45, 45),
        (s46, 46),
        (s47, 47),
        (s48, 48),
        (s49, 49),
        (s50, 50),
        (s51, 51),
        (s52, 52),
        (s53, 53),
        (s54, 54),
        (s55, 55),
        (s56, 56),
        (s57, 57),
        (s58, 58),
        (s59, 59),
        (s60, 60),
        (s61, 61),
        (s62, 62),
        (s63, 63),
        (s64, 64),
        (s65, 65),
        (s66, 66),
        (s67, 67),
        (s68, 68),
        (s69, 69),
        (s70, 70),
        (s71, 71),
        (s72, 72),
        (s73, 73),
        (s74, 74),
        (s75, 75),
        (s76, 76),
        (s77, 77),
        (s78, 78),
        (s79, 79),
        (s80, 80),
        (s81, 81),
        (s82, 82),
        (s83, 83),
        (s84, 84),
        (s85, 85),
        (s86, 86),
        (s87, 87),
        (s88, 88),
        (s89, 89),
        (s90, 90),
        (s91, 91),
        (s92, 92),
        (s93, 93),
        (s94, 94),
        (s95, 95),
        (s96, 96),
        (s97, 97),
        (s98, 98),
        (s99, 99),
    );

    pub fn deep_domain() -> HtnDomain {
        HtnDomain::from_root(deep_root)
            .build()
            .expect("well-formed")
    }

    pub fn deep_initial_state(domain: &HtnDomain) -> PlanState {
        PlanState::build(&domain.components).finish() // Progress(0), all flags false
    }

    pub fn deep_single_actor_episode(domain: &HtnDomain, state: &PlanState) -> usize {
        let mut state = state.clone();
        let mut planner = HtnPlanner::new(domain);
        let plan = planner.plan("deep_root", &state);
        let steps = plan.task_names().len();
        crate::common::execute_plan(domain, &mut state, &plan);
        steps
    }

    pub fn deep_frame_world(n: usize) -> (World, Schedule) {
        let domain = deep_domain();
        let state = deep_initial_state(&domain);
        let mut world = World::new();
        world
            .spawn_batch((0..n).map(|_| Scratch(PlanState::clone(&state))))
            .count();
        world.insert_resource(HtnRes(domain));
        let mut schedule = Schedule::default();
        schedule.add_systems(run_ai);
        (world, schedule)
    }
}

// ---------------------------------------------------------------------------
// bevy_bae — entity-tree domains, string-keyed props, operators as systems
// ---------------------------------------------------------------------------

mod bae_side {
    use bevy_app::prelude::*;
    use bevy_bae::prelude::*;
    use bevy_ecs::prelude::*;
    use bevy_ecs::world::World;

    fn done_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }
    fn pickup_item_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }
    fn goto_b_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }
    fn open_door_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }
    fn goto_a_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }
    fn close_door_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }

    /// One agent = one `Plan` entity + its entity-tree domain (BAE's native
    /// model). Unset props read as `false`, which is exactly the initial
    /// state, so no initial `Props` insert is needed.
    pub fn spawn_agent(commands: &mut Commands) -> Entity {
        commands
            .spawn((
                Plan::new(),
                Select,
                tasks![
                    // Terminal: done when the item is picked up.
                    (
                        conditions![Condition::eq("item_picked_up", true)],
                        Operator::new(done_op),
                    ),
                    // The working branch: open the door, cross, pick up.
                    (
                        Sequence,
                        tasks![
                            (
                                conditions![Condition::eq("door_open", false)],
                                Operator::new(open_door_op),
                                effects![Effect::set("door_open", true)],
                            ),
                            (
                                conditions![
                                    Condition::eq("in_room_b", false),
                                    Condition::eq("door_open", true),
                                ],
                                Operator::new(goto_b_op),
                                effects![Effect::set("in_room_b", true)],
                            ),
                            (
                                conditions![
                                    Condition::eq("in_room_b", true),
                                    Condition::eq("item_picked_up", false),
                                ],
                                Operator::new(pickup_item_op),
                                effects![Effect::set("item_picked_up", true)],
                            ),
                        ],
                    ),
                    // Red herrings (the example's goto_a / close_door), lower
                    // priority than the working branch.
                    (
                        Sequence,
                        tasks![(
                            conditions![
                                Condition::eq("door_open", true),
                                Condition::eq("in_room_b", true),
                            ],
                            Operator::new(goto_a_op),
                            effects![Effect::set("in_room_b", false)],
                        )],
                    ),
                    (
                        Sequence,
                        tasks![(
                            conditions![Condition::eq("door_open", false)],
                            Operator::new(close_door_op),
                            effects![Effect::set("door_open", false)],
                        )],
                    ),
                ],
            ))
            .id()
    }

    pub struct BaeApp {
        pub app: App,
        pub agents: Vec<Entity>,
    }

    /// Headless BAE app: the plugin's systems run on `Update`.
    pub fn app_with_agents(n: usize) -> BaeApp {
        let mut app = App::new();
        app.add_plugins(BaePlugin::new(Update));
        let mut agents = Vec::with_capacity(n);
        {
            let world = app.world_mut();
            let mut commands = world.commands();
            for _ in 0..n {
                agents.push(spawn_agent(&mut commands));
            }
        }
        app.world_mut().flush();
        BaeApp { app, agents }
    }

    /// Whether the agent's episode has reached the goal.
    pub fn picked_up(world: &mut World, agent: Entity) -> bool {
        world
            .get::<Props>(agent)
            .map(|p| p.get::<bool>("item_picked_up"))
            .unwrap_or(false)
    }

    /// Reset one agent to the initial state: props back to all-false and an
    /// empty plan (which forces a replan on the next update).
    pub fn reset_agent(app: &mut App, agent: Entity) {
        app.world_mut()
            .entity_mut(agent)
            .insert((Props::default(), Plan::new()));
    }

    /// Deep-chain reset: like [`reset_agent`], but re-seeds the numeric
    /// `progress` prop (unset props read as `Bool(false)`, which no step
    /// condition matches).
    pub fn deep_reset_agent(app: &mut App, agent: Entity) {
        app.world_mut()
            .entity_mut(agent)
            .insert((Props::default().with("progress", 0.0f32), Plan::new()));
    }

    // -- deep chain: 100 rooms, a 101-step plan ------------------------------

    pub const DEPTH: f32 = 100.0;

    fn step_op(In(_input): In<OperatorInput>) -> OperatorStatus {
        OperatorStatus::Success
    }

    /// One deep agent: `Plan` + `Select` → `Sequence` of `DEPTH` step tasks
    /// (condition `progress == i`, effect `progress = i + 1`) plus the final
    /// pickup. Task entities are spawned programmatically via the `TaskOf`
    /// relationship (spawn order = sequence order). `Props` is seeded with
    /// `progress = 0.0`: unset props read as `Bool(false)` in BAE, so an
    /// unset numeric prop would never satisfy `Condition::eq(name, 0.0)`.
    pub fn spawn_deep_agent(commands: &mut Commands) -> Entity {
        let agent = commands
            .spawn((
                Plan::new(),
                Select,
                Props::default().with("progress", 0.0f32),
            ))
            .id();
        let sequence = commands.spawn(Sequence).id();
        commands.entity(sequence).insert(TaskOf(agent));
        // DEPTH step tasks (step_i: progress == i → i + 1), then the pickup
        // (progress == DEPTH) — 101 task entities, a 101-step plan.
        for i in 0..DEPTH as i64 {
            commands.spawn((
                TaskOf(sequence),
                conditions![Condition::eq("progress", i as f32)],
                Operator::new(step_op),
                effects![Effect::set("progress", (i + 1) as f32)],
            ));
        }
        commands.spawn((
            TaskOf(sequence),
            conditions![Condition::eq("progress", DEPTH)],
            Operator::new(pickup_item_op),
            effects![Effect::set("item_picked_up", true)],
        ));
        agent
    }

    pub fn deep_app_with_agents(n: usize) -> BaeApp {
        let mut app = App::new();
        app.add_plugins(BaePlugin::new(Update));
        let mut agents = Vec::with_capacity(n);
        {
            let world = app.world_mut();
            let mut commands = world.commands();
            for _ in 0..n {
                agents.push(spawn_deep_agent(&mut commands));
            }
        }
        app.world_mut().flush();
        BaeApp { app, agents }
    }
}

// ---------------------------------------------------------------------------
// bevy_htnp — string-keyed WorldState, TaskRegistry, declarative goals
// (vendored; see third_party/bevy_htnp/Cargo.toml)
// ---------------------------------------------------------------------------

mod htnp_side {
    use bevy14::prelude::*;
    use bevy_htnp::planning::goals::Goal;
    use bevy_htnp::planning::plan_data::TimeSlicedTreeGen;
    use bevy_htnp::prelude::*;

    // Task-marker components. Manual `Component` impls: bevy 0.14's derive
    // emits `bevy_ecs::…` paths that would resolve to this crate's bevy_ecs
    // 0.18 (two engine generations coexist in this bench).
    macro_rules! htnp_marker {
        ($name:ident) => {
            #[derive(Default)]
            pub struct $name;
            impl bevy14::ecs::component::Component for $name {
                const STORAGE_TYPE: bevy14::ecs::component::StorageType =
                    bevy14::ecs::component::StorageType::Table;
            }
        };
    }

    htnp_marker!(PickupItemMarker);
    htnp_marker!(GotoBMarker);
    htnp_marker!(OpenDoorMarker);
    htnp_marker!(GotoAMarker);
    htnp_marker!(CloseDoorMarker);

    pub const TASK_NAMES: [&str; 5] =
        ["pickup_item", "goto_b", "open_door", "goto_a", "close_door"];

    pub fn task_registry() -> TaskRegistry {
        let mut reg = TaskRegistry::new();
        reg.task::<PickupItemMarker, _>(
            "pickup_item",
            Requirements::new()
                .req_equals("in_room_b", true)
                .req_equals("item_picked_up", false)
                .build(),
            WorldState::new().add("item_picked_up", true).build(),
            1.0,
        );
        reg.task::<GotoBMarker, _>(
            "goto_b",
            Requirements::new()
                .req_equals("door_open", true)
                .req_equals("in_room_b", false)
                .build(),
            WorldState::new().add("in_room_b", true).build(),
            1.0,
        );
        reg.task::<OpenDoorMarker, _>(
            "open_door",
            Requirements::new().req_equals("door_open", false).build(),
            WorldState::new().add("door_open", true).build(),
            1.0,
        );
        reg.task::<GotoAMarker, _>(
            "goto_a",
            Requirements::new()
                .req_equals("door_open", true)
                .req_equals("in_room_b", true)
                .build(),
            WorldState::new().add("in_room_b", false).build(),
            1.0,
        );
        reg.task::<CloseDoorMarker, _>(
            "close_door",
            Requirements::new().req_equals("door_open", false).build(),
            WorldState::new().add("door_open", false).build(),
            1.0,
        );
        reg
    }

    pub fn initial_world() -> WorldState {
        WorldState::new()
            .add("in_room_b", false)
            .add("door_open", false)
            .add("item_picked_up", false)
            .build()
    }

    fn agent_tasks() -> Vec<Task> {
        TASK_NAMES
            .iter()
            .map(|name| Task::primitive(*name))
            .collect()
    }

    fn agent_goals() -> Vec<Goal> {
        vec![Goal::new(
            "get_item",
            Requirements::new()
                .req_equals("item_picked_up", true)
                .build(),
            1.0,
        )]
    }

    /// The user-side operator systems: realize each task's postconditions on
    /// the agent's `HtnAgentWorld`, then report success (htnp's execution
    /// contract — the registry's postcons are *planning* data; the task
    /// system applies the real effect).
    macro_rules! task_system {
        ($fn_name:ident, $marker:ty, $($key:literal => $value:literal),* $(,)?) => {
            fn $fn_name(
                mut q: Query<(Entity, &HtnAgentState, &mut HtnAgentWorld), With<$marker>>,
                mut commands: Commands,
            ) {
                for (entity, state, mut agent_world) in &mut q {
                    if *state == HtnAgentState::Running {
                        $(agent_world.0.add($key, $value);)*
                        commands.entity(entity).insert(HtnAgentState::Success);
                    }
                }
            }
        };
    }

    task_system!(pickup_item_system, PickupItemMarker, "item_picked_up" => true);
    task_system!(goto_b_system, GotoBMarker, "in_room_b" => true);
    task_system!(open_door_system, OpenDoorMarker, "door_open" => true);
    task_system!(goto_a_system, GotoAMarker, "in_room_b" => false);
    task_system!(close_door_system, CloseDoorMarker, "door_open" => false);

    pub struct HtnpApp {
        pub app: App,
        pub agents: Vec<Entity>,
        tasks: Vec<Task>,
        goals: Vec<Goal>,
    }

    /// Headless htnp app: the plugin's systems run on `Update`, chained for a
    /// deterministic per-frame cadence.
    pub fn app_with_agents(n: usize) -> HtnpApp {
        let mut app = App::new();
        app.add_plugins(HtnPlanningPlugin::new().orchestrate(OrchestrateFor::FasterResponse));
        app.insert_resource(task_registry());
        app.add_systems(
            Update,
            (
                pickup_item_system,
                goto_b_system,
                open_door_system,
                goto_a_system,
                close_door_system,
            ),
        );

        let tasks = agent_tasks();
        let goals = agent_goals();
        let mut agents = Vec::with_capacity(n);
        for _ in 0..n {
            let mut agent = HtnAgent::new();
            for name in TASK_NAMES {
                agent.add_task(Task::primitive(name));
            }
            agent.add_goal(
                "get_item",
                Requirements::new()
                    .req_equals("item_picked_up", true)
                    .build(),
                1.0,
            );
            let entity = app
                .world_mut()
                .spawn((
                    agent,
                    TimeSlicedTreeGen::new_initialized(tasks.clone(), goals.clone()),
                    HtnAgentWorld(initial_world()),
                ))
                .id();
            agents.push(entity);
        }
        HtnpApp {
            app,
            agents,
            tasks,
            goals,
        }
    }

    /// Whether the agent's episode has reached the goal.
    pub fn picked_up(app: &mut App, agent: Entity) -> bool {
        app.world()
            .get::<HtnAgentWorld>(agent)
            .is_some_and(|w| w.0.get("item_picked_up") == Some(Variant::Bool(true)))
    }

    /// Reset one agent to the initial state: purge execution state, restore
    /// the world state, and insert a fresh (empty) plan-tree generator so the
    /// next update replans from scratch.
    pub fn reset_agent(htnp: &mut HtnpApp, agent: Entity) {
        let app = &mut htnp.app;
        app.world_mut().entity_mut(agent).remove::<(
            HtnAgentPlan,
            HtnAgentState,
            HtnAgentCurrentTask,
            PickupItemMarker,
        )>();
        app.world_mut().entity_mut(agent).remove::<(
            GotoBMarker,
            OpenDoorMarker,
            GotoAMarker,
            CloseDoorMarker,
        )>();
        app.world_mut().entity_mut(agent).insert((
            HtnAgentWorld(initial_world()),
            TimeSlicedTreeGen::new_initialized(htnp.tasks.clone(), htnp.goals.clone()),
        ));
    }

    // -- deep chain: 100 rooms, a 101-step plan ------------------------------

    pub const DEPTH: f32 = 100.0;

    htnp_marker!(StepMarker);

    fn deep_task_registry() -> TaskRegistry {
        let mut reg = TaskRegistry::new();
        for i in 0..DEPTH as i64 {
            reg.task::<StepMarker, _>(
                format!("step_{i}"),
                Requirements::new().req_equals("progress", i as f32).build(),
                WorldState::new().add("progress", (i + 1) as f32).build(),
                1.0,
            );
        }
        reg.task::<PickupItemMarker, _>(
            "pickup_item",
            Requirements::new().req_equals("progress", DEPTH).build(),
            WorldState::new().add("item_picked_up", true).build(),
            1.0,
        );
        reg
    }

    fn deep_initial_world() -> WorldState {
        WorldState::new()
            .add("progress", 0.0)
            .add("item_picked_up", false)
            .build()
    }

    /// The user-side operator system for the deep chain's step tasks: bump the
    /// agent's progress and report success.
    fn deep_step_system(
        mut q: Query<(Entity, &HtnAgentState, &mut HtnAgentWorld), With<StepMarker>>,
        mut commands: Commands,
    ) {
        for (entity, state, mut agent_world) in &mut q {
            if *state == HtnAgentState::Running {
                let current = match agent_world.0.get("progress") {
                    Some(Variant::Number(n)) => n,
                    _ => 0.0,
                };
                agent_world.0.add("progress", current + 1.0);
                commands.entity(entity).insert(HtnAgentState::Success);
            }
        }
    }

    pub struct HtnpDeepApp {
        pub app: App,
        pub agents: Vec<Entity>,
        tasks: Vec<Task>,
        goals: Vec<Goal>,
    }

    pub fn deep_app_with_agents(n: usize) -> HtnpDeepApp {
        let mut app = App::new();
        app.add_plugins(HtnPlanningPlugin::new().orchestrate(OrchestrateFor::FasterResponse));
        app.insert_resource(deep_task_registry());
        app.add_systems(Update, (pickup_item_system, deep_step_system));

        let mut tasks = Vec::with_capacity(DEPTH as usize + 1);
        for i in 0..DEPTH as i64 {
            tasks.push(Task::primitive(format!("step_{i}")));
        }
        tasks.push(Task::primitive("pickup_item"));
        let goals = vec![Goal::new(
            "get_item",
            Requirements::new()
                .req_equals("item_picked_up", true)
                .build(),
            1.0,
        )];

        let mut agents = Vec::with_capacity(n);
        for _ in 0..n {
            let mut agent = HtnAgent::new();
            for task in &tasks {
                agent.add_task(task.clone());
            }
            agent.add_goal(
                "get_item",
                Requirements::new()
                    .req_equals("item_picked_up", true)
                    .build(),
                1.0,
            );
            let entity = app
                .world_mut()
                .spawn((
                    agent,
                    TimeSlicedTreeGen::new_initialized(tasks.clone(), goals.clone()),
                    HtnAgentWorld(deep_initial_world()),
                ))
                .id();
            agents.push(entity);
        }
        HtnpDeepApp {
            app,
            agents,
            tasks,
            goals,
        }
    }

    pub fn deep_picked_up(app: &mut App, agent: Entity) -> bool {
        app.world()
            .get::<HtnAgentWorld>(agent)
            .is_some_and(|w| w.0.get("item_picked_up") == Some(Variant::Bool(true)))
    }

    pub fn deep_reset_agent(htnp: &mut HtnpDeepApp, agent: Entity) {
        let app = &mut htnp.app;
        app.world_mut().entity_mut(agent).remove::<(
            HtnAgentPlan,
            HtnAgentState,
            HtnAgentCurrentTask,
            PickupItemMarker,
            StepMarker,
        )>();
        app.world_mut().entity_mut(agent).insert((
            HtnAgentWorld(deep_initial_world()),
            TimeSlicedTreeGen::new_initialized(htnp.tasks.clone(), htnp.goals.clone()),
        ));
    }
}

// ---------------------------------------------------------------------------
// The benchmark
// ---------------------------------------------------------------------------

const POPULATIONS: [usize; 3] = [1_000, 10_000, 50_000];
/// Defensive cap on per-episode update loops (the episodes are a handful of
/// frames; hitting this means a library failed to reach the goal).
const EPISODE_FRAME_CAP: usize = 100;

fn competitor_comparison(c: &mut Criterion) {
    // --- Single-actor: one complete AI episode -----------------------------
    {
        let domain = bhtn_side::domain();
        let state = bhtn_side::initial_state(&domain);
        let mut bae = bae_side::app_with_agents(1);
        let bae_agent = bae.agents[0];
        let mut htnp = htnp_side::app_with_agents(1);
        let htnp_agent = htnp.agents[0];

        let mut group = c.benchmark_group("fetch_item_single_actor");
        group.throughput(criterion::Throughput::Elements(1));
        group.bench_function("bevy_bhtn", |b| {
            b.iter(|| black_box(bhtn_side::single_actor_episode(&domain, &state)))
        });
        group.bench_function("bevy_bae", |b| {
            b.iter(|| {
                bae_side::reset_agent(&mut bae.app, bae_agent);
                let mut frames = 0;
                while !bae_side::picked_up(bae.app.world_mut(), bae_agent) {
                    bae.app.update();
                    frames += 1;
                    assert!(frames <= EPISODE_FRAME_CAP, "BAE episode did not finish");
                }
                black_box(frames);
            })
        });
        group.bench_function("bevy_htnp", |b| {
            b.iter(|| {
                htnp_side::reset_agent(&mut htnp, htnp_agent);
                let mut frames = 0;
                while !htnp_side::picked_up(&mut htnp.app, htnp_agent) {
                    htnp.app.update();
                    frames += 1;
                    assert!(frames <= EPISODE_FRAME_CAP, "htnp episode did not finish");
                }
                black_box(frames);
            })
        });
        group.finish();
    }

    // --- Planning frame: a population plans in one frame --------------------
    for n in POPULATIONS {
        let mut bae = bae_side::app_with_agents(n);
        let bae_agents = bae.agents.clone();
        let mut htnp = htnp_side::app_with_agents(n);
        let htnp_agents = htnp.agents.clone();
        let (mut bhtn_world, mut bhtn_schedule) = bhtn_side::frame_world(n);

        let mut group = c.benchmark_group(format!("fetch_item_planning_frame_{n}"));
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_function("bevy_bhtn", |b| {
            b.iter(|| bhtn_schedule.run(&mut bhtn_world))
        });
        group.bench_function("bevy_bae", |b| {
            b.iter(|| {
                // Clear plan state (part of BAE's replan cost), then one
                // update: replan + first operator for every agent.
                for &agent in &bae_agents {
                    bae_side::reset_agent(&mut bae.app, agent);
                }
                bae.app.update();
            })
        });
        group.bench_function("bevy_htnp", |b| {
            b.iter(|| {
                // Fresh plan-tree generators (part of htnp's replan cost),
                // then one update: the chained systems plan every agent.
                for &agent in &htnp_agents {
                    htnp_side::reset_agent(&mut htnp, agent);
                }
                htnp.app.update();
            })
        });
        group.finish();
    }

    // --- Deep chain: a 101-step plan (plan size +~1.5 orders of magnitude) --
    {
        let domain = bhtn_side::deep_domain();
        let state = bhtn_side::deep_initial_state(&domain);
        let mut bae = bae_side::deep_app_with_agents(1);
        let bae_agent = bae.agents[0];
        let mut htnp = htnp_side::deep_app_with_agents(1);
        let htnp_agent = htnp.agents[0];

        let mut group = c.benchmark_group("deep_chain_single_actor");
        group.throughput(criterion::Throughput::Elements(1));
        group.bench_function("bevy_bhtn", |b| {
            b.iter(|| black_box(bhtn_side::deep_single_actor_episode(&domain, &state)))
        });
        group.bench_function("bevy_bae", |b| {
            b.iter(|| {
                bae_side::deep_reset_agent(&mut bae.app, bae_agent);
                let mut frames = 0;
                while !bae_side::picked_up(bae.app.world_mut(), bae_agent) {
                    bae.app.update();
                    frames += 1;
                    assert!(
                        frames <= 4 * EPISODE_FRAME_CAP,
                        "BAE deep episode did not finish"
                    );
                }
                black_box(frames);
            })
        });
        group.bench_function("bevy_htnp", |b| {
            b.iter(|| {
                htnp_side::deep_reset_agent(&mut htnp, htnp_agent);
                let mut frames = 0;
                while !htnp_side::deep_picked_up(&mut htnp.app, htnp_agent) {
                    htnp.app.update();
                    frames += 1;
                    assert!(
                        frames <= 4 * EPISODE_FRAME_CAP,
                        "htnp deep episode did not finish"
                    );
                }
                black_box(frames);
            })
        });
        group.finish();
    }

    // --- Deep-chain planning frames (100 / 1k agents — see module docs) -----
    for n in [100usize, 1_000] {
        let (mut bhtn_world, mut bhtn_schedule) = bhtn_side::deep_frame_world(n);
        let mut bae = bae_side::deep_app_with_agents(n);
        let bae_agents = bae.agents.clone();
        let mut htnp = htnp_side::deep_app_with_agents(n);
        let htnp_agents = htnp.agents.clone();

        let mut group = c.benchmark_group(format!("deep_chain_planning_frame_{n}"));
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_function("bevy_bhtn", |b| {
            b.iter(|| bhtn_schedule.run(&mut bhtn_world))
        });
        group.bench_function("bevy_bae", |b| {
            b.iter(|| {
                for &agent in &bae_agents {
                    bae_side::deep_reset_agent(&mut bae.app, agent);
                }
                bae.app.update();
            })
        });
        group.bench_function("bevy_htnp", |b| {
            b.iter(|| {
                for &agent in &htnp_agents {
                    htnp_side::deep_reset_agent(&mut htnp, agent);
                }
                htnp.app.update();
            })
        });
        group.finish();
    }
}

criterion_group!(benches, competitor_comparison);
criterion_main!(benches);
