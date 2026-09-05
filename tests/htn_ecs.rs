//! Pins for the Bevy ECS execution layer (`bevy_bhtn::ecs`): the [`HtnAgent`]
//! component, [`HtnConfig`], and the exclusive [`htn_ai_system`] driver's
//! plan → validate → execute-one-step → replan loop over real components.

use bevy_bhtn::ecs::{htn_ai_system, AgentRoot, HtnAgent, HtnConfig, PlanEvery};
use bevy_bhtn::tasks::{GoalBuilder, TaskBuilder};
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;

/// The agents' plan state lives in ordinary components.
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Battery(i32);
/// Inserted by the action under test (command dispatch observable).
#[derive(Component, Default, Debug)]
struct Gathered;

fn charge_domain() -> HtnDomain {
    fn charge(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|battery: &Battery| battery.0 >= 3);
        task.branch()
            .precondition(|battery: &Battery| battery.0 < 3)
            .then(gather)
            .then(charge);
    }
    fn gather(task: &mut TaskBuilder) {
        task.precondition(|battery: &Battery| battery.0 < 3)
            .effect(|battery: &mut Battery| battery.0 += 1)
            .action(|cmds: &mut EntityCommands| {
                cmds.insert(Gathered);
            });
    }
    HtnDomain::from_root(charge)
        .build()
        .expect("charge domain is well-formed")
}

fn goal_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(earn).then(root);
        task.branch().precondition(|gold: &Gold| gold.0 >= 3);
    }
    fn earn(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 < 3)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn three_gold(task: &mut GoalBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 3);
    }
    HtnDomain::from_root(root).goal(three_gold).build().unwrap()
}

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(i32);

/// A full agent lifecycle: planless agent plans on its first tick, executes
/// one primitive per tick — action commands dispatched, effects committed to
/// the real components — and finishes with the goal condition true and the
/// plan cleared for the next planning cycle.
#[test]
fn agent_plans_executes_and_completes() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(charge_domain()));
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    // Tick 1: plans (3x gather) and executes the first step.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert_eq!(agent.plan().map(|p| p.len()), Some(3));
    assert_eq!(agent.cursor(), 1);
    assert_eq!(
        world.get::<Battery>(entity).unwrap().0,
        1,
        "effect committed to the real component"
    );
    assert!(
        world.get::<Gathered>(entity).is_some(),
        "action commands were flushed"
    );

    // Ticks 2–3: two more steps; the plan completes on tick 3.
    htn_ai_system(&mut world);
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan().is_none(), "finished plan is dropped");
    assert_eq!(agent.cursor(), 0);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);

    // Tick 4: replans; the terminal branch now matches and plans nothing.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan().is_none());
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}

/// The world drifted since planning (another system spent the resource): the
/// driver re-validates the next step's preconditions against the real world.
/// A failed check drops the plan and **replans from reality in the same
/// tick** — the fresh plan is selected against the world as it is *now* and
/// executes immediately, instead of idling until the next tick.
#[test]
fn world_drift_triggers_replan_and_recovery() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(charge_domain()));
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    // Tick 1: plans and executes one gather.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

    // Simulate drift: another system charges the battery past the gather
    // gate (battery < 3) that the plan's remaining steps assume.
    world.get_mut::<Battery>(entity).unwrap().0 = 3;

    // Tick 2: the plan's next gather is invalid → dropped — and the fresh
    // plan is selected the *same tick*: the terminal branch matches (battery
    // >= 3), so the agent goes planless with no effect applied.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan().is_none(), "invalid step aborts the plan");
    assert_eq!(
        world.get::<Battery>(entity).unwrap().0,
        3,
        "no effect applied"
    );

    // Tick 3: still planless — the terminal branch keeps matching.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}

/// The same-tick property, made observable: drift into a state where the
/// replanned plan is **non-empty**, so the replacement plan's first step
/// executes within the very tick the drift was detected.
#[test]
fn drift_replans_and_executes_same_tick() {
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct Cell(pub i32);

    fn root(task: &mut TaskBuilder) {
        task.branch().precondition(|c: &Cell| c.0 >= 5); // terminal: done
        task.branch()
            .precondition(|c: &Cell| (3..5).contains(&c.0))
            .then(fast_charge);
        task.branch()
            .precondition(|c: &Cell| c.0 < 3)
            .then(gather)
            .then(root);
    }
    fn gather(task: &mut TaskBuilder) {
        task.precondition(|c: &Cell| c.0 < 3)
            .effect(|c: &mut Cell| c.0 += 1);
    }
    fn fast_charge(task: &mut TaskBuilder) {
        task.precondition(|c: &Cell| (3..5).contains(&c.0))
            .effect(|c: &mut Cell| c.0 = 6);
    }
    let domain = HtnDomain::from_root(root).build().unwrap();

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain));
    let entity = world.spawn((Cell(0), HtnAgent::default())).id();

    // Ticks 1–2: the plan [gather, gather, gather, fast_charge] executes its
    // first two gathers (0 → 2).
    htn_ai_system(&mut world);
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Cell>(entity).unwrap().0, 2);
    assert_eq!(world.get::<HtnAgent>(entity).unwrap().cursor(), 2);

    // External disturbance: another system charges the cell past the gather
    // gate — the plan's third gather (precondition < 3) can never fire.
    world.get_mut::<Cell>(entity).unwrap().0 = 4;

    // Tick 3: the gather step's validation fails → plan dropped → replanned
    // from reality **this tick**: the fresh plan is [fast_charge], and its
    // step executes within the same tick (cell jumps 4 → 6 — the *old* plan
    // would have produced 5 via gather, and only on the next tick).
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert_eq!(
        world.get::<Cell>(entity).unwrap().0,
        6,
        "the replanned step executed in the drift tick itself"
    );
    assert!(
        agent.plan().is_none(),
        "one-step plan completed; planless again"
    );

    // Tick 4: the terminal branch (>= 5) matches — idle.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Cell>(entity).unwrap().0, 6);
}

/// An empty plan is stored as planless (never wedges the agent), and the
/// driver serves agents of different shapes in one pass with config tuning
/// honored (look-ahead off still plans).
#[test]
fn driver_handles_multiple_agents_and_config_tuning() {
    let mut cfg = HtnConfig::new(goal_domain());
    cfg.lookahead = bevy_bhtn::selection::LookaheadMode::Off;
    let mut world = World::new();
    world.insert_resource(cfg);

    let a = world.spawn((Gold(2), HtnAgent::default())).id();
    let b = world.spawn((Gold(0), HtnAgent::default())).id();

    for _ in 0..3 {
        htn_ai_system(&mut world);
    }

    assert_eq!(world.get::<Gold>(a).unwrap().0, 3);
    assert_eq!(world.get::<Gold>(b).unwrap().0, 3);
    // Both agents finished; a replan on the next tick finds the terminal
    // branch and stores an empty (planless) result.
    htn_ai_system(&mut world);
    assert!(world.get::<HtnAgent>(a).unwrap().plan().is_none());
    assert!(world.get::<HtnAgent>(b).unwrap().plan().is_none());
}

/// Agents without the domain's components still work: missing components
/// materialize as `Default` in the scratchpad, so preconditions evaluate
/// against defaults instead of panicking.
#[test]
fn missing_components_materialize_as_defaults() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(charge_domain()));
    // No Battery component at all — defaults to 0.
    let entity = world.spawn(HtnAgent::default()).id();

    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert_eq!(agent.plan().map(|p| p.len()), Some(3));
    // The effect wrote Battery back onto the entity (write-back inserts it).
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);
}

/// The backward planner's goal is reachable through the ECS layer's domain:
/// planning from the entity's real components reaches the goal effects.
#[test]
fn goal_domain_reaches_goal_from_real_components() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(goal_domain()));
    let entity = world.spawn((Gold(0), HtnAgent::default())).id();

    for _ in 0..3 {
        htn_ai_system(&mut world);
    }
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 3);
}

// ---------------------------------------------------------------------------
// Subsecond hot-reload support (feature `hotpatching`): the driver detects a
// patch via `HotPatchChanges`, re-records + re-bakes the domain through the
// config's rebuild closure, and drops every agent's plan. The actual code
// patching is the dx CLI's job; these tests pin the swap machinery.
// ---------------------------------------------------------------------------

#[cfg(feature = "hotpatching")]
mod hotpatch {
    use super::*;

    /// Two domains over the same component with different effect values, so a
    /// successful rebuild is observable through the committed effect. Task
    /// functions must be *named* (identity comes from `type_name`), so each
    /// variant gets its own module.
    mod writes_one {
        use super::*;
        pub fn root(task: &mut TaskBuilder) {
            task.branch().precondition(|b: &Battery| b.0 >= 3);
            task.branch().precondition(|b: &Battery| b.0 < 3).then(step);
        }
        pub fn step(task: &mut TaskBuilder) {
            task.effect(|b: &mut Battery| b.0 = 1);
        }
    }
    mod writes_two {
        use super::*;
        pub fn root(task: &mut TaskBuilder) {
            task.branch().precondition(|b: &Battery| b.0 >= 3);
            task.branch().precondition(|b: &Battery| b.0 < 3).then(step);
        }
        pub fn step(task: &mut TaskBuilder) {
            task.effect(|b: &mut Battery| b.0 = 2);
        }
    }

    fn domain_with_value(value: i32) -> HtnDomain {
        match value {
            1 => HtnDomain::from_root(writes_one::root).build().unwrap(),
            2 => HtnDomain::from_root(writes_two::root).build().unwrap(),
            _ => unreachable!("test domains write 1 or 2"),
        }
    }

    /// Simulate a hot patch: insert (if missing) and bump `HotPatchChanges`.
    fn land_patch(world: &mut World) {
        if world.get_resource::<bevy_ecs::HotPatchChanges>().is_none() {
            world.insert_resource(bevy_ecs::HotPatchChanges);
        }
        world
            .resource_mut::<bevy_ecs::HotPatchChanges>()
            .set_changed();
    }

    #[test]
    fn hot_patch_rebuilds_domain_and_drops_plans() {
        let mut world = World::new();
        // Initial domain writes 1; the "patched" rebuild returns one that
        // writes 7 — the observable behavior change.
        world.insert_resource(HtnConfig::with_rebuild(domain_with_value(1), || {
            Ok::<_, bevy_bhtn::HtnError>(domain_with_value(2))
        }));
        let entity = world.spawn((Battery(0), HtnAgent::default())).id();

        // Tick 1: baseline observation (no rebuild), plan + execute → 1.
        htn_ai_system(&mut world);
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

        // A patch lands between ticks.
        land_patch(&mut world);

        // Tick 2: the driver rebuilds the domain (now writes 2), drops the
        // agent's plan, and the agent replans + executes under the new
        // domain in the same tick.
        htn_ai_system(&mut world);
        assert_eq!(
            world.get::<Battery>(entity).unwrap().0,
            2,
            "the rebuilt domain's effect committed"
        );

        // No further patch: behavior is stable.
        htn_ai_system(&mut world);
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 2);
    }

    #[test]
    fn failed_rebuild_keeps_the_old_domain() {
        let mut world = World::new();
        world.insert_resource(HtnConfig::with_rebuild(domain_with_value(1), || {
            Err(bevy_bhtn::HtnError::builder("patch does not compile"))
        }));
        let entity = world.spawn((Battery(0), HtnAgent::default())).id();

        htn_ai_system(&mut world);
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

        land_patch(&mut world);
        htn_ai_system(&mut world);
        // The rebuild failed: the old domain still drives behavior (it
        // writes 1 again), and the agent keeps planning normally.
        assert_eq!(
            world.get::<Battery>(entity).unwrap().0,
            1,
            "failed rebuild keeps the old domain"
        );
    }

    #[test]
    fn patch_without_rebuild_closure_is_a_noop() {
        let mut world = World::new();
        world.insert_resource(HtnConfig::new(domain_with_value(1)));
        let entity = world.spawn((Battery(0), HtnAgent::default())).id();

        htn_ai_system(&mut world);
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

        land_patch(&mut world);
        htn_ai_system(&mut world);
        // No rebuild closure: the domain is untouched (still writes 1).
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);
    }

    /// A rebuild that changes the registry (a new component enters the
    /// domain) is supported: the parked scratchpad is dropped and rebuilt
    /// from the new registry, and agents replan cleanly.
    #[test]
    fn rebuild_with_new_component_resets_the_scratchpad() {
        use bevy_bhtn::state::PlanComponent;

        #[derive(Component, Clone, Default, Debug, PartialEq)]
        struct Extra(i32);

        fn domain_with_extra() -> HtnDomain {
            fn root(task: &mut TaskBuilder) {
                task.branch().precondition(|b: &Battery| b.0 >= 3);
                task.branch().precondition(|b: &Battery| b.0 < 3).then(step);
            }
            fn step(task: &mut TaskBuilder) {
                task.precondition(|e: &Extra| e.0 == 0)
                    .effect(|b: &mut Battery| b.0 = 7)
                    .effect(|e: &mut Extra| e.0 = 9);
            }
            HtnDomain::from_root(root).build().unwrap()
        }
        fn plain_domain() -> HtnDomain {
            domain_with_value(1)
        }

        let mut world = World::new();
        world.insert_resource(HtnConfig::with_rebuild(plain_domain(), || {
            Ok::<HtnDomain, bevy_bhtn::HtnError>(domain_with_extra())
        }));
        let entity = world
            .spawn((Battery(0), Extra(0), HtnAgent::default()))
            .id();

        htn_ai_system(&mut world);
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

        land_patch(&mut world);
        htn_ai_system(&mut world);
        // The rebuilt domain's step wrote both components: the scratchpad was
        // rebuilt against the new registry and the agent replanned.
        assert_eq!(world.get::<Battery>(entity).unwrap().0, 7);
        assert_eq!(world.get::<Extra>(entity).unwrap().0, 9);

        fn _assert_component<T: PlanComponent>() {}
        _assert_component::<Extra>();
    }
}

// ---------------------------------------------------------------------------
// LOD scheduling (`PlanEvery`) and unsolvable-domain legibility
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Hits(pub u32);

/// One-step domain: a planless agent replans and executes every run.
fn always_hit_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(hit);
    }
    fn hit(task: &mut TaskBuilder) {
        task.effect(|h: &mut Hits| h.0 += 1);
    }
    HtnDomain::from_root(root).build().unwrap()
}

/// `PlanEvery(n)` paces (re)planning to once every `n` runs while the agent
/// sits planless: over 10 runs a throttled agent acts ⌈10/3⌉ = 4 times where
/// an unthrottled one acts 10. Execution of an existing plan is never delayed
/// (the first plan's single step executes on the same run it was planned).
#[test]
fn plan_every_throttles_replanning() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(always_hit_domain()));
    let throttled = world
        .spawn((Hits(0), HtnAgent::default(), PlanEvery(3)))
        .id();
    let free = world.spawn((Hits(0), HtnAgent::default())).id();

    for _ in 0..10 {
        htn_ai_system(&mut world);
    }

    assert_eq!(
        world.get::<Hits>(throttled).unwrap().0,
        4,
        "plans at runs 1, 4, 7, 10"
    );
    assert_eq!(
        world.get::<Hits>(free).unwrap().0,
        10,
        "unthrottled control"
    );
}

/// An agent whose domain genuinely has no decomposition gets `NoPlan` from
/// the planner — the driver treats it like an empty plan (planless, idle) and
/// keeps ticking without wedging or panicking, retrying at the `PlanEvery`
/// cadence when one is present.
#[test]
fn unsolvable_domain_leaves_the_agent_planless_not_wedged() {
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct Locked(bool);
    fn root(task: &mut TaskBuilder) {
        task.branch().then(impossible);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|l: &Locked| l.0);
    }
    let domain = HtnDomain::from_root(root).build().unwrap();

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain));
    let entity = world
        .spawn((Locked(false), HtnAgent::default(), PlanEvery(2)))
        .id();

    for _ in 0..6 {
        htn_ai_system(&mut world);
    }
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan().is_none(), "no plan is ever stored");
    assert_eq!(agent.cursor(), 0);
}

// ---------------------------------------------------------------------------
// Quiet-prefix elision (replan dedup, after Fluid HTN's early replan
// rejection) — a fresh plan's leading run of steps that dispatch no action
// and whose effects are already reflected in the world is skipped, so a
// same-tick repair replan resumes at its first consequential step.
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gate(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct GoA(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct GoB(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Step1(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Done(pub bool);

/// A three-step journey: prepare the gate, then two gated acts. `ensure` is
/// idempotent and action-free — quiet once the gate is up.
fn gate_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().precondition(|d: &Done| d.0); // terminal: finished
        task.branch()
            .then(ensure)
            .then(act_a)
            .then(act_b)
            .then(root);
    }
    fn ensure(task: &mut TaskBuilder) {
        task.effect(|g: &mut Gate| g.0 = true);
    }
    fn act_a(task: &mut TaskBuilder) {
        task.precondition(|g: &Gate| g.0)
            .precondition(|a: &GoA| a.0)
            .effect(|s: &mut Step1| s.0 = true);
    }
    fn act_b(task: &mut TaskBuilder) {
        task.precondition(|g: &Gate| g.0)
            .precondition(|b: &GoB| b.0)
            .precondition(|s: &Step1| s.0)
            .effect(|d: &mut Done| d.0 = true);
    }
    HtnDomain::from_root(root).build().unwrap()
}

/// Mid-plan drift replans into the same domain; the fresh plan's `ensure`
/// prefix is quiet (the gate is already set — its effect would change
/// nothing), so the replan resumes at `act_a` instead of re-running
/// `ensure`. The tick the blocker clears, the journey advances — no tick is
/// spent re-executing a step whose work is already done.
#[test]
fn replan_elides_quiet_prefix_and_advances_same_tick() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(gate_domain()));
    let entity = world
        .spawn((
            Gate(false),
            GoA(true),
            GoB(true),
            Step1(false),
            Done(false),
            HtnAgent::default(),
        ))
        .id();

    // Tick 1: plan [ensure, act_a, act_b]; `ensure` is NOT quiet (the gate
    // flips) and executes.
    htn_ai_system(&mut world);
    assert!(world.get::<Gate>(entity).unwrap().0, "the gate was set");
    assert_eq!(world.get::<HtnAgent>(entity).unwrap().cursor(), 1);

    // The world closes `act_a`'s precondition mid-plan.
    world.get_mut::<GoA>(entity).unwrap().0 = false;

    // Tick 2: `act_a` drifts → dropped → replanned same tick → the fresh
    // plan's `ensure` prefix IS quiet → elided → `act_a` still blocked →
    // planless. (Without elision this tick would re-execute `ensure` and
    // hold the plan.)
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(
        agent.plan().is_none(),
        "the blocked replan consumed both passes"
    );
    assert!(!world.get::<Step1>(entity).unwrap().0, "act_a never ran");

    // The world opens the way.
    world.get_mut::<GoA>(entity).unwrap().0 = true;

    // Tick 3: plan → the quiet `ensure` prefix is elided → `act_a`
    // validates and executes **within this tick**. Without elision this
    // tick would re-run `ensure` and `act_a` would land a tick later.
    htn_ai_system(&mut world);
    assert!(
        world.get::<Step1>(entity).unwrap().0,
        "the replanned plan resumed at its first consequential step"
    );
    assert_eq!(world.get::<HtnAgent>(entity).unwrap().cursor(), 2);

    // Tick 4: `act_b` finishes the journey.
    htn_ai_system(&mut world);
    assert!(world.get::<Done>(entity).unwrap().0);
    assert!(world.get::<HtnAgent>(entity).unwrap().plan().is_none());
}

/// Elision must never skip a step whose effect changes state — even one
/// that already executed earlier in the plan. The counterexample that kills
/// naive structural plan comparison: an executed step's effect gets undone
/// by drift; the replan re-derives the step; skipping it would livelock.
#[test]
fn elision_never_skips_state_changing_steps() {
    fn root(task: &mut TaskBuilder) {
        task.branch().precondition(|d: &Done| d.0); // terminal: finished
        task.branch()
            .precondition(|c: &Count| c.0 >= 2)
            .then(act)
            .then(root);
        task.branch().then(increment).then(root);
    }
    fn increment(task: &mut TaskBuilder) {
        task.precondition(|c: &Count| c.0 < 3)
            .effect(|c: &mut Count| c.0 += 1);
    }
    fn act(task: &mut TaskBuilder) {
        task.precondition(|c: &Count| c.0 >= 2)
            .effect(|d: &mut Done| d.0 = true);
    }
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct Count(pub i32);
    let domain = HtnDomain::from_root(root).build().unwrap();

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain));
    let entity = world
        .spawn((Count(0), Done(false), HtnAgent::default()))
        .id();

    // Tick 1: plan [increment, increment, act]; the first increment runs
    // (count 1, cursor 1).
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Count>(entity).unwrap().0, 1);

    // Drift: an external system undoes the executed increment (the hammer
    // knocked back out of the pocket).
    world.get_mut::<Count>(entity).unwrap().0 = 0;

    // Tick 2: the plan's second increment still validates (count < 3) and
    // runs — count back to 1, cursor 2.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Count>(entity).unwrap().0, 1);

    // Tick 3: `act` drifts (count < 2) → dropped → replanned same tick: the
    // fresh plan is [increment, act] — and the increment is NOT quiet (1 → 2
    // changes bytes), so it re-executes **within the drift tick**. (A
    // structural elision that skipped the re-derived increment would
    // livelock: count stuck at 1, `act` unreachable, forever.)
    htn_ai_system(&mut world);
    assert_eq!(
        world.get::<Count>(entity).unwrap().0,
        2,
        "the state-changing step re-executed in the drift tick itself"
    );
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert_eq!(agent.cursor(), 1, "the fresh plan is mid-flight");

    // Tick 4: `act` validates and finishes.
    htn_ai_system(&mut world);
    assert!(world.get::<Done>(entity).unwrap().0);
    assert!(world.get::<HtnAgent>(entity).unwrap().plan().is_none());
}

/// A plan whose every step is quiet does nothing: it degrades to planless
/// instead of burning ticks executing no-ops (and never wedges — the next
/// tick simply replans the same way).
#[test]
fn fully_quiet_plans_degrade_to_planless() {
    fn root(task: &mut TaskBuilder) {
        task.branch().precondition(|d: &Done| d.0); // unreachable: done is false
        task.branch().then(ensure).then(root);
    }
    fn ensure(task: &mut TaskBuilder) {
        task.effect(|g: &mut Gate| g.0 = true);
    }
    let domain = HtnDomain::from_root(root).build().unwrap();

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain).with_sanity_limit(10));
    let entity = world
        .spawn((Gate(true), Done(false), HtnAgent::default()))
        .id();

    // The only decomposable branch recurses on `ensure`, whose effect is
    // already reflected — the truncated plan is entirely quiet. Elision
    // empties it into planlessness; the agent idles instead of ticking
    // through ten no-ops.
    for tick in 1..=3 {
        htn_ai_system(&mut world);
        let agent = world.get::<HtnAgent>(entity).unwrap();
        assert!(agent.plan().is_none(), "tick {tick}: planless, not theater");
        assert_eq!(agent.cursor(), 0);
        assert!(
            world.get::<Gate>(entity).unwrap().0,
            "gate never re-toggled"
        );
    }
}

/// `AgentRoot` selects the archetype root per agent: one domain, two NPC
/// kinds — a default agent plans from the domain root, an agent carrying
/// [`AgentRoot`](bevy_bhtn::ecs::AgentRoot) plans the registered extra root
/// instead. An out-of-bounds index is an ordinary planning error (the agent
/// idles), not a wedge.
#[test]
fn agent_root_selects_the_archetype_root() {
    fn charge(task: &mut TaskBuilder) {
        task.branch().then(gather);
    }
    fn gather(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 += 1);
    }
    fn drain_root(task: &mut TaskBuilder) {
        task.branch().then(drain);
    }
    fn drain(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 -= 1);
    }
    let domain = HtnDomain::from_root(charge)
        .add_root(drain_root)
        .build()
        .expect("archetype domain is well-formed");
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain));
    let default_agent = world.spawn((Battery(0), HtnAgent::default())).id();
    let second = world.resource::<HtnConfig>().domain.task_index(drain_root).unwrap() as u32;
    let drain_agent = world
        .spawn((Battery(5), HtnAgent::default(), AgentRoot(second)))
        .id();

    htn_ai_system(&mut world);
    // The default agent runs the domain root's behavior (gather: 0 -> 1);
    // the archetype agent runs the extra root's (drain: 5 -> 4).
    assert_eq!(world.get::<Battery>(default_agent).unwrap().0, 1);
    assert_eq!(world.get::<Battery>(drain_agent).unwrap().0, 4);

    // An out-of-bounds archetype index is an ordinary planning error: the
    // agent idles planless instead of wedging.
    let broken = world
        .spawn((Battery(0), HtnAgent::default(), AgentRoot(9_999)))
        .id();
    htn_ai_system(&mut world);
    assert!(world.get::<HtnAgent>(broken).unwrap().plan().is_none());
    assert_eq!(world.get::<Battery>(broken).unwrap().0, 0);
}
