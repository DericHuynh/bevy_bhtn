//! Pins for the Bevy ECS execution layer (`bevy_bhtn::ecs`): the [`HtnAgent`]
//! component, [`HtnConfig`], and the exclusive [`htn_ai_system`] driver's
//! plan → validate → execute-one-step → replan loop over real components.

use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig, PlanEvery};
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
    assert_eq!(agent.plan.as_ref().map(|p| p.len()), Some(3));
    assert_eq!(agent.cursor, 1);
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
    assert!(agent.plan.is_none(), "finished plan is dropped");
    assert_eq!(agent.cursor, 0);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);

    // Tick 4: replans; the terminal branch now matches and plans nothing.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan.is_none());
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}

/// The world drifted since planning (another system spent the resource): the
/// driver re-validates the next step's preconditions against the real world,
/// drops the plan, and the agent recovers by replanning once the world allows
/// progress again.
#[test]
fn world_drift_triggers_replan_and_recovery() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(charge_domain()));
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    // Tick 1: plans and executes one gather.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

    // Simulate drift: another system drains the battery below what the plan
    // assumed (the plan's next step expects battery < 3 — still true — so
    // instead invalidate by fully draining to 0 and saturating the plan's
    // assumption is unchanged; use a *different* kind of drift: remove the
    // component entirely, so the scratchpad sees Default(0) — still < 3).
    // The meaningful drift test: the plan's remaining steps assume battery
    // grows monotonically; drain it and confirm the driver still validates
    // against reality rather than replaying stale effects.
    world.get_mut::<Battery>(entity).unwrap().0 = 0;

    // Tick 2: the plan is still valid (battery < 3), so it executes —
    // validation is against the *real* world, not the stale plan state.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);

    // Now make the next step's precondition fail: saturate the battery so the
    // `gather` branch precondition (battery < 3) no longer holds.
    world.get_mut::<Battery>(entity).unwrap().0 = 3;

    // Tick 3: the plan's next gather is invalid → plan dropped (replan next
    // tick), no effect applied.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan.is_none(), "invalid step aborts the plan");
    assert_eq!(
        world.get::<Battery>(entity).unwrap().0,
        3,
        "no effect applied"
    );

    // Tick 4: replans; the terminal branch matches, nothing executes.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
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
    assert!(world.get::<HtnAgent>(a).unwrap().plan.is_none());
    assert!(world.get::<HtnAgent>(b).unwrap().plan.is_none());
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
    assert_eq!(agent.plan.as_ref().map(|p| p.len()), Some(3));
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
    let throttled = world.spawn((Hits(0), HtnAgent::default(), PlanEvery(3))).id();
    let free = world.spawn((Hits(0), HtnAgent::default())).id();

    for _ in 0..10 {
        htn_ai_system(&mut world);
    }

    assert_eq!(
        world.get::<Hits>(throttled).unwrap().0,
        4,
        "plans at runs 1, 4, 7, 10"
    );
    assert_eq!(world.get::<Hits>(free).unwrap().0, 10, "unthrottled control");
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
    assert!(agent.plan.is_none(), "no plan is ever stored");
    assert_eq!(agent.cursor, 0);
}
