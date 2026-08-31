//! Pins for the Bevy ECS execution layer (`bevy_bhtn::ecs`): the [`HtnAgent`]
//! component, [`HtnConfig`], and the exclusive [`htn_ai_system`] driver's
//! plan → validate → execute-one-step → replan loop over real components.

use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
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
    cfg.lookahead = false;
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
