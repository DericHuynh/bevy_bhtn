//! Pins for the plan lifecycle bridge ([`PlanEvent`] →
//! `Messages<PlanEvent>`): the driver reports plans installed and replaced
//! (Fluid HTN's `OnNewPlan` / `OnReplacePlan(old, new)`), steps that failed
//! re-validation against live reality (`OnCurrentTaskFailed`), and plans
//! that ran to completion — with zero planner-core involvement, and only
//! when `HtnConfig::plan_events` is enabled.

use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig, PlanEvent};
use bevy_bhtn::planner::PlanStatus;
use bevy_bhtn::tasks::TaskBuilder;
use bevy_bhtn::HtnDomain;
use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::prelude::*;

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Battery(i32);

/// One persistent reader per test, created after the first tick (a fresh
/// cursor replays everything still buffered — including the previous tick's
/// messages — so the standard pattern is a cursor that survives the
/// per-tick `update()` and sees each tick's events exactly once).
struct EventLog {
    cursor: MessageCursor<PlanEvent>,
}

impl EventLog {
    fn start(world: &mut World) -> Self {
        let messages = world.resource::<Messages<PlanEvent>>();
        Self {
            cursor: messages.get_cursor(),
        }
    }

    /// Positioned past everything already buffered — for tests that do not
    /// assert the first tick's events.
    fn after(world: &mut World) -> Self {
        let messages = world.resource::<Messages<PlanEvent>>();
        Self {
            cursor: messages.get_cursor_current(),
        }
    }

    fn read(&mut self, world: &mut World) -> Vec<PlanEvent> {
        let messages = world.resource::<Messages<PlanEvent>>();
        self.cursor.read(messages).cloned().collect()
    }
}

/// A plain execute-to-completion episode: `PlanReplaced { old: None }` when
/// the agent first plans, nothing while the plan runs, `PlanCompleted` when
/// the last step executes, and nothing on the idle replan that follows (an
/// empty terminal plan installs nothing).
#[test]
fn plan_events_stream_the_lifecycle() {
    fn charge(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|battery: &Battery| battery.0 >= 3);
        task.branch()
            .precondition(|battery: &Battery| battery.0 < 3)
            .then(gather)
            .then(charge);
    }
    fn gather(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 += 1);
    }
    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(HtnDomain::from_root(charge).build().unwrap()).with_plan_events(true),
    );
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    // Tick 1: the agent plans (3x gather) and executes the first step.
    htn_ai_system(&mut world);
    let mut log = EventLog::start(&mut world);
    let events = log.read(&mut world);
    assert_eq!(events.len(), 2, "install + first step: {events:?}");
    match &events[0] {
        PlanEvent::PlanReplaced { entity: e, old, new } => {
            assert_eq!(*e, entity);
            assert!(old.is_none(), "the agent was planless: {old:?}");
            assert_eq!(new.len(), 3);
            assert_eq!(new.status(), PlanStatus::Complete);
        }
        other => panic!("expected PlanReplaced, got {other:?}"),
    }
    assert!(matches!(&events[1], PlanEvent::StepExecuted { entity: e, .. } if *e == entity));

    // Tick 2: mid-plan execution — only the per-step hook fires.
    htn_ai_system(&mut world);
    let events = log.read(&mut world);
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(matches!(events[0], PlanEvent::StepExecuted { .. }));

    // Tick 3: the last step completes the plan.
    htn_ai_system(&mut world);
    let events = log.read(&mut world);
    assert_eq!(events.len(), 2, "step + completion: {events:?}");
    assert!(matches!(events[0], PlanEvent::StepExecuted { .. }));
    assert!(matches!(events[1], PlanEvent::PlanCompleted { entity: e } if e == entity));
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);

    // Tick 4: replans — the terminal branch matches, nothing is installed,
    // and an empty plan reports nothing.
    htn_ai_system(&mut world);
    assert!(log.read(&mut world).is_empty());
}

/// A step that fails re-validation against the drifted world emits
/// `StepFailed`, and the same-tick repair replan emits `PlanReplaced` with
/// the dropped plan as `old` and the fresh plan as `new`.
#[test]
fn drift_emits_step_failed_then_a_replacement_plan() {
    fn adaptive(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|battery: &Battery| battery.0 >= 2)
            .then(recharge);
        task.branch().then(gather).then(gather);
    }
    fn gather(task: &mut TaskBuilder) {
        task.precondition(|battery: &Battery| battery.0 < 2)
            .effect(|battery: &mut Battery| battery.0 += 1);
    }
    fn recharge(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 = 5);
    }
    let domain = HtnDomain::from_root(adaptive).build().unwrap();
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain).with_plan_events(true));
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    // Tick 1: plans [gather, gather] and executes the first gather.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);
    let mut log = EventLog::after(&mut world);

    // Drift: another system charges the battery past the gather gate.
    world.get_mut::<Battery>(entity).unwrap().0 = 2;

    // Tick 2: the planned step fails re-validation (StepFailed), the driver
    // replans from reality in the same tick, and the fresh plan — the rich
    // branch's recharge — is reported as a replacement of the dropped plan.
    htn_ai_system(&mut world);
    let events = log.read(&mut world);
    assert_eq!(
        events.len(),
        4,
        "failure + replacement + step + completion: {events:?}"
    );
    match &events[0] {
        PlanEvent::StepFailed { entity: e, task: _, task_name } => {
            assert_eq!(*e, entity);
            assert_eq!(*task_name, "gather");
        }
        other => panic!("expected StepFailed first, got {other:?}"),
    }
    match &events[1] {
        PlanEvent::PlanReplaced { old, new, .. } => {
            let old = old.as_ref().expect("the drifted plan is the old plan");
            assert_eq!(old.len(), 2, "the interrupted two-gather plan");
            let config = world.resource::<HtnConfig>();
            assert_eq!(new.task_names(&config.domain), vec!["recharge"]);
        }
        other => panic!("expected PlanReplaced second, got {other:?}"),
    }
    // The replacement planned, executed, and completed within the same tick.
    assert!(matches!(&events[2], PlanEvent::StepExecuted { entity: e, task_name: "recharge", .. } if *e == entity));
    assert!(matches!(events[3], PlanEvent::PlanCompleted { entity: e } if e == entity));
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 5);
}

/// Resuming a paused plan reports `PlanReplaced` with the paused plan as
/// `old` — the pause consumed its prefix and handed the remaining work to a
/// new plan — and the resumed plan's own completion reports `PlanCompleted`.
#[test]
fn pause_resume_reports_the_paused_plan_as_the_replaced_old() {
    fn journey(task: &mut TaskBuilder) {
        task.branch().precondition(|battery: &Battery| battery.0 >= 3);
        task.branch()
            .then(step)
            .pause_plan()
            .then(step)
            .then(step);
    }
    fn step(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 += 1);
    }
    let domain = HtnDomain::from_root(journey).build().unwrap();
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain).with_plan_events(true));
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    // Tick 1: the leg is planned (paused) and its single step executes.
    htn_ai_system(&mut world);
    let mut log = EventLog::start(&mut world);
    let events = log.read(&mut world);
    assert_eq!(events.len(), 2, "install + step: {events:?}");
    let PlanEvent::PlanReplaced { old, new, .. } = &events[0] else {
        panic!("expected an install, got {events:?}")
    };
    assert!(old.is_none());
    assert_eq!(new.status(), PlanStatus::Paused);
    assert_eq!(new.len(), 1);

    // Tick 2: the resume replaces the paused plan with the completed
    // continuation (which executes its first step in the same tick).
    htn_ai_system(&mut world);
    let events = log.read(&mut world);
    assert_eq!(events.len(), 2, "resume + step: {events:?}");
    let PlanEvent::PlanReplaced { old, new, .. } = &events[0] else {
        panic!("expected a replacement, got {events:?}")
    };
    let old = old.as_ref().expect("the paused plan is the old plan");
    assert_eq!(old.status(), PlanStatus::Paused);
    assert_eq!(old.len(), 1);
    assert!(old.resume().is_some());
    assert_eq!(new.status(), PlanStatus::Complete);
    assert_eq!(new.len(), 2, "the remaining leg");
    assert!(new.resume().is_none());

    // Tick 3: the resumed plan completes.
    htn_ai_system(&mut world);
    let events = log.read(&mut world);
    assert_eq!(events.len(), 2, "step + completion: {events:?}");
    assert!(matches!(events[0], PlanEvent::StepExecuted { .. }));
    assert!(matches!(events[1], PlanEvent::PlanCompleted { .. }));
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}

/// The bridge is opt-in: without `with_plan_events`, no events are written
/// (the `Messages<PlanEvent>` resource still exists, matching the trace
/// bridge's always-allocated discipline).
#[test]
fn events_disabled_by_default() {
    fn charge(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|battery: &Battery| battery.0 >= 3);
        task.branch()
            .precondition(|battery: &Battery| battery.0 < 3)
            .then(gather);
    }
    fn gather(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 += 1);
    }
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(HtnDomain::from_root(charge).build().unwrap()));
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    htn_ai_system(&mut world);
    let mut log = EventLog::start(&mut world);
    htn_ai_system(&mut world);
    htn_ai_system(&mut world);

    assert_eq!(log.read(&mut world).len(), 0);
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}
