//! Pins for the GTN compilation module: task sharing (the `fin_o` scheme of
//! Alford et al., Theorem 4.8) and task insertion, both compiled at bake time
//! into the plain HTN network. The compiled network must plan and execute
//! like a hand-written domain — same planner, same look-ahead, same driver.

use bevy_bhtn::gtn::SharedMarks;
use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{TaskBuilder, TaskFn};
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::Component;

/// `HtnPlanner::plan` with the root fn-item type inferred from the fn value
/// (fn-item types cannot be named directly in turbofish).
fn plan_root<F: TaskFn>(planner: &mut HtnPlanner, _root: F, state: &PlanState) -> Plan {
    planner.plan(_root, state).expect("plan")
}

// ---------------------------------------------------------------------------
// Task sharing — Theorem 4.8
// ---------------------------------------------------------------------------

mod share_tasks {
    use super::*;

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct Equipped(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct Trailhead(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct Summit(pub bool);

    /// The "done once" pattern the sharing contract requires: after a real
    /// execution the precondition no longer holds.
    pub fn equip(task: &mut TaskBuilder) {
        task.precondition(|e: &Equipped| !e.0)
            .effect(|e: &mut Equipped| e.0 = true);
    }
    pub fn hike(task: &mut TaskBuilder) {
        task.precondition(|e: &Equipped| e.0)
            .effect(|t: &mut Trailhead| t.0 = true);
    }
    pub fn climb(task: &mut TaskBuilder) {
        task.precondition(|t: &Trailhead| t.0)
            .effect(|s: &mut Summit| s.0 = true);
    }
}

#[test]
fn shared_task_appears_once_in_the_plan() {
    use share_tasks::*;
    fn root(task: &mut TaskBuilder) {
        task.branch().then(equip).then(hike).then(equip).then(climb);
    }

    let domain = HtnDomain::from_root(root)
        .share_task(equip)
        .build()
        .unwrap();
    // The wrapper compound exists; the user-facing name still resolves to the
    // (marker-carrying) primitive.
    assert!(domain.get_task("gtn/shared:equip").is_some());
    assert!(domain.get_task("equip").is_some());

    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_root(&mut planner, root, &state);
    assert!(plan.is_complete());
    // First occurrence runs `do`; the second takes the empty `done` method.
    assert_eq!(plan.task_names(&domain), ["equip", "hike", "climb"]);

    // Execution: only the user's own `effects` are committed (the driver's
    // contract); the marker is an expected effect — never committed — so the
    // executed state is exactly what the user's own effects produced.
    let mut executed = state.clone();
    for &s in plan.steps() {
        if let bevy_bhtn::Task::Primitive(p) = &domain.tasks[s as usize] {
            for e in &p.effects {
                e.apply(&mut executed);
            }
        }
    }
    let equipped = domain.components.slot_of::<Equipped>().unwrap();
    assert!(executed.get_slot::<Equipped>(equipped).0);
    let marks = domain.components.slot_of::<SharedMarks>().unwrap();
    assert!(executed.get_slot::<SharedMarks>(marks).0.is_empty());
}

#[test]
fn distinct_shared_tasks_track_distinct_markers() {
    use share_tasks::*;
    fn root(task: &mut TaskBuilder) {
        task.branch()
            .then(equip)
            .then(hike)
            .then(equip)
            .then(hike)
            .then(climb)
            .then(climb);
    }

    let domain = HtnDomain::from_root(root)
        .share_task(equip)
        .share_task(hike)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_root(&mut planner, root, &state);
    assert!(plan.is_complete());
    // The shared tasks are deduped independently; the unshared `climb` still
    // appears once per occurrence.
    assert_eq!(plan.task_names(&domain), ["equip", "hike", "climb", "climb"]);
}

#[test]
fn sharing_a_compound_is_a_builder_error() {
    use share_tasks::*;
    fn root(task: &mut TaskBuilder) {
        task.branch().then(hike);
    }
    let err = HtnDomain::from_root(root)
        .share_task(root)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("not a primitive"));
}

// ---------------------------------------------------------------------------
// Task insertion — gap compilation
// ---------------------------------------------------------------------------

mod repair_tasks {
    use super::*;

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct Traveled(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct HasKey(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct Escaped(pub bool);

    /// Precondition-gated so repeated insertion cannot loop.
    pub fn travel(task: &mut TaskBuilder) {
        task.precondition(|t: &Traveled| !t.0)
            .effect(|t: &mut Traveled| t.0 = true);
    }
    /// Not referenced by any method body: only insertion can reach it.
    pub fn pick_key(task: &mut TaskBuilder) {
        task.precondition(|k: &HasKey| !k.0)
            .effect(|k: &mut HasKey| k.0 = true);
    }
    pub fn escape(task: &mut TaskBuilder) {
        task.precondition(|k: &HasKey| k.0)
            .effect(|e: &mut Escaped| e.0 = true);
    }
    pub fn root(task: &mut TaskBuilder) {
        task.branch().then(travel).then(escape);
    }
}

#[test]
fn insertion_repairs_a_plan_that_would_otherwise_fail() {
    use repair_tasks::*;
    // Without insertion the chain is unsatisfiable: nothing sets `HasKey`.
    // That is a genuine dead end — reported as `NoPlan`, never as an empty
    // `Complete` plan (the legibility contract).
    let plain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&plain.components).finish();
    let mut planner = HtnPlanner::new(&plain);
    assert!(
        matches!(planner.plan(root, &state), Err(bevy_bhtn::HtnError::NoPlan)),
        "no decomposition exists without insertion"
    );

    // With insertion (and `pick_key` registered as insertable) the search
    // backtracks into a gap and repairs with `pick_key` — which no method
    // body references.
    let repaired = HtnDomain::from_root(root)
        .insertable(pick_key)
        .with_insertion()
        .build()
        .unwrap();
    assert!(repaired.get_task("gtn/insert").is_some());
    let state = PlanState::build(&repaired.components).finish();
    let mut planner = HtnPlanner::new(&repaired);
    let plan = plan_root(&mut planner, root, &state);
    assert!(plan.is_complete());
    assert_eq!(plan.task_names(&repaired), ["travel", "pick_key", "escape"]);
}

#[test]
fn insertion_leaves_clean_plans_clean() {
    use repair_tasks::*;
    // `pick_key` precondition-gates `escape`, but a domain whose own body
    // already satisfies everything must plan identically with and without
    // insertion (the stop method is first; gaps are repair-only).
    fn plain_root(task: &mut TaskBuilder) {
        task.branch().then(pick_key).then(escape);
    }
    let plain = HtnDomain::from_root(plain_root).build().unwrap();
    let with_gaps = HtnDomain::from_root(plain_root)
        .insertable(pick_key)
        .with_insertion()
        .build()
        .unwrap();

    for domain in [&plain, &with_gaps] {
        let state = PlanState::build(&domain.components).finish();
        let mut planner = HtnPlanner::new(domain);
        let plan = plan_root(&mut planner, plain_root, &state);
        assert!(plan.is_complete());
        assert_eq!(plan.task_names(domain), ["pick_key", "escape"]);
    }
}

#[test]
fn insertion_respects_sharing() {
    use share_tasks::*;
    fn root(task: &mut TaskBuilder) {
        task.branch().then(climb2);
    }
    fn climb2(task: &mut TaskBuilder) {
        task.precondition(|e: &Equipped| e.0)
            .effect(|s: &mut Summit2| s.0 = true);
    }
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct Summit2(pub bool);

    // `equip` is never referenced by a body: the only way to run it is
    // insertion, and the candidate routes through the shared wrapper (the
    // raw primitive is not a candidate). The plan is the inserted `equip`
    // followed by the climb — exactly once, since a second wrapper take
    // would hit the marker's `done` method (zero steps).
    let domain = HtnDomain::from_root(root)
        .insertable(equip)
        .share_task(equip)
        .with_insertion()
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_root(&mut planner, root, &state);
    assert!(plan.is_complete());
    assert_eq!(plan.task_names(&domain), ["equip", "climb2"]);
}
