//! Pins for PausePlan (`MethodBuilder::pause_plan`) and the driver's
//! resume-from-pause flow: a pause marker truncates the compiled plan at a
//! fixed member position, the still-queued work travels in the plan's
//! [`ResumePoint`](bevy_bhtn::planner::ResumePoint) (tasks + committed MTR),
//! and [`HtnPlanner::resume`](bevy_bhtn::planner::HtnPlanner::resume)
//! decomposes it against the state the world is in *then* — never
//! re-decomposing (or backtracking into) the executed prefix.

use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
use bevy_bhtn::mcts::MctsSearcher;
use bevy_bhtn::planner::{HtnPlanner, PlanStatus, ResumePoint, ResumeStep};
use bevy_bhtn::selection::HtnSearchStrategy;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::TaskBuilder;
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;
use std::sync::Arc;

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Wood(i32);

fn take_a(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 += 1);
}
fn take_b(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 += 10);
}
fn take_c(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 += 100);
}
/// Unsatisfiable in every test state: the precondition reads a fixed
/// component against an impossible bound.
fn need_rich(task: &mut TaskBuilder) {
    task.precondition(|g: &Gold| g.0 >= 1000).effect(|g: &mut Gold| g.0 += 1);
}

fn empty_state(domain: &HtnDomain) -> PlanState {
    PlanState::build(&domain.components).finish()
}

fn state_with(domain: &HtnDomain, gold: i32) -> PlanState {
    PlanState::build(&domain.components)
        .set(Gold(gold))
        .finish()
}

/// The three-leg domain: each pause truncates the compiled plan, and the
/// marker itself travels in the resume point so the resumed search stops at
/// the next one too.
fn legs_domain() -> HtnDomain {
    fn legs(task: &mut TaskBuilder) {
        task.branch()
            .then(take_a)
            .pause_plan()
            .then(take_b)
            .pause_plan()
            .then(take_c);
    }
    HtnDomain::from_root(legs).build().unwrap()
}

#[test]
fn pause_marker_truncates_the_plan_and_records_the_resume_point() {
    let domain = legs_domain();
    let state = empty_state(&domain);
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(domain.root, &state).expect("paused plan");

    assert_eq!(plan.status(), PlanStatus::Paused);
    assert!(plan.is_paused());
    assert_eq!(plan.task_names(&domain), vec!["take_a"]);
    let b = domain.task_index(take_b).unwrap() as u32;
    let c = domain.task_index(take_c).unwrap() as u32;
    let resume = plan.resume().expect("resume point");
    // The still-queued work, in execution order, chained pause included.
    assert_eq!(
        resume.tasks,
        vec![ResumeStep::Task(b), ResumeStep::Pause, ResumeStep::Task(c)]
    );
    // The MTR at the pause: the method choices down to the marked method.
    assert_eq!(resume.mtr, vec![0]);
    assert_eq!(plan.mtr(), vec![0usize]);
}

#[test]
fn chained_pauses_plan_one_leg_per_resume() {
    let domain = legs_domain();
    let state = empty_state(&domain);
    let mut planner = HtnPlanner::new(&domain);

    // Leg 1 was planned by the initial call (the pinned truncation test
    // covers its shape); resume into leg 2.
    let plan = planner.plan(domain.root, &state).expect("paused plan");
    let resumed = planner
        .resume(plan.resume().as_ref().expect("resume point"), &state)
        .expect("second leg");
    assert_eq!(resumed.status(), PlanStatus::Paused);
    assert_eq!(resumed.task_names(&domain), vec!["take_b"]);
    let c = domain.task_index(take_c).unwrap() as u32;
    assert_eq!(
        resumed.resume().as_ref().expect("resume point").tasks,
        vec![ResumeStep::Task(c)]
    );

    // And into leg 3, which completes.
    let last_leg = planner
        .resume(resumed.resume().as_ref().expect("resume point"), &state)
        .expect("final leg");
    assert_eq!(last_leg.status(), PlanStatus::Complete);
    assert!(last_leg.resume().is_none());
    assert_eq!(last_leg.task_names(&domain), vec!["take_c"]);
}

/// A pause after a method's last member defers everything still queued
/// *behind the method* — the ancestor suffix, not just the method's own
/// remainder.
#[test]
fn pause_after_the_last_member_defers_the_ancestor_suffix() {
    fn leg(task: &mut TaskBuilder) {
        task.branch().then(take_a).pause_plan();
    }
    fn outer(task: &mut TaskBuilder) {
        task.branch().then(leg).then(take_c);
    }
    let domain = HtnDomain::from_root(outer).build().unwrap();
    let state = empty_state(&domain);
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(domain.root, &state).expect("paused plan");

    assert_eq!(plan.status(), PlanStatus::Paused);
    assert_eq!(plan.task_names(&domain), vec!["take_a"]);
    let c = domain.task_index(take_c).unwrap() as u32;
    assert_eq!(
        plan.resume().as_ref().expect("resume point").tasks,
        vec![ResumeStep::Task(c)]
    );
    // Both method commitments (outer's and leg's) are in the recorded chain.
    assert_eq!(plan.mtr(), vec![0usize, 0]);

    let resumed = planner
        .resume(plan.resume().as_ref().unwrap(), &state)
        .expect("suffix");
    assert_eq!(resumed.status(), PlanStatus::Complete);
    assert_eq!(resumed.task_names(&domain), vec!["take_c"]);
    // The committed chain is seeded into the resumed plan's MTR.
    assert_eq!(resumed.mtr(), vec![0usize, 0]);
}

/// Resume decomposes against the state it is *given* — the state the world
/// will be in when the prefix has run — not the state the pause was taken
/// at. The suffix's method choice flips with it.
#[test]
fn resume_replans_the_suffix_against_the_state_it_is_given() {
    fn chooser(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|g: &Gold| g.0 >= 10)
            .then(take_c);
        task.branch().then(take_b);
    }
    fn rooted(task: &mut TaskBuilder) {
        task.branch().then(take_a).pause_plan().then(chooser);
    }
    let domain = HtnDomain::from_root(rooted).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);

    let plan = planner
        .plan(domain.root, &state_with(&domain, 0))
        .expect("paused plan");
    // Paused with the chooser unexamined (it is behind the marker).
    assert_eq!(plan.task_names(&domain), vec!["take_a"]);

    // Resume in the poor world: the chooser's rich branch fails, the plain
    // branch runs.
    let poor = planner
        .resume(plan.resume().as_ref().unwrap(), &state_with(&domain, 0))
        .expect("poor suffix");
    assert_eq!(poor.task_names(&domain), vec!["take_b"]);

    // Resume in the rich world: the chooser's rich branch applies.
    let rich = planner
        .resume(plan.resume().as_ref().unwrap(), &state_with(&domain, 15))
        .expect("rich suffix");
    assert_eq!(rich.task_names(&domain), vec!["take_c"]);
}

/// The remaining work has no decomposition in the resume state: `NoPlan` —
/// the caller's recovery is a fresh plan from the root, like any other
/// failed plan.
#[test]
fn resume_fails_with_no_plan_when_the_suffix_cannot_decompose() {
    fn doomed_tail(task: &mut TaskBuilder) {
        task.branch().then(take_a).pause_plan().then(need_rich);
    }
    let domain = HtnDomain::from_root(doomed_tail).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner
        .plan(domain.root, &state_with(&domain, 0))
        .expect("paused plan");
    assert_eq!(plan.status(), PlanStatus::Paused);

    let err = planner
        .resume(plan.resume().as_ref().unwrap(), &state_with(&domain, 0))
        .unwrap_err();
    assert!(
        matches!(err, bevy_bhtn::HtnError::NoPlan),
        "unsatisfiable suffix is NoPlan, got {err:?}"
    );
}

#[test]
fn resume_rejects_out_of_bounds_task_indices() {
    let domain = legs_domain();
    let state = empty_state(&domain);
    let mut planner = HtnPlanner::new(&domain);
    let err = planner
        .resume(
            &ResumePoint {
                tasks: vec![ResumeStep::Task(9999)],
                mtr: Vec::new(),
            },
            &state,
        )
        .unwrap_err();
    match err {
        bevy_bhtn::HtnError::UnregisteredTask { type_name } => {
            assert_eq!(type_name, "<task index 9999>");
        }
        other => panic!("expected UnregisteredTask, got {other:?}"),
    }
}

/// The plan cut must sit at a fixed member position — a marker on a
/// partially-ordered branch is rejected at bake.
#[test]
fn pause_marker_on_a_partially_ordered_branch_is_a_build_error() {
    fn unordered_pause(task: &mut TaskBuilder) {
        let mut mb = task.branch();
        mb.subtask(take_a);
        mb.subtask(take_b);
        mb.pause_plan();
    }
    let err = HtnDomain::from_root(unordered_pause).build().unwrap_err();
    assert!(
        err.to_string().contains("pause_plan"),
        "error names the pause marker: {err}"
    );
}

/// A pause with no work queued behind it is vacuous: the plan completes.
#[test]
fn vacuous_pause_with_no_work_after_it_completes() {
    fn tail_pause(task: &mut TaskBuilder) {
        task.branch().then(take_a).then(take_b).pause_plan();
    }
    let domain = HtnDomain::from_root(tail_pause).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(domain.root, &empty_state(&domain)).expect("plan");
    assert_eq!(plan.status(), PlanStatus::Complete);
    assert_eq!(plan.task_names(&domain), vec!["take_a", "take_b"]);
    assert!(plan.resume().is_none());
}

/// The look-ahead sweep proves only the PRE-pause prefix: a doomed step
/// *behind* the marker does not refute the method (the far side is not
/// planned in this plan — over-validating it is what the marker forbids),
/// while a doomed step *before* the marker still skips it.
#[test]
fn lookahead_sweep_stops_at_the_pause_marker() {
    // Post-pause doom tolerated: branch 0 is chosen even though the step
    // behind its marker can never run.
    fn doomed_suffix(task: &mut TaskBuilder) {
        task.branch()
            .then(take_a)
            .pause_plan()
            .then(need_rich);
        task.branch().then(take_c);
    }
    let domain = HtnDomain::from_root(doomed_suffix).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(domain.root, &state_with(&domain, 0)).expect("plan");
    assert_eq!(plan.status(), PlanStatus::Paused);
    assert_eq!(plan.task_names(&domain), vec!["take_a"]);

    // Pre-pause doom refuted: the sweep skips the marked method entirely.
    fn doomed_prefix(task: &mut TaskBuilder) {
        task.branch()
            .then(need_rich)
            .pause_plan()
            .then(take_a);
        task.branch().then(take_a);
    }
    let domain = HtnDomain::from_root(doomed_prefix).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(domain.root, &state_with(&domain, 0)).expect("plan");
    assert_eq!(plan.status(), PlanStatus::Complete);
    assert_eq!(plan.task_names(&domain), vec!["take_a"]);
    assert_eq!(plan.mtr(), vec![1usize], "the marked method was skipped");
}

/// Pause markers shape the built-in search only: a `Custom` searcher owns
/// its search and plans straight through them.
#[test]
fn custom_searcher_plans_through_pause_markers() {
    let domain = legs_domain();
    let state = empty_state(&domain);
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::Custom(Arc::new(MctsSearcher::new(64))));
    let plan = planner.plan(domain.root, &state).expect("mcts plan");
    assert_eq!(plan.status(), PlanStatus::Complete);
    assert_eq!(
        plan.task_names(&domain),
        vec!["take_a", "take_b", "take_c"]
    );
}

// ---------------------------------------------------------------------------
// Driver: execute the prefix, then resume from the pause
// ---------------------------------------------------------------------------

fn journey_domain() -> HtnDomain {
    fn journey(task: &mut TaskBuilder) {
        task.branch().precondition(|g: &Gold| g.0 >= 3);
        task.branch()
            .then(take_one)
            .pause_plan()
            .then(take_one)
            .then(take_one);
    }
    fn take_one(task: &mut TaskBuilder) {
        task.effect(|g: &mut Gold| g.0 += 1);
    }
    HtnDomain::from_root(journey).build().unwrap()
}

/// The agent executes the paused plan's prefix, keeps the exhausted plan,
/// resumes the decomposition on the next run, and completes — one step per
/// tick throughout, gold accumulating exactly once per step.
#[test]
fn driver_executes_prefix_then_resumes_from_the_pause() {
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(journey_domain()));
    let entity = world.spawn((Gold(0), HtnAgent::default())).id();

    // Tick 1: plans the first leg ([take_one], paused) and executes it. The
    // exhausted paused plan is KEPT (not dropped like a finished Complete).
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    let plan = agent.plan().expect("paused plan survives the tick");
    assert_eq!(plan.status(), PlanStatus::Paused);
    assert_eq!(agent.cursor(), 1, "the prefix's single step executed");
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 1);

    // Tick 2: the driver resumes decomposition from the pause (same committed
    // chain — no replan from the root) and executes the resumed plan's first
    // step.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    let plan = agent.plan().expect("resumed plan running");
    assert_eq!(plan.status(), PlanStatus::Complete);
    assert_eq!(plan.len(), 2, "the remaining leg was planned");
    assert_eq!(agent.cursor(), 1);
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 2);

    // Tick 3: the resumed plan's last step completes it and the plan drops.
    htn_ai_system(&mut world);
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert!(agent.plan().is_none(), "completed plan is dropped");
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 3);

    // Tick 4: replans from the root — the terminal branch matches and the
    // agent rests (the pause never re-executes completed work).
    htn_ai_system(&mut world);
    assert!(world.get::<HtnAgent>(entity).unwrap().plan().is_none());
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 3);
}

/// Resume selects the suffix against reality as it is when the prefix has
/// run: a component changed between the prefix and the resume flips the
/// suffix's branch.
#[test]
fn driver_resume_selects_the_suffix_against_reality() {
    fn journey(task: &mut TaskBuilder) {
        task.branch().precondition(|g: &Gold| g.0 >= 6);
        task.branch().then(take_one).pause_plan().then(chooser);
    }
    fn take_one(task: &mut TaskBuilder) {
        task.effect(|g: &mut Gold| g.0 += 1);
    }
    fn chooser(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|g: &Gold| g.0 >= 5)
            .then(bulk_haul);
        task.branch().then(scrap);
    }
    fn bulk_haul(task: &mut TaskBuilder) {
        task.effect(|w: &mut Wood| w.0 += 100);
    }
    fn scrap(task: &mut TaskBuilder) {
        task.effect(|w: &mut Wood| w.0 += 1);
    }
    let domain = HtnDomain::from_root(journey).build().unwrap();

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain));
    let entity = world.spawn((Gold(0), Wood(0), HtnAgent::default())).id();

    // Tick 1: plan the leg, execute its single step (Gold 0 → 1); the plan
    // pauses with the chooser queued behind the marker.
    htn_ai_system(&mut world);
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 1);
    assert_eq!(
        world.get::<HtnAgent>(entity).unwrap().plan().map(|p| p.status()),
        Some(PlanStatus::Paused)
    );

    // The world moves while the prefix runs: gold arrives from elsewhere.
    world.get_mut::<Gold>(entity).unwrap().0 = 5;

    // Tick 2: the resume sees Gold == 5 and picks the chooser's rich branch.
    htn_ai_system(&mut world);
    assert_eq!(
        world.get::<Wood>(entity).unwrap().0,
        100,
        "the resumed suffix selected bulk_haul against live reality"
    );
    assert_eq!(world.get::<Gold>(entity).unwrap().0, 5);
}
