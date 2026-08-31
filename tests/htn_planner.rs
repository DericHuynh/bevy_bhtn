//! Planner + AI logic tests using the shared [`HtnTestBed`].
//!
//! Exercises **forward planning** (MTR decomposition + backtracking), **backward
//! / goal-state planning**, and **plan validity** (execution: applying a plan's
//! task effects in order reaches the terminal state).
//!
//! Domains are deliberately *terminating* so assertions are exact and robust —
//! unbounded recursion (e.g. a miner that re-loops until a gold target) is the
//! reference `bevy_htn`'s shape and is covered by the builder-fixture tests,
//! not by exact-plan assertions.

mod common;
use common::HtnTestBed;

use bevy_bhtn::prelude::*;
use bevy_ecs::prelude::Component;
use ustr::Ustr;

// ---------------------------------------------------------------------------
// Travel domain — mirrors the classic bevy_htn `test_travel_htn`.
// Two root methods, exercises backtracking (walk when close, taxi when far).
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Copy, Default, Debug, PartialEq, Eq)]
enum Spot {
    #[default]
    Home,
    #[allow(dead_code)]
    Other,
    Park,
}

#[derive(Component, Clone, Default, Debug)]
struct Cash(pub i32);
#[derive(Component, Clone, Default, Debug)]
struct DistanceToPark(pub i32);
#[derive(Component, Clone, Default, Debug)]
struct Happy(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct MyLocation(pub Spot);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct TaxiLocation(pub Spot);

fn go_to_park(task: &mut TaskBuilder) {
    task.branch().then(walk);
    task.branch().then(call_taxi).then(ride_taxi).then(pay_taxi);
}

fn walk(task: &mut TaskBuilder) {
    task.precondition(|d: &DistanceToPark| d.0 <= 4)
        .precondition(|loc: &MyLocation| loc.0 != Spot::Park)
        .precondition(|h: &Happy| !h.0)
        .effect(|loc: &mut MyLocation| loc.0 = Spot::Park)
        .effect(|h: &mut Happy| h.0 = true);
}

fn call_taxi(task: &mut TaskBuilder) {
    task.precondition(|c: &Cash| c.0 >= 1)
        .effect(|taxi: &mut TaxiLocation, my: &mut MyLocation| taxi.0 = my.0);
}

fn ride_taxi(task: &mut TaskBuilder) {
    task.precondition(|taxi: &TaxiLocation, my: &MyLocation| taxi.0 == my.0)
        .precondition(|c: &Cash| c.0 >= 1)
        .effect(|taxi: &mut TaxiLocation| taxi.0 = Spot::Park)
        .effect(|my: &mut MyLocation| my.0 = Spot::Park)
        .effect(|h: &mut Happy| h.0 = true);
}

fn pay_taxi(task: &mut TaskBuilder) {
    task.precondition(|taxi: &TaxiLocation| taxi.0 == Spot::Park)
        .precondition(|c: &Cash| c.0 >= 1)
        .effect(|c: &mut Cash| c.0 -= 1);
}

fn travel_domain() -> HtnDomain {
    HtnDomain::from_root(go_to_park)
        .build()
        .expect("travel domain is well-formed")
}

/// A travel scratchpad: `cash` and `distance_to_park` set, everything else at
/// its default (home, unhappy, no taxi).
fn travel_state(domain: &HtnDomain, cash: i32, distance_to_park: i32) -> PlanState {
    PlanState::build(&domain.components)
        .set(Cash(cash))
        .set(DistanceToPark(distance_to_park))
        .finish()
}

// ---------------------------------------------------------------------------
// Forward planning (deterministic, terminating)
// ---------------------------------------------------------------------------

#[test]
fn forward_plans_walk_when_close() {
    let bed = HtnTestBed::new(travel_domain(), "go_to_park");
    let state = travel_state(bed.domain(), 0, 1);
    assert_eq!(bed.plan_forward(&state), vec![Ustr::from("walk")]);
}

#[test]
fn forward_plans_taxi_when_far() {
    let bed = HtnTestBed::new(travel_domain(), "go_to_park");
    let state = travel_state(bed.domain(), 10, 9);
    // Walk fails (too far) -> backtracks -> taxi succeeds.
    assert_eq!(
        bed.plan_forward(&state),
        vec![
            Ustr::from("call_taxi"),
            Ustr::from("ride_taxi"),
            Ustr::from("pay_taxi")
        ]
    );
}

#[test]
fn forward_plan_is_terminal_and_executes() {
    let bed = HtnTestBed::new(travel_domain(), "go_to_park");
    let mut state = travel_state(bed.domain(), 10, 9);
    let plan = bed.plan_forward(&state);
    assert_eq!(plan.len(), 3);

    // Execute: apply each planned task's effects in order.
    for name in &plan {
        let Some(Task::Primitive(p)) = bed.domain().get_task(name) else {
            panic!("planned task `{name}` missing");
        };
        for e in p.effects.iter() {
            e.apply(&mut state);
        }
    }

    let my_location = bed.domain().components.get::<MyLocation>().unwrap();
    let happy = bed.domain().components.get::<Happy>().unwrap();
    let cash = bed.domain().components.get::<Cash>().unwrap();
    // Terminal state: at the park, happy, taxi paid for.
    assert_eq!(state.get::<MyLocation>(my_location).0, Spot::Park);
    assert!(state.get::<Happy>(happy).0);
    assert_eq!(state.get::<Cash>(cash).0, 9);
}

// ---------------------------------------------------------------------------
// Forward planning: an already-satisfied goal yields an empty plan.
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug)]
struct Powered(pub bool);

fn switch_on(task: &mut TaskBuilder) {
    task.effect(|p: &mut Powered| p.0 = true);
}

fn ensure_on(task: &mut TaskBuilder) {
    task.branch().precondition(|p: &Powered| p.0);
    task.branch().then(switch_on);
}

fn idempotent_domain() -> HtnDomain {
    HtnDomain::from_root(ensure_on)
        .build()
        .expect("idempotent domain is well-formed")
}

#[test]
fn forward_plan_returns_empty_when_goal_already_met() {
    let bed = HtnTestBed::new(idempotent_domain(), "ensure_on");
    let on = PlanState::build(&bed.domain().components)
        .set(Powered(true))
        .finish();
    assert!(bed.plan_forward(&on).is_empty());
    let off = PlanState::build(&bed.domain().components)
        .set(Powered(false))
        .finish();
    assert_eq!(bed.plan_forward(&off), vec![Ustr::from("switch_on")]);
}

// ---------------------------------------------------------------------------
// Backward (goal-state) planning
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug)]
struct HasOre(pub bool);
#[derive(Component, Clone, Default, Debug)]
struct HasRope(pub bool);

fn mine(task: &mut TaskBuilder) {
    task.precondition(|o: &HasOre| !o.0)
        .effect(|o: &mut HasOre| o.0 = true);
}

fn ore_root(task: &mut TaskBuilder) {
    task.branch().then(mine);
}

fn have_ore(goal: &mut GoalBuilder) {
    goal.effect(|o: &mut HasOre| o.0 = true);
}

fn goal_domain() -> HtnDomain {
    HtnDomain::from_root(ore_root)
        .goal(have_ore)
        .build()
        .expect("goal domain is well-formed")
}

#[test]
fn backward_plan_finds_satisfying_leaf() {
    let bed = HtnTestBed::new(goal_domain(), "ore_root");
    let state = PlanState::build(&bed.domain().components).finish();
    let plan = bed
        .plan_backward("have_ore", &state)
        .expect("back plan reaches goal");
    assert_eq!(plan, vec![Ustr::from("mine")]);
}

fn mine_bare(task: &mut TaskBuilder) {
    task.effect(|o: &mut HasOre| o.0 = true);
}

fn unreachable_root(task: &mut TaskBuilder) {
    task.branch().then(mine_bare);
}

fn want_more(goal: &mut GoalBuilder) {
    goal.effect(|o: &mut HasOre| o.0 = true)
        .effect(|r: &mut HasRope| r.0 = true);
}

#[test]
fn backward_plan_rejects_unreachable_goal() {
    let domain = HtnDomain::from_root(unreachable_root)
        .goal(want_more)
        .build()
        .expect("unreachable-goal domain is well-formed");
    let bed = HtnTestBed::new(domain, "unreachable_root");
    let state = PlanState::build(&bed.domain().components).finish();
    // `has_rope` is never produced by any primitive -> goal unreachable.
    assert!(bed.plan_backward("want_more", &state).is_err());
}

// ---------------------------------------------------------------------------
// Task-function-recorded condition/effect details + evaluation
// ---------------------------------------------------------------------------

#[test]
fn task_functions_record_conditions_and_effects() {
    let bed = HtnTestBed::new(travel_domain(), "go_to_park");
    let Some(Task::Primitive(walk)) = bed.domain().get_task("walk") else {
        panic!("walk primitive missing");
    };
    // The `walk` task function recorded its three preconditions and two
    // effects, in declaration order.
    assert_eq!(walk.preconditions.len(), 3);
    assert_eq!(walk.effects.len(), 2);

    // Conditions evaluate against the scratchpad: close + not at the park +
    // unhappy is pickable; standing at the park is not.
    let near = travel_state(bed.domain(), 10, 3);
    assert!(walk.preconditions_met(&near));
    let at_park = PlanState::build(&bed.domain().components)
        .set(MyLocation(Spot::Park))
        .finish();
    assert!(!walk.preconditions_met(&at_park));

    // Effects apply to the scratchpad: walking lands at the park, happy.
    let mut state = travel_state(bed.domain(), 10, 3);
    for e in walk.effects.iter() {
        e.apply(&mut state);
    }
    let my_location = bed.domain().components.get::<MyLocation>().unwrap();
    let happy = bed.domain().components.get::<Happy>().unwrap();
    assert_eq!(state.get::<MyLocation>(my_location).0, Spot::Park);
    assert!(state.get::<Happy>(happy).0);
}

#[test]
fn conditions_evaluate_against_state() {
    let bed = HtnTestBed::new(travel_domain(), "go_to_park");
    let Some(Task::Primitive(walk)) = bed.domain().get_task("walk") else {
        panic!("walk primitive missing");
    };
    // The first recorded precondition is the distance gate (`<= 4`).
    let near = travel_state(bed.domain(), 10, 3);
    let far = travel_state(bed.domain(), 10, 9);
    assert!(walk.preconditions[0].evaluate(&near));
    assert!(!walk.preconditions[0].evaluate(&far));
}

#[test]
fn effects_apply_to_state() {
    let bed = HtnTestBed::new(idempotent_domain(), "ensure_on");
    let Some(Task::Primitive(switch)) = bed.domain().get_task("switch_on") else {
        panic!("switch_on primitive missing");
    };
    let mut state = PlanState::build(&bed.domain().components).finish();
    for e in switch.effects.iter() {
        e.apply(&mut state);
    }
    let powered = bed.domain().components.get::<Powered>().unwrap();
    assert!(state.get::<Powered>(powered).0);
}
