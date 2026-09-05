//! Nested-plan and replanning tests for `bevy_bhtn`.
//!
//! The planner is stateless — `HtnPlanner::plan` clones the state and returns a
//! fresh [`bevy_bhtn::planner::Plan`]. That makes "changing requirements
//! mid-execution" a matter of re-planning against an updated world state each
//! turn, which is exactly how a turn-based game (and the reference `bevy_htn`
//! "ReplanRequest" flow) consumes it. These tests pin that behaviour: deep
//! nesting decomposes in the right order, and a plan adapts when the world
//! changes between turns.

mod common;
use common::HtnTestBed;

use bevy_bhtn::prelude::*;
use bevy_ecs::prelude::Component;
use ustr::Ustr;

// ---------------------------------------------------------------------------
// A three-level nested domain: root -> middle -> leaves.
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
enum Room {
    #[default]
    Outside,
    Hall,
    #[allow(dead_code)]
    Vault,
}

#[derive(Component, Clone, Default, Debug)]
struct TorchLit(pub bool);
#[derive(Component, Clone, Default, Debug)]
struct ChestOpen(pub bool);

fn get_treasure(task: &mut TaskBuilder) {
    task.branch().then(traverse).then(open_chest);
}

fn traverse(task: &mut TaskBuilder) {
    task.branch().then(enter_hall).then(light_torch);
}

fn enter_hall(task: &mut TaskBuilder) {
    task.precondition(|room: &Room| *room == Room::Outside)
        .effect(|room: &mut Room| *room = Room::Hall);
}

fn light_torch(task: &mut TaskBuilder) {
    task.effect(|torch: &mut TorchLit| torch.0 = true);
}

fn open_chest(task: &mut TaskBuilder) {
    task.precondition(|torch: &TorchLit| torch.0)
        .precondition(|room: &Room| *room != Room::Outside)
        .effect(|chest: &mut ChestOpen| chest.0 = true);
}

fn nested_domain() -> HtnDomain {
    HtnDomain::from_root(get_treasure)
        .build()
        .expect("nested domain is well-formed")
}

#[test]
fn deep_nested_plan_decomposes_in_order() {
    let bed = HtnTestBed::new(nested_domain());
    let state = PlanState::build(&bed.domain().components).finish();
    let plan = bed.plan_forward(&state);
    // Level 1 (get_treasure) -> Level 2 (traverse) -> leaves, in order.
    assert_eq!(
        plan,
        vec![
            Ustr::from("enter_hall"),
            Ustr::from("light_torch"),
            Ustr::from("open_chest")
        ]
    );
}

// ---------------------------------------------------------------------------
// Mid-execution replanning: the world changes between turns.
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug)]
struct Temperature(pub i32);
#[derive(Component, Clone, Default, Debug)]
struct Sheltered(pub bool);
#[derive(Component, Clone, Default, Debug)]
struct Supplies(pub i32);

fn survive(task: &mut TaskBuilder) {
    task.branch()
        .precondition(|temp: &Temperature| temp.0 < 15)
        .precondition(|sheltered: &Sheltered| !sheltered.0)
        .then(seek_shelter)
        .then(warm_up);
    task.branch()
        .precondition(|temp: &Temperature| temp.0 < 15)
        .precondition(|sheltered: &Sheltered| sheltered.0)
        .then(warm_up);
    task.branch()
        .precondition(|temp: &Temperature| temp.0 >= 15)
        .then(scavenge);
    task.branch().then(wait);
}

fn seek_shelter(task: &mut TaskBuilder) {
    task.precondition(|sheltered: &Sheltered| !sheltered.0)
        .effect(|sheltered: &mut Sheltered| sheltered.0 = true);
}

fn warm_up(task: &mut TaskBuilder) {
    task.precondition(|sheltered: &Sheltered| sheltered.0)
        .effect(|temp: &mut Temperature| temp.0 += 20);
}

fn scavenge(task: &mut TaskBuilder) {
    task.precondition(|temp: &Temperature| temp.0 >= 15)
        .effect(|supplies: &mut Supplies| supplies.0 += 1);
}

fn wait(_task: &mut TaskBuilder) {}

fn survival_domain() -> HtnDomain {
    HtnDomain::from_root(survive)
        .build()
        .expect("survival domain is well-formed")
}

/// A survival scratchpad with the given temperature (shelterless, empty-handed).
fn survival_state(domain: &HtnDomain, temperature: i32) -> PlanState {
    PlanState::build(&domain.components)
        .set(Temperature(temperature))
        .finish()
}

/// Read component `T` out of a scratchpad (via the domain's slot registry).
fn read<'a, T: PlanComponent>(domain: &'a HtnDomain, state: &'a PlanState) -> &'a T {
    state.get_slot::<T>(domain.components.slot_of::<T>().unwrap())
}

/// Apply a single planned primitive's effects to `state` (simulating executing
/// that step of the plan).
fn apply_effects(bed: &HtnTestBed, task_name: &str, state: &mut PlanState) {
    let Some(Task::Primitive(p)) = bed.domain().get_task(task_name) else {
        panic!("task `{task_name}` not a primitive");
    };
    for e in p.effects.iter() {
        e.apply(state);
    }
}

#[test]
fn plan_adapts_when_world_changes_mid_execution() {
    let bed = HtnTestBed::new(survival_domain());

    // Turn 1: it's cold — plan to take shelter then warm up.
    let mut state = survival_state(bed.domain(), 5);
    let plan_1 = bed.plan_forward(&state);
    assert_eq!(
        plan_1,
        vec![Ustr::from("seek_shelter"), Ustr::from("warm_up")]
    );

    // Execute the shelter step (the world changes).
    apply_effects(&bed, plan_1[0], &mut state);

    // Turn 2: still cold (warm-up not yet done) — replan should still aim to
    // warm up, now shelter is secured.
    let plan_2 = bed.plan_forward(&state);
    assert_eq!(plan_2, vec![Ustr::from("warm_up")]);

    // Turn 3: warm up executes, temperature rises above the threshold.
    apply_effects(&bed, plan_2[0], &mut state);
    assert!(read::<Temperature>(bed.domain(), &state).0 >= 15);

    // Turn 4: requirement changed — now it's warm, so switch to scavenging.
    let plan_3 = bed.plan_forward(&state);
    assert_eq!(plan_3, vec![Ustr::from("scavenge")]);
}

#[test]
fn changing_requirements_force_a_new_root_branch() {
    let bed = HtnTestBed::new(survival_domain());
    // Start warm -> scavenge immediately.
    let state = survival_state(bed.domain(), 20);
    let plan_warm = bed.plan_forward(&state);
    assert_eq!(plan_warm, vec![Ustr::from("scavenge")]);

    // Requirements change (cold snap) mid-plan -> replan to shelter.
    let cold = survival_state(bed.domain(), 5);
    let plan_cold = bed.plan_forward(&cold);
    assert_eq!(
        plan_cold,
        vec![Ustr::from("seek_shelter"), Ustr::from("warm_up")]
    );
}
