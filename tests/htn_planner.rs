//! Planner + AI logic tests using the shared [`HtnTestBed`].
//!
//! Exercises **forward planning** (MTR decomposition + backtracking), **backward
//! / goal-state planning**, and **plan validity** (execution: applying a plan's
//! task effects in order reaches the terminal state).
//!
//! Domains are deliberately *terminating* so assertions are exact and robust —
//! unbounded recursion (e.g. a miner that re-loops until a gold target) is the
//! reference `bevy_htn`'s shape and is covered by the parser conformance tests,
//! not by exact-plan assertions.

mod common;
use common::HtnTestBed;

use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_reflect::TypeRegistry;
use cdda_htn::{Effect, HtnCondition, Task};

// ---------------------------------------------------------------------------
// Travel domain — mirrors the classic bevy_htn `test_travel_htn`.
// Two root methods, exercises backtracking (walk when close, taxi when far).
// ---------------------------------------------------------------------------

#[derive(Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Default)]
enum Spot {
    #[default]
    Home,
    Other,
    Park,
}

#[derive(Reflect, Clone, Debug, Default)]
struct TravelState {
    cash: i32,
    distance_to_park: i32,
    happy: bool,
    my_location: Spot,
    taxi_location: Spot,
}

fn register_travel(registry: &mut TypeRegistry) {
    registry.register::<TravelState>();
    registry.register::<Spot>();
}

const TRAVEL_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "GoToPark" {
    method "walk" {
        subtasks: [Walk]
    }
    method "taxi" {
        subtasks: [CallTaxi, RideTaxi, PayTaxi]
    }
}

primitive_task "Walk" {
    operator: NoopOperator
    preconditions: [distance_to_park <= 4, my_location != Spot::Park, happy == false]
    effects: [
        my_location = Spot::Park,
        happy = true,
    ]
}

primitive_task "CallTaxi" {
    operator: NoopOperator
    preconditions: [cash >= 1]
    effects: [taxi_location = my_location]
}

primitive_task "RideTaxi" {
    operator: NoopOperator
    preconditions: [taxi_location == my_location, cash >= 1]
    effects: [
        taxi_location = Spot::Park,
        my_location = Spot::Park,
        happy = true,
    ]
}

primitive_task "PayTaxi" {
    operator: NoopOperator
    preconditions: [taxi_location == Spot::Park, cash >= 1]
    effects: [cash -= 1]
}
"#;

// ---------------------------------------------------------------------------
// Forward planning (deterministic, terminating)
// ---------------------------------------------------------------------------

#[test]
fn forward_plans_walk_when_close() {
    let bed = HtnTestBed::new(TRAVEL_HTN, "GoToPark", register_travel);
    let start = TravelState {
        distance_to_park: 1,
        ..Default::default()
    };
    assert_eq!(bed.plan_forward(&start), vec!["Walk"]);
}

#[test]
fn forward_plans_taxi_when_far() {
    let bed = HtnTestBed::new(TRAVEL_HTN, "GoToPark", register_travel);
    let start = TravelState {
        cash: 10,
        distance_to_park: 9,
        ..Default::default()
    };
    // Walk fails (too far) -> backtracks -> taxi succeeds.
    assert_eq!(
        bed.plan_forward(&start),
        vec!["CallTaxi", "RideTaxi", "PayTaxi"]
    );
}

#[test]
fn forward_plan_is_terminal_and_executes() {
    let bed = HtnTestBed::new(TRAVEL_HTN, "GoToPark", register_travel);
    let mut state = TravelState {
        cash: 10,
        distance_to_park: 9,
        ..Default::default()
    };
    let plan = bed.plan_forward(&state);
    assert_eq!(plan.len(), 3);

    // Execute: apply each planned task's effects in order.
    for name in &plan {
        let Some(Task::Primitive(p)) = bed.domain().get_task(name) else {
            panic!("planned task `{name}` missing");
        };
        for e in p.effects.iter() {
            e.apply(&mut state, bed.registry());
        }
    }
    // Terminal state: at the park, happy, taxi paid for.
    assert_eq!(state.my_location, Spot::Park);
    assert!(state.happy);
    assert_eq!(state.cash, 9);
}

// ---------------------------------------------------------------------------
// Forward planning: an already-satisfied goal yields an empty plan.
// ---------------------------------------------------------------------------

const IDEMPOTENT_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "EnsureOn" {
    method "already on" {
        preconditions: [powered == true]
        subtasks: []
    }
    method "switch on" {
        subtasks: [SwitchOn]
    }
}

primitive_task "SwitchOn" {
    operator: NoopOperator
    effects: [powered = true]
}
"#;

#[derive(Reflect, Clone, Debug, Default)]
struct PowerState {
    powered: bool,
}

fn register_power(registry: &mut TypeRegistry) {
    registry.register::<PowerState>();
}

#[test]
fn forward_plan_returns_empty_when_goal_already_met() {
    let bed = HtnTestBed::new(IDEMPOTENT_HTN, "EnsureOn", register_power);
    let on = PowerState { powered: true };
    assert_eq!(bed.plan_forward(&on), Vec::<String>::new());
    let off = PowerState { powered: false };
    assert_eq!(bed.plan_forward(&off), vec!["SwitchOn"]);
}

// ---------------------------------------------------------------------------
// Backward (goal-state) planning
// ---------------------------------------------------------------------------

const GOAL_HTN: &str = r#"
schema {
    version: 0.1.0
}

primitive_task "Mine" {
    operator: NoopOperator
    preconditions: [has_ore == false]
    effects: [has_ore = true]
}

goal_task "HaveOre" {
    effects: [has_ore = true]
}
"#;

#[derive(Reflect, Clone, Debug, Default)]
struct OreState {
    has_ore: bool,
}

fn register_ore(registry: &mut TypeRegistry) {
    registry.register::<OreState>();
}

#[test]
fn backward_plan_finds_satisfying_leaf() {
    let bed = HtnTestBed::new(GOAL_HTN, "Mine", register_ore);
    let plan = bed
        .plan_backward("HaveOre", &OreState::default())
        .expect("back plan reaches goal");
    assert_eq!(plan, vec!["Mine"]);
}

#[test]
fn backward_plan_rejects_unreachable_goal() {
    let bed = HtnTestBed::new(
        r#"
        schema {
            version: 0.1.0
        }

        primitive_task "Mine" {
            operator: NoopOperator
            effects: [has_ore = true]
        }

        goal_task "WantMore" {
            effects: [has_ore = true, has_rope = true]
        }
        "#,
        "Mine",
        register_ore,
    );
    // `has_rope` is never produced by any primitive -> goal unreachable.
    assert!(bed.plan_backward("WantMore", &OreState::default()).is_err());
}

// ---------------------------------------------------------------------------
// DSL parsing details + condition/effect evaluation
// ---------------------------------------------------------------------------

#[test]
fn dsl_parses_conditions_and_effects() {
    let bed = HtnTestBed::new(TRAVEL_HTN, "GoToPark", register_travel);
    let Some(Task::Primitive(walk)) = bed.domain().get_task("Walk") else {
        panic!("Walk primitive missing");
    };
    assert!(walk.preconditions.iter().any(|c| matches!(
        c,
        HtnCondition::LessThanInt {
            field,
            orequals: true,
            ..
        } if field == "distance_to_park"
    )));
    assert!(walk
        .effects
        .iter()
        .any(|e| matches!(e, Effect::SetEnum { enum_variant: v, .. } if v == "Park")));
    assert!(walk
        .effects
        .iter()
        .any(|e| matches!(e, Effect::SetBool { field: f, value: true, .. } if f == "happy")));
}

#[test]
fn conditions_evaluate_against_state() {
    let state = TravelState {
        cash: 10,
        distance_to_park: 3,
        ..Default::default()
    };
    let condensed = HtnCondition::LessThanInt {
        field: "distance_to_park".into(),
        threshold: 4,
        orequals: true,
    };
    assert!(condensed.evaluate(state.as_reflect()));
}

#[test]
fn effects_apply_to_state() {
    let bed = HtnTestBed::new(TRAVEL_HTN, "GoToPark", register_travel);
    let mut state = TravelState::default();
    Effect::SetBool {
        field: "happy".into(),
        value: true,
    }
    .apply(&mut state, bed.registry());
    assert!(state.happy);
}
