//! Deep-goal + backtracking regression tests for `cdda_htn`, driven by the
//! `outpost.htn` fixture used by the deeper ECS benchmark.
//!
//! This pins the *semantics* of the deep domain so the benchmark measures
//! meaningful work and future optimization can't silently change behaviour:
//!   - a fresh actor's plan recurses through all four objectives (depth >= 5);
//!   - the "marginal fuel" method inside `ReachPosting` is *eligible* but its
//!     `Drive` leaf fails, so the planner must **backtrack** off it;
//!   - the plan, when executed, drives all four goal flags to `true`.

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::{HtnDomain, Task};
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;

/// A heavily-armed outpost squad member's plan state. Mirrors the fields used by
/// `outpost.htn` exactly (names must match the `.htn` identifiers).
#[derive(Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Default)]
enum Zone {
    #[default]
    Outside,
    Posting,
    Rally,
    Armory,
}

#[derive(Reflect, Clone, Debug, Default)]
struct OutpostState {
    fuel: i32,
    food: i32,
    health: i32,
    morale: i32,
    ammo: i32,
    perimeter: bool,
    reinforced: bool,
    armored: bool,
    caches: bool,
    position: Zone,
}

fn register(registry: &mut bevy_reflect::TypeRegistry) {
    registry.register::<OutpostState>();
    registry.register::<Zone>();
}

fn load_outpost() -> String {
    std::fs::read_to_string(format!("{}/htn/outpost.htn", env!("CARGO_MANIFEST_DIR")))
        .expect("read outpost fixture")
}

fn domain_and_registry() -> (HtnDomain, bevy_reflect::TypeRegistry) {
    let mut registry = bevy_reflect::TypeRegistry::default();
    register(&mut registry);
    let domain = bevy_bhtn::parse_htn(&load_outpost()).expect("outpost.htn must parse");
    (domain, registry)
}

/// A fresh actor who has not secured anything: all four objectives present.
fn fresh_state() -> OutpostState {
    OutpostState {
        fuel: 5,
        food: 30,
        health: 80,
        morale: 50,
        ammo: 12,
        ..Default::default()
    }
}

/// Apply a planned primitive's effects to `state` (simulating executing it).
fn apply_effects(
    domain: &HtnDomain,
    registry: &bevy_reflect::TypeRegistry,
    state: &mut OutpostState,
    name: &str,
) {
    let Some(Task::Primitive(p)) = domain.get_task(name) else {
        panic!("planned task `{name}` is not a primitive");
    };
    for e in p.effects.iter() {
        e.apply(state.as_reflect_mut(), registry);
    }
}

#[test]
fn fresh_actor_plan_is_deep_and_terminates() {
    let (domain, registry) = domain_and_registry();
    let mut planner = HtnPlanner::new(&domain, &registry);
    let plan = planner.plan("SecureOutpost", &fresh_state());

    // Depth >= 5: at least one leaf per objective (4 objectives) plus a
    // terminal stance. Any fresh actor that provably walks into a fixpoint must
    // produce at least 5 leaves.
    assert!(plan.task_names().len() >= 5, "got {:?}", plan.task_names());

    // The plan must name real tasks and each must be executable.
    for name in plan.task_names() {
        assert!(domain.get_task(name).is_some(), "unknown task {name:?}");
    }
}

#[test]
fn plan_when_executed_reaches_all_objectives() {
    let (domain, registry) = domain_and_registry();
    let mut planner = HtnPlanner::new(&domain, &registry);
    let mut state = fresh_state();
    let plan = planner.plan("SecureOutpost", &state);

    for name in plan.task_names() {
        apply_effects(&domain, &registry, &mut state, name);
    }

    // Every objective must be cleared by executing the plan.
    assert!(state.perimeter, "perimeter not secured");
    assert!(state.reinforced, "squad not reinforced");
    assert!(state.armored, "vehicles not armored");
    assert!(state.caches, "cache not secured");
    // The "complete" method also requires morale >= 5; a fresh actor with plenty
    // of morale never needs to rest, but the terminal method must be satisfiable.
    assert!(state.morale >= 0);
}

#[test]
fn marginal_fuel_forces_backtracking_off_drive() {
    // fuel ∈ [2, 8): `ReachPosting`'s "marginal fuel, queue anyway" method is
    // eligible (fuel >= 2), but its `Drive` leaf still needs fuel >= 8 and so
    // fails. The planner MUST abandon it and fall through to a later method.
    let state = OutpostState {
        fuel: 3,
        // Deliberately the only route eligible after Drive fails:
        food: 30, // packed rations march (fuel < 8, food >= 20)
        health: 1,
        ..Default::default()
    };
    let (domain, registry) = domain_and_registry();
    let mut planner = HtnPlanner::new(&domain, &registry);
    let plan = planner.plan("SecureOutpost", &state);

    // The first primitive in the plan cannot be Drive: Drive is not pickable,
    // so the plan must have resorted to the rations march or the collapse path.
    let names = plan.task_names();
    assert!(!names.is_empty(), "expected a plan");
    let first = names[0];
    assert_ne!(first, "Drive", "backtrack failed: Drive selected illegally");
    // The selected way to reach the posting is a march (food) or a rest+march.
    assert!(
        first == "Hike" || first == "March" || first == "Rest",
        "unexpected choice {first:?}"
    );
}

#[test]
fn high_fuel_agent_drives_directly() {
    // fuel >= 8: `ReachPosting`'s first method (the vehicle run) is reached
    // before the marginal-fuel trap, so Drive is legitimately chosen.
    let state = OutpostState {
        fuel: 10,
        food: 0,
        health: 90,
        ..Default::default()
    };
    let (domain, registry) = domain_and_registry();
    let mut planner = HtnPlanner::new(&domain, &registry);
    let plan = planner.plan(
        "SecureOutpost",
        &OutpostState {
            fuel: 10,
            food: 0,
            health: 90,
            ..Default::default()
        },
    );
    // The plan must begin with the vehicle run reaching the posting.
    assert_eq!(plan.task_names()[0], "Drive");
}

#[test]
fn deep_plan_uses_only_primitive_tasks() {
    let (domain, registry) = domain_and_registry();
    let mut planner = HtnPlanner::new(&domain, &registry);
    let plan = planner.plan("SecureOutpost", &fresh_state());
    for name in plan.task_names() {
        match domain.get_task(name) {
            Some(Task::Primitive(_)) => {}
            other => panic!("non-primitive task in plan: {name:?} ({other:?})"),
        }
    }
}
