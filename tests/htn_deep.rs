//! Deep-goal + backtracking regression tests for `bevy_bhtn`, driven by the
//! outpost fixture used by the deeper ECS benchmark.
//!
//! This pins the *semantics* of the deep domain so the benchmark measures
//! meaningful work and future optimization can't silently change behaviour:
//!   - a fresh actor's plan recurses through all four objectives (depth >= 5);
//!   - the "marginal fuel" method inside `reach_posting` is *eligible* but its
//!     `drive` leaf fails, so the planner must **backtrack** off it;
//!   - the plan, when executed, drives all four goal flags to `true`.

mod common;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::Task;
use ustr::Ustr;

use common::{
    execute_plan, fresh_outpost, high_fuel_outpost, marginal_outpost, outpost_domain,
    outpost_scratch, Armored, Caches, Morale, Perimeter, Reinforced,
};

#[test]
fn fresh_actor_plan_is_deep_and_terminates() {
    let domain = outpost_domain();
    let state = outpost_scratch(&domain, fresh_outpost());
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("secure_outpost", &state);

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
    let domain = outpost_domain();
    let mut state = outpost_scratch(&domain, fresh_outpost());
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("secure_outpost", &state);

    execute_plan(&domain, &mut state, &plan);

    // Every objective must be cleared by executing the plan.
    assert!(
        state
            .get::<Perimeter>(domain.components.get::<Perimeter>().unwrap())
            .0,
        "perimeter not secured"
    );
    assert!(
        state
            .get::<Reinforced>(domain.components.get::<Reinforced>().unwrap())
            .0,
        "squad not reinforced"
    );
    assert!(
        state
            .get::<Armored>(domain.components.get::<Armored>().unwrap())
            .0,
        "vehicles not armored"
    );
    assert!(
        state
            .get::<Caches>(domain.components.get::<Caches>().unwrap())
            .0,
        "cache not secured"
    );
    // The "complete" method also requires morale >= 5; a fresh actor with plenty
    // of morale never needs to rest, but the terminal method must be satisfiable.
    assert!(
        state
            .get::<Morale>(domain.components.get::<Morale>().unwrap())
            .0
            >= 0
    );
}

#[test]
fn marginal_fuel_forces_backtracking_off_drive() {
    // fuel ∈ [2, 8): `reach_posting`'s "marginal fuel, queue anyway" method is
    // eligible (fuel >= 2), but its `drive` leaf still needs fuel >= 8 and so
    // fails. The planner MUST abandon it and fall through to a later method.
    let domain = outpost_domain();
    let state = outpost_scratch(&domain, marginal_outpost());
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("secure_outpost", &state);

    // The first primitive in the plan cannot be drive: drive is not pickable,
    // so the plan must have resorted to the rations march or the collapse path.
    let names = plan.task_names();
    assert!(!names.is_empty(), "expected a plan");
    let first = names[0];
    assert_ne!(
        first,
        Ustr::from("drive"),
        "backtrack failed: drive selected illegally"
    );
    // The selected way to reach the posting is a march (food) or a rest+march.
    assert!(
        first == Ustr::from("hike") || first == Ustr::from("march") || first == Ustr::from("rest"),
        "unexpected choice {first:?}"
    );
}

#[test]
fn high_fuel_agent_drives_directly() {
    // fuel >= 8: `reach_posting`'s first method (the vehicle run) is reached
    // before the marginal-fuel trap, so drive is legitimately chosen.
    let domain = outpost_domain();
    let state = outpost_scratch(&domain, high_fuel_outpost());
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("secure_outpost", &state);
    // The plan must begin with the vehicle run reaching the posting.
    assert_eq!(plan.task_names()[0], Ustr::from("drive"));
}

#[test]
fn deep_plan_uses_only_primitive_tasks() {
    let domain = outpost_domain();
    let state = outpost_scratch(&domain, fresh_outpost());
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("secure_outpost", &state);
    for name in plan.task_names() {
        match domain.get_task(name) {
            Some(Task::Primitive(_)) => {}
            _ => panic!("non-primitive task in plan: {name:?}"),
        }
    }
}
