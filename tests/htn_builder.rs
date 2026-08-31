//! Pins for the function-graph domain builder: task functions are recorded
//! once, baked into a flat indexed graph, and validated at `build()` time.
//! Covers recording semantics (declaration order, root selection, recursion,
//! names), build-time validation errors, and that baked domains feed
//! summaries/look-ahead exactly like the old parsed ones.

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{GoalBuilder, TaskBuilder};
use bevy_bhtn::{BackPlanner, HtnDomain, HtnError, Task};
use bevy_ecs::prelude::*;
use std::any::TypeId;

/// Capture a function item's `TypeId` (fn item types are unnameable in type
/// position, but generic inference recovers them from the value).
fn type_id_of<F: 'static>(_: F) -> TypeId {
    TypeId::of::<F>()
}

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Noise(bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Energy(i32);

/// Declaration order is preserved, the root is the function passed to
/// `from_root`, and task names come from the function names.
#[test]
fn builder_preserves_declaration_order_and_root() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(leaf);
        task.branch().then(second);
    }
    fn leaf(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    fn second(task: &mut TaskBuilder) {
        task.branch().then(leaf);
    }

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    assert_eq!(
        domain.tasks.iter().map(Task::name).collect::<Vec<_>>(),
        ["root", "leaf", "second"]
    );
    assert!(domain.root_task().is_compound());
    assert_eq!(domain.root_task().name(), "root");
    assert!(domain.get_task("leaf").is_some());
    assert!(domain.get_task("missing").is_none());
    // TypeId graph identity resolves too.
    assert_eq!(domain.task_index_by_type(type_id_of(root)), Some(0));
}

/// Subtask references may point at tasks recorded later, and recursive
/// `.then` references become plain graph edges (the function is recorded
/// exactly once).
#[test]
fn recursion_and_forward_references_bake_to_edges() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(late).then(tail);
    }
    fn late(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = true);
    }
    fn tail(task: &mut TaskBuilder) {
        task.precondition(|noise: &Noise| noise.0);
    }
    // A task that references itself: recorded once, edge to itself.
    fn spiral_root(task: &mut TaskBuilder) {
        task.branch().precondition(|gold: &Gold| gold.0 >= 2);
        task.branch().then(spiral_step).then(spiral_root);
    }
    fn spiral_step(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    // `tail` is a precondition-only primitive: pickable, executes as a no-op.
    assert_eq!(planner.plan("root", &state).task_names(), ["late", "tail"]);

    let spiral = HtnDomain::from_root(spiral_root)
        .build()
        .expect("well-formed");
    let summary = spiral.task_summary("spiral_root").expect("summary present");
    assert!(summary.recursive, "self-reference is a recursion edge");
    let state = PlanState::build(&spiral.components).finish();
    let mut planner = HtnPlanner::new(&spiral);
    assert_eq!(
        planner.plan("spiral_root", &state).task_names(),
        ["spiral_step", "spiral_step"],
        "two iterations reach the terminal branch"
    );
}

/// Duplicate task names (two same-named functions in different modules) are
/// rejected at `build` time.
#[test]
fn duplicate_task_names_yield_builder_error() {
    mod a {
        use super::*;
        pub fn dup(task: &mut TaskBuilder) {
            task.effect(|gold: &mut Gold| gold.0 = 1);
        }
    }
    mod b {
        use super::*;
        pub fn dup(task: &mut TaskBuilder) {
            task.effect(|noise: &mut Noise| noise.0 = true);
        }
    }
    fn root(task: &mut TaskBuilder) {
        task.branch().then(a::dup);
    }

    // b::dup is never referenced, so this builds fine...
    HtnDomain::from_root(root)
        .build()
        .expect("unreferenced duplicates are fine");

    fn clashing(task: &mut TaskBuilder) {
        task.branch().then(a::dup).then(b::dup);
    }
    let err = HtnDomain::from_root(clashing).build().unwrap_err();
    assert!(matches!(err, HtnError::Builder { .. }));
    assert!(err.to_string().contains("duplicate"));
}

/// A task mixing compound (`branch`) and primitive (`effect`) declarations is
/// rejected at `build` time.
#[test]
fn mixed_declarations_yield_builder_error() {
    fn confused(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
        task.branch().then(confused);
    }
    fn root(task: &mut TaskBuilder) {
        task.branch().then(confused);
    }
    let err = HtnDomain::from_root(root).build().unwrap_err();
    assert!(matches!(err, HtnError::Builder { .. }));
    assert!(err.to_string().contains("mixes"));
}

/// A root function that declares no branches cannot start a forward search.
#[test]
fn branchless_root_yields_builder_error() {
    fn leafy(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    let err = HtnDomain::from_root(leafy).build().unwrap_err();
    assert!(matches!(err, HtnError::Builder { .. }));
    assert!(err.to_string().contains("branches"));
}

/// Expected effects and actions round-trip onto the baked primitive.
#[test]
fn builder_records_expected_effects_and_actions() {
    fn mover(task: &mut TaskBuilder) {
        task.expected(|gold: &mut Gold| gold.0 = 99)
            .action(|cmds: &mut EntityCommands| {
                cmds.insert(Moved);
            });
    }
    fn root(task: &mut TaskBuilder) {
        task.branch().then(mover);
    }
    #[derive(Component, Default)]
    struct Moved;

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    let Task::Primitive(p) = domain.get_task("mover").unwrap() else {
        panic!("mover is primitive");
    };
    assert_eq!(p.effects.len(), 0);
    assert_eq!(p.expected_effects.len(), 1);
    assert!(p.action.is_some());
}

/// Baked domains get inferred summaries at build time, so the look-ahead
/// sweep runs exactly as it did for parsed domains: the doomed recursion is
/// refuted in one sweep and the safe method is chosen.
#[test]
fn baked_domains_get_summaries_and_lookahead() {
    fn act(task: &mut TaskBuilder) {
        task.branch().then(prime).then(spiral).then(impossible);
        task.branch().then(safe);
    }
    fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }
    fn prime(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = true);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(act).build().expect("well-formed");
    let summary = domain.task_summary("spiral").expect("summary present");
    assert_eq!(summary.min_yield, usize::MAX);
    assert!(!summary.terminating);

    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan("act", &state).task_names(), ["safe"]);
}

/// A baked domain plans end-to-end: compound decomposition, preconditions,
/// effects, and back-planning from a recorded goal all work.
#[test]
fn baked_domain_plans_forward_and_backward() {
    fn wealthy(task: &mut TaskBuilder) {
        task.branch().precondition(|gold: &Gold| gold.0 >= 3);
        task.branch().then(work).then(wealthy);
    }
    fn work(task: &mut TaskBuilder) {
        task.precondition(|noise: &Noise| !noise.0)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn three_gold(task: &mut GoalBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 3);
    }

    let domain = HtnDomain::from_root(wealthy)
        .goal(three_gold)
        .build()
        .expect("well-formed");

    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        planner.plan("wealthy", &state).task_names(),
        ["work", "work", "work"]
    );

    let mut back = BackPlanner::new(&domain);
    let back_plan = back.plan("three_gold", &state).expect("reachable");
    assert_eq!(back_plan.task_names(), ["work"]);
}

/// The full pipeline through the shared bed helper: plan, execute on the
/// scratchpad, and observe the effects.
#[test]
fn baked_domain_executes_on_the_scratchpad() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(recharge).then(root);
        task.branch().precondition(|energy: &Energy| energy.0 >= 3);
    }
    fn recharge(task: &mut TaskBuilder) {
        task.precondition(|energy: &Energy| energy.0 < 3)
            .effect(|energy: &mut Energy| energy.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    let mut state = PlanState::build(&domain.components).set(Energy(0)).finish();
    let mut planner = HtnPlanner::new(&domain);

    let plan = planner.plan("root", &state);
    assert_eq!(plan.task_names(), ["recharge", "recharge", "recharge"]);

    for name in plan.task_names() {
        let Task::Primitive(p) = domain.get_task(name).unwrap() else {
            panic!("plans are primitive sequences");
        };
        p.apply_effects(&mut state);
    }
    assert_eq!(
        state
            .get::<Energy>(domain.components.get::<Energy>().unwrap())
            .0,
        3
    );
}
