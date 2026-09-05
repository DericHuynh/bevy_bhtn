//! Pins for the function-graph domain builder: task functions are recorded
//! once, baked into a flat indexed graph, and validated at `build()` time.
//! Covers recording semantics (declaration order, root selection, recursion,
//! names), build-time validation errors, and that baked domains feed
//! summaries/look-ahead exactly like the old parsed ones.

use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{GoalBuilder, GoalFn, TaskBuilder, TaskFn};
use bevy_bhtn::{BackPlanner, HtnDomain, HtnError, HtnResult, Task, TaskSummary};
use bevy_ecs::prelude::*;
use std::any::TypeId;

/// A task fn's item type cannot be named directly, so the lookup-by-type API
/// is reached through these inference helpers: the fn value pins `F` to the
/// fn item's unique type, resolved through the baked `TypeId` index.
fn plan_of<F: TaskFn>(planner: &mut HtnPlanner<'_>, _f: F, state: &PlanState) -> Plan {
    planner.plan(_f, state).expect("plan")
}

fn summary_of<F: TaskFn>(domain: &HtnDomain, _f: F) -> Option<&TaskSummary> {
    domain.task_summary(_f)
}

fn back_plan_of<F: GoalFn>(
    planner: &mut BackPlanner<'_>,
    _f: F,
    state: &PlanState,
) -> HtnResult<Plan> {
    planner.plan(_f, state)
}

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
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(&domain),
        ["late", "tail"]
    );

    let spiral = HtnDomain::from_root(spiral_root)
        .build()
        .expect("well-formed");
    let summary = summary_of(&spiral, spiral_root).expect("summary present");
    assert!(summary.recursive, "self-reference is a recursion edge");
    let state = PlanState::build(&spiral.components).finish();
    let mut planner = HtnPlanner::new(&spiral);
    assert_eq!(
        plan_of(&mut planner, spiral_root, &state).task_names(&spiral),
        ["spiral_step", "spiral_step"],
        "two iterations reach the terminal branch"
    );
}

/// Two distinct task functions with the same display name (last path segment
/// of `type_name`) are separate identities: both bake, resolve through their
/// own `TypeId`s, and both execute. Display names are not identity — the
/// old name-keyed bake check spuriously rejected this shape (regression pin
/// for the lookup-by-type migration; the same collision broke closures).
#[test]
fn same_display_names_are_distinct_identities() {
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
        task.branch().then(a::dup).then(b::dup);
    }

    let domain = HtnDomain::from_root(root)
        .build()
        .expect("same-named task fns are distinct identities and bake fine");
    // Distinct `TypeId`s resolve to distinct task indices.
    let a_idx = domain.task_index_by_type(type_id_of(a::dup)).unwrap();
    let b_idx = domain.task_index_by_type(type_id_of(b::dup)).unwrap();
    assert_ne!(a_idx, b_idx, "the two `dup` fns must not alias");
    // Display-name introspection is first-wins (and finds *something*).
    assert!(domain.get_task("dup").is_some());
    // Both tasks exist and execute in declaration order. Steps are addressed
    // by index (plan.names is display-only and both steps share "dup").
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(plan.task_names(&domain), ["dup", "dup"]);
    let mut executed = state.clone();
    for &step in plan.steps() {
        let Task::Primitive(p) = &domain.tasks[step as usize] else {
            panic!("plans are primitive sequences");
        };
        p.apply_effects(&mut executed);
    }
    let gold = domain.components.slot_of::<Gold>().unwrap();
    let noise = domain.components.slot_of::<Noise>().unwrap();
    assert_eq!(executed.get_slot::<Gold>(gold).0, 1, "a::dup ran");
    assert!(executed.get_slot::<Noise>(noise).0, "b::dup ran");
}

/// Registering the same goal function twice is a bake error — the second
/// registration would silently shadow the first in the `TypeId` index
/// (regression pin for the goal `TypeId` index added with lookup-by-type).
#[test]
fn duplicate_goal_fn_registration_is_a_bake_error() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(leaf);
    }
    fn leaf(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    fn want_gold(goal: &mut GoalBuilder) {
        goal.effect(|gold: &mut Gold| gold.0 = 3);
    }

    let err = HtnDomain::from_root(root)
        .goal(want_gold)
        .goal(want_gold)
        .build()
        .unwrap_err();
    assert!(matches!(err, HtnError::Builder { .. }));
    assert!(err.to_string().contains("duplicate goal function"));
}

/// Planning from a task function that was never recorded in the domain yields
/// An unregistered root function is a legible error (never a panic, never a
/// silent empty plan) — the type-addressed analogue of the old unknown-name
/// behavior, now keyed by the fn item's `TypeId`.
#[test]
fn unregistered_root_yields_an_error() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(leaf);
    }
    fn leaf(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    fn never_registered(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 99);
    }

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    assert_eq!(domain.task_index(never_registered), None);
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let err = planner.plan(never_registered, &state).unwrap_err();
    assert!(
        matches!(err, HtnError::UnregisteredTask { .. }),
        "error names the offending fn: {err}"
    );
    assert!(err.to_string().contains("htn_builder"));
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
    let summary = summary_of(&domain, spiral).expect("summary present");
    assert_eq!(summary.min_yield, usize::MAX);
    assert!(!summary.terminating);

    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(plan_of(&mut planner, act, &state).task_names(&domain), ["safe"]);
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
        plan_of(&mut planner, wealthy, &state).task_names(&domain),
        ["work", "work", "work"]
    );

    let mut back = BackPlanner::new(&domain);
    let back_plan = back_plan_of(&mut back, three_gold, &state).expect("reachable");
    // Compound participation: the greedy chains through `wealthy`'s recursive
    // method until its own `gold >= 3` gate closes — the plan really reaches
    // the goal's value (the old primitive-only greedy stopped after one `work`,
    // leaving gold at 1).
    assert_eq!(back_plan.task_names(&domain), ["work", "work", "work"]);
    let mut executed = state.clone();
    for &s in back_plan.steps() {
        if let Task::Primitive(p) = &domain.tasks[s as usize] {
            p.apply_effects(&mut executed);
        }
    }
    let gold = domain.components.slot_of::<Gold>().unwrap();
    assert_eq!(executed.get_slot::<Gold>(gold).0, 3);
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

    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(plan.task_names(&domain), ["recharge", "recharge", "recharge"]);

    for name in plan.task_names(&domain) {
        let Task::Primitive(p) = domain.get_task(name).unwrap() else {
            panic!("plans are primitive sequences");
        };
        p.apply_effects(&mut state);
    }
    assert_eq!(
        state
            .get_slot::<Energy>(domain.components.slot_of::<Energy>().unwrap())
            .0,
        3
    );
}
