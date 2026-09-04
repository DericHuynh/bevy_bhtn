//! Exhaustive pins for the MCTS search strategy (`src/mcts.rs`): plan
//! correctness, determinism, state immutability, policy independence, driver
//! integration, and bounded behavior on pathological domains.

use std::sync::Arc;

use bevy_bhtn::domain::SelectionPolicy;
use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
use bevy_bhtn::mcts::MctsSearcher;
use bevy_bhtn::planner::PlanStatus;
use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::selection::{HtnSearchStrategy, SearchOverride, Searcher};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{TaskBuilder, TaskFn};
use bevy_bhtn::{HtnDomain, Task};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(pub i32);

fn mine(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 += 1);
}
fn impossible(task: &mut TaskBuilder) {
    task.precondition(|g: &Gold| g.0 > 1000);
}
fn safe(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 = 5);
}
fn advance_counter(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 += 1);
}

/// Root with a doomed first branch and a working second one.
fn root_two_branches(task: &mut TaskBuilder) {
    task.branch().then(impossible).then(mine);
    task.branch().then(safe);
}

/// A chain of `N` sequential choice points: each link has a doomed method and
/// a working one that increments the counter and recurses — the only plan
/// threads every working method (20 primitives), then the terminal fires.
macro_rules! choice_chain {
    ($(($link:ident, $next:ident)),* $(,)?) => {
        $(
            fn $link(task: &mut TaskBuilder) {
                task.branch().then(impossible);
                task.branch().then(advance_counter).then($next);
            }
        )*
    };
}

fn chain_root(task: &mut TaskBuilder) {
    task.branch().then(link0).then(chain_root);
    task.branch().precondition(|g: &Gold| g.0 >= 20); // terminal: done
}

choice_chain!(
    (link0, link1),
    (link1, link2),
    (link2, link3),
    (link3, link4),
    (link4, link5),
    (link5, link6),
    (link6, link7),
    (link7, link8),
    (link8, link9),
    (link9, link10),
    (link10, link11),
    (link11, link12),
    (link12, link13),
    (link13, link14),
    (link14, link15),
    (link15, link16),
    (link16, link17),
    (link17, link18),
    (link18, link19),
    (link19, chain_tail),
);

fn chain_tail(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold| g.0 += 1);
}

/// Execute a plan's simulated effects onto a scratchpad (the bench idiom).
fn execute(domain: &HtnDomain, state: &PlanState, plan: &[u32]) -> PlanState {
    let mut out = state.clone();
    for &s in plan {
        if let Task::Primitive(p) = &domain.tasks[s as usize] {
            p.apply_effects(&mut out);
        }
    }
    out
}

/// `HtnPlanner::plan` with the root fn-item type inferred from the fn value
/// (fn-item types cannot be named directly in turbofish).
fn plan_root<F: TaskFn>(planner: &mut HtnPlanner, _root: F, state: &PlanState) -> Plan {
    planner.plan(_root, state).expect("plan")
}

// ---------------------------------------------------------------------------
// Plan correctness
// ---------------------------------------------------------------------------

/// MCTS explores past the first branch and returns the working one, complete,
/// under its own name.
#[test]
fn mcts_finds_the_working_branch() {
    let domain = HtnDomain::from_root(root_two_branches).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(500);
    let plan = mcts.search(&domain, &state).expect("a plan exists");
    assert_eq!(plan.status, PlanStatus::Complete);
    assert_eq!(plan.task_names(), ["safe"]);
}

/// The returned plan is a finished decomposition: executing its effects in
/// order reaches the goal value.
#[test]
fn mcts_plan_executes_to_goal() {
    let domain = HtnDomain::from_root(root_two_branches).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(500);
    let plan = mcts.search(&domain, &state).expect("a plan exists");
    let executed = execute(&domain, &state, &plan.steps);
    let gold = domain.components.get::<Gold>().unwrap();
    assert_eq!(executed.get::<Gold>(gold).0, 5);
}

/// A satisfied terminal plans to the empty (but complete) program.
#[test]
fn mcts_empty_plan_when_goal_already_met() {
    fn done(task: &mut TaskBuilder) {
        task.branch().precondition(|g: &Gold| g.0 >= 5);
        task.branch().then(safe).then(done);
    }
    let domain = HtnDomain::from_root(done).build().unwrap();
    let state = PlanState::build(&domain.components).set(Gold(7)).finish();

    let mcts = MctsSearcher::new(100);
    let plan = mcts
        .search(&domain, &state)
        .expect("terminal plans trivially");
    assert!(plan.is_empty());
    assert_eq!(plan.status, PlanStatus::Complete);
}

/// No applicable method anywhere: the search returns `None` (no fake prefix).
#[test]
fn mcts_returns_none_on_unplannable_domain() {
    fn dead(task: &mut TaskBuilder) {
        task.branch().then(impossible);
    }
    let domain = HtnDomain::from_root(dead).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(500);
    assert!(mcts.search(&domain, &state).is_none());
}

/// Twenty sequential choice points, each requiring the second method, plan to
/// a 20-step program that really reaches the target.
#[test]
fn mcts_solves_deep_choice_chains() {
    let domain = HtnDomain::from_root(chain_root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(2000);
    let plan = mcts.search(&domain, &state).expect("the chain plans");
    assert!(plan.is_complete());
    // One incrementing primitive per link (20) plus the tail's increment.
    assert_eq!(plan.steps.len(), 21);
    let executed = execute(&domain, &state, &plan.steps);
    let gold = domain.components.get::<Gold>().unwrap();
    assert_eq!(executed.get::<Gold>(gold).0, 21);
}

/// Nested choice points: only one combination of method choices wins, and
/// MCTS finds it by expansion.
#[test]
fn mcts_finds_the_winning_combination() {
    fn inner(task: &mut TaskBuilder) {
        task.branch().then(impossible);
        task.branch().then(advance_two);
    }
    fn advance_two(task: &mut TaskBuilder) {
        task.effect(|g: &mut Gold| g.0 = 5);
    }
    fn outer(task: &mut TaskBuilder) {
        task.branch().then(impossible);
        task.branch().then(inner);
    }
    let domain = HtnDomain::from_root(outer).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(1000);
    let plan = mcts
        .search(&domain, &state)
        .expect("outer→inner→advance wins");
    let names: Vec<String> = plan.task_names().iter().map(|u| u.to_string()).collect();
    assert_eq!(names, ["advance_two"]);
}

// ---------------------------------------------------------------------------
// Determinism & immutability
// ---------------------------------------------------------------------------

/// No randomness: the same state always yields the same plan — across calls
/// of one searcher and across separate searcher instances.
#[test]
fn mcts_is_deterministic() {
    let domain = HtnDomain::from_root(root_two_branches).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(300);
    let first = mcts.search(&domain, &state).expect("a plan exists");
    let second = mcts.search(&domain, &state).expect("a plan exists");
    assert_eq!(first.steps, second.steps, "same instance, same state");

    let mcts2 = MctsSearcher::new(300);
    let third = mcts2.search(&domain, &state).expect("a plan exists");
    assert_eq!(first.steps, third.steps, "separate instance, same state");
}

/// The input scratchpad is never mutated by a search.
#[test]
fn mcts_does_not_mutate_the_input_state() {
    let domain = HtnDomain::from_root(root_two_branches).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();

    let mcts = MctsSearcher::new(300);
    let _ = mcts.search(&domain, &state);

    assert_eq!(state.get::<Gold>(gold).0, 0, "input untouched");
}

// ---------------------------------------------------------------------------
// Budget & degenerate inputs
// ---------------------------------------------------------------------------

/// Zero iterations: no tree, no plan.
#[test]
fn mcts_zero_iterations_returns_none() {
    let domain = HtnDomain::from_root(root_two_branches).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    assert!(MctsSearcher::new(0).search(&domain, &state).is_none());
}

/// Budget edges: a single-method root completes via inlining at any
/// iteration count (even 0 work beyond the root advance); a doomed first
/// branch costs exactly one extra iteration in a two-branch domain.
#[test]
fn mcts_iteration_budget_edges() {
    fn one(task: &mut TaskBuilder) {
        task.branch().then(safe);
    }
    let one_domain = HtnDomain::from_root(one).build().unwrap();
    let state = PlanState::build(&one_domain.components).finish();

    // The root advance inlines the single method: one iteration suffices.
    let plan = MctsSearcher::new(1)
        .search(&one_domain, &state)
        .expect("trivial");
    assert_eq!(plan.task_names(), ["safe"]);

    // A doomed first branch costs exactly one extra iteration.
    let two_domain = HtnDomain::from_root(root_two_branches).build().unwrap();
    let two_state = PlanState::build(&two_domain.components).finish();
    let plan = MctsSearcher::new(2)
        .search(&two_domain, &two_state)
        .expect("2 iters");
    assert_eq!(plan.task_names(), ["safe"]);
    assert!(MctsSearcher::new(1)
        .search(&two_domain, &two_state)
        .is_none());
}

/// Rollouts are bounded: on a non-terminating domain every rollout hits the
/// cap (a loss) and the search terminates with `None` instead of hanging.
#[test]
fn mcts_is_bounded_on_non_terminating_domains() {
    fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }
    let domain = HtnDomain::from_root(spiral).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(200).with_rollout_depth(50);
    assert!(mcts.search(&domain, &state).is_none());
}

/// MCTS ignores baked selection policies: a `HighestUtility` policy scoring
/// the doomed branch above the working one does not stop MCTS from returning
/// the working plan (utility closures do not bias the tree or the rollout).
#[test]
fn mcts_ignores_baked_selection_policies() {
    fn root_utility(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::HighestUtility);
        task.branch()
            .utility_fn(|_: &Gold| 100.0)
            .then(impossible)
            .then(mine);
        task.branch().then(safe);
    }
    let domain = HtnDomain::from_root(root_utility).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mcts = MctsSearcher::new(500);
    let plan = mcts.search(&domain, &state).expect("a plan exists");
    assert_eq!(plan.task_names(), ["safe"]);
}

/// The default planner's sanity limit truncates the gate domain's doomed
/// enumeration to a partial prefix; MCTS (which never unwinds) still finds
/// the complete plan under the same default budget.
#[test]
fn mcts_beats_the_sanity_truncation_on_gate_domain() {
    let domain = common::gate_domain();
    let state = PlanState::build(&domain.components).finish();

    // DFS, look-ahead off, default budget: truncated prefix.
    let mut dfs = HtnPlanner::new(&domain);
    dfs.set_lookahead(false);
    let partial = plan_root(&mut dfs, common::gate_tasks::gate_root, &state);
    assert!(partial.is_partial());

    // MCTS, same default budget: the full winning plan.
    let mcts = MctsSearcher::new(5000);
    let plan = mcts
        .search(&domain, &state)
        .expect("strike→gate_final wins");
    assert!(plan.is_complete());
    let names: Vec<String> = plan.task_names().iter().map(|u| u.to_string()).collect();
    assert_eq!(names, ["strike", "gate_final"]);
}

// ---------------------------------------------------------------------------
// Driver integration
// ---------------------------------------------------------------------------

/// The driver dispatches MCTS via `HtnSearchStrategy::Custom`: the agent
/// plans and executes its episode to the goal.
#[test]
fn driver_runs_the_mcts_strategy() {
    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(HtnDomain::from_root(root_two_branches).build().unwrap())
            .with_strategy(HtnSearchStrategy::Custom(Arc::new(MctsSearcher::new(300)))),
    );

    let survivor = world.spawn((Gold(0), HtnAgent::default())).id();
    let mut schedule = Schedule::default();
    schedule.add_systems(htn_ai_system);

    let mut ticks = 0;
    while world.get::<Gold>(survivor).map(|g| g.0).unwrap_or(0) < 5 {
        ticks += 1;
        assert!(ticks <= 20, "the agent never finished");
        schedule.run(&mut world);
    }
    // The planless tick plans and executes the first (only) step.
    assert!(world
        .get::<HtnAgent>(survivor)
        .is_some_and(|h| h.plan.is_none()));
}

/// A per-agent `SearchOverride` selects MCTS while the global strategy stays
/// DepthFirst.
#[test]
fn per_agent_override_selects_mcts() {
    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(HtnDomain::from_root(root_two_branches).build().unwrap())
            .with_strategy(HtnSearchStrategy::DepthFirst),
    );

    let survivor = world
        .spawn((
            Gold(0),
            HtnAgent::default(),
            SearchOverride {
                strategy: Some(HtnSearchStrategy::Custom(Arc::new(MctsSearcher::new(300)))),
                sanity_limit: None,
            },
        ))
        .id();

    let mut schedule = Schedule::default();
    schedule.add_systems(htn_ai_system);

    let mut ticks = 0;
    while world.get::<Gold>(survivor).map(|g| g.0).unwrap_or(0) < 5 {
        ticks += 1;
        assert!(ticks <= 20, "the overridden agent never finished");
        schedule.run(&mut world);
    }
}

/// A population shares one `Arc<MctsSearcher>`: every agent plans correctly
/// against the shared, stateless searcher.
#[test]
fn mcts_is_shared_across_a_population() {
    let shared: Arc<dyn Searcher> = Arc::new(MctsSearcher::new(200));
    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(HtnDomain::from_root(root_two_branches).build().unwrap())
            .with_strategy(HtnSearchStrategy::Custom(Arc::clone(&shared))),
    );

    let agents: Vec<Entity> = (0..8)
        .map(|_| world.spawn((Gold(0), HtnAgent::default())).id())
        .collect();

    let mut schedule = Schedule::default();
    schedule.add_systems(htn_ai_system);

    let mut ticks = 0;
    while agents
        .iter()
        .any(|&a| world.get::<Gold>(a).map(|g| g.0).unwrap_or(0) < 5)
    {
        ticks += 1;
        assert!(ticks <= 40, "the population never finished");
        schedule.run(&mut world);
    }
    for &a in &agents {
        assert!(world.get::<HtnAgent>(a).is_some_and(|h| h.plan.is_none()));
    }
}

/// A planless result (None → empty plan) never wedges the agent: it replans
/// next tick.
#[test]
fn driver_replans_when_mcts_finds_nothing() {
    fn dead(task: &mut TaskBuilder) {
        task.branch().then(impossible);
    }
    let domain = HtnDomain::from_root(dead).build().unwrap();
    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(domain)
            .with_strategy(HtnSearchStrategy::Custom(Arc::new(MctsSearcher::new(100)))),
    );

    let survivor = world.spawn((Gold(0), HtnAgent::default())).id();
    let mut schedule = Schedule::default();
    schedule.add_systems(htn_ai_system);

    for _ in 0..5 {
        schedule.run(&mut world);
        let agent = world.get::<HtnAgent>(survivor).unwrap();
        assert!(
            agent.plan.is_none() || agent.plan.as_ref().unwrap().is_empty(),
            "a dead domain must not wedge the agent on a stale plan"
        );
    }
}

/// The searcher object is `Send + Sync` (shareable across threads).
#[test]
fn mcts_searcher_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MctsSearcher>();
}

#[path = "common/mod.rs"]
mod common;
