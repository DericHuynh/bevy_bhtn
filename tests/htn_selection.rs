//! Pins for the Phase 1 API: branch names, selection policies
//! (FirstMatch / HighestUtility / WeightedRandom / Custom), cost signals,
//! fail-fast strategy, per-agent `SearchOverride`, the `Searcher` trait, and
//! the decomposition-trace contract.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bevy_bhtn::domain::{BranchCandidate, BranchRanker, SelectionPolicy};
use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::selection::{
    DecompositionTrace, HtnSearchStrategy, SearchOverride, Searcher, TraceOutcome,
};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{GoalBuilder, TaskBuilder, TaskFn};
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;

/// A task fn's item type cannot be named directly, so the lookup-by-type API
/// is reached through these inference helpers: the fn value pins `F` to the
/// fn item's unique type, resolved through the baked `TypeId` index.
fn plan_of<F: TaskFn>(planner: &mut HtnPlanner<'_>, _f: F, state: &PlanState) -> Plan {
    planner.plan(_f, state)
}

fn task_index_of<F: TaskFn>(domain: &HtnDomain, _f: F) -> Option<usize> {
    domain.task_index(_f)
}

fn plan_traced_of<F: TaskFn>(
    planner: &mut HtnPlanner<'_>,
    _f: F,
    state: &PlanState,
    trace: &mut Vec<DecompositionTrace>,
) -> Plan {
    planner.plan_traced(_f, state, trace)
}

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Distance(i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Battery(i32);

// ---------------------------------------------------------------------------
// Named branches
// ---------------------------------------------------------------------------

#[test]
fn named_branches_are_recorded() {
    fn root(task: &mut TaskBuilder) {
        task.branch()
            .named("snipe")
            .precondition(|gold: &Gold| gold.0 > 10)
            .then(win);
        task.branch().named("melee").then(win);
    }
    fn win(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let methods = match domain.get_task("root").unwrap() {
        bevy_bhtn::Task::Compound(c) => &c.methods,
        _ => panic!("root is compound"),
    };
    assert_eq!(methods[0].name, Some("snipe"));
    assert_eq!(methods[1].name, Some("melee"));
}

// ---------------------------------------------------------------------------
// HighestUtility
// ---------------------------------------------------------------------------

fn utility_root(task: &mut TaskBuilder) {
    task.select(SelectionPolicy::HighestUtility);
    task.branch()
        .named("big")
        .precondition(|gold: &Gold| gold.0 >= 5)
        .utility(100.0)
        .then(win);
    task.branch().named("small").utility(40.0).then(win);
}
fn win(task: &mut TaskBuilder) {
    task.effect(|gold: &mut Gold| gold.0 += 1);
}
fn utility_domain() -> HtnDomain {
    HtnDomain::from_root(utility_root).build().unwrap()
}

/// The highest-utility *valid* branch wins; ties and invalid branches behave
/// per the documented contract.
#[test]
fn highest_utility_selects_the_best_valid_branch() {
    let domain = utility_domain();
    let mut planner = HtnPlanner::new(&domain);

    // Both branches valid: "big" (100) beats "small" (40).
    let rich = PlanState::build(&domain.components).set(Gold(10)).finish();
    assert_eq!(
        plan_of(&mut planner, utility_root, &rich).task_names(),
        ["win"]
    );
    let mtr = plan_of(&mut planner, utility_root, &rich);
    assert_eq!(mtr.mtr().0, [0], "branch 0 (big) selected");

    // "big" invalid (gold < 5): "small" is the only valid branch.
    let poor = PlanState::build(&domain.components).set(Gold(0)).finish();
    let plan = plan_of(&mut planner, utility_root, &poor);
    assert_eq!(plan.task_names(), ["win"]);
    assert_eq!(plan.mtr().0, [1], "branch 1 (small) selected");
}

/// The default FirstMatch policy ignores utility entirely: declaration order
/// wins even when a later branch scores higher.
#[test]
fn first_match_ignores_utility() {
    fn root(task: &mut TaskBuilder) {
        task.branch().named("low").utility(1.0).then(win);
        task.branch().named("high").utility(100.0).then(win);
    }
    fn win(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).mtr().0,
        [0],
        "declaration order"
    );
}

/// Dynamic utility (`utility_fn`) is evaluated against the scratchpad at
/// branch-evaluation time.
#[test]
fn utility_fn_scores_from_components() {
    fn root(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::HighestUtility);
        task.branch()
            .named("close")
            .utility_fn(|d: &Distance| (100 - d.0).max(0) as f32)
            .then(win);
        task.branch().named("far").utility(10.0).then(win);
    }
    fn win(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);

    // Distance 2 → "close" scores 98 > 10.
    let near = PlanState::build(&domain.components)
        .set(Distance(2))
        .finish();
    assert_eq!(plan_of(&mut planner, root, &near).mtr().0, [0]);

    // Distance 200 → "close" scores 0 < 10.
    let far = PlanState::build(&domain.components)
        .set(Distance(200))
        .finish();
    assert_eq!(plan_of(&mut planner, root, &far).mtr().0, [1]);
}

// ---------------------------------------------------------------------------
// WeightedRandom
// ---------------------------------------------------------------------------

fn weighted_root(task: &mut TaskBuilder) {
    task.select(SelectionPolicy::WeightedRandom { seed: 42 });
    task.branch().named("likely").utility(100.0).then(win);
    task.branch().named("unlikely").utility(0.001).then(win);
}

fn weighted_domain(seed: u64) -> HtnDomain {
    let _ = seed;
    HtnDomain::from_root(weighted_root).build().unwrap()
}

/// The weighted sampler is deterministic for a given seed and state: two
/// planner runs produce identical MTRs. And a near-zero-weight branch is
/// essentially never taken when a heavy branch is valid.
#[test]
fn weighted_random_is_deterministic_and_weight_respecting() {
    let domain = weighted_domain(42);
    let state = PlanState::build(&domain.components).finish();

    let mut a = HtnPlanner::new(&domain);
    let mut b = HtnPlanner::new(&domain);
    let plan_a = plan_of(&mut a, weighted_root, &state);
    let plan_b = plan_of(&mut b, weighted_root, &state);
    assert_eq!(plan_a.mtr(), plan_b.mtr(), "same seed → same order");

    // Weight 100 vs 0.001: the heavy branch wins with overwhelming
    // probability — pin it (the sampler is deterministic, so this is exact).
    assert_eq!(plan_a.mtr().0, [0]);
}

/// Backtracking exhausts the *sampled* order instead of re-sampling: a
/// high-utility branch that fails downstream must not starve the fallback.
#[test]
fn weighted_random_exhausts_sampled_order_on_backtrack() {
    fn root(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::WeightedRandom { seed: 7 });
        // Sampled first (high utility) but doomed downstream.
        task.branch()
            .named("doomed")
            .utility(100.0)
            .then(prime)
            .then(impossible);
        // Fallback.
        task.branch().named("safe").utility(1.0).then(safe);
    }
    fn prime(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(),
        ["safe"],
        "the fallback is reached after the sampled branch fails"
    );
}

// ---------------------------------------------------------------------------
// Custom ranker
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CountingRanker {
    calls: AtomicUsize,
}

impl BranchRanker for CountingRanker {
    fn rank(&self, candidates: &[BranchCandidate<'_>], _state: &PlanState, out: &mut Vec<u32>) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Reverse declaration order — the opposite of FirstMatch.
        for c in candidates.iter().rev() {
            out.push(c.index);
        }
    }
}

#[test]
fn custom_ranker_orders_branches_and_is_sanitized() {
    fn root(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::Custom(Arc::new(CountingRanker::default())));
        task.branch().named("first").then(win);
        task.branch().named("second").then(win);
    }
    fn win(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    // The ranker reverses declaration order → branch 1 is tried first.
    assert_eq!(plan_of(&mut planner, root, &state).mtr().0, [1]);
}

/// A misbehaving ranker (dropping candidates) cannot make branches
/// unreachable: omitted valid branches are appended in declaration order.
#[test]
fn ranker_omissions_fall_back_to_declaration_order() {
    struct DroppingRanker;

    impl BranchRanker for DroppingRanker {
        fn rank(&self, candidates: &[BranchCandidate<'_>], _state: &PlanState, out: &mut Vec<u32>) {
            // Only ever offers the first candidate.
            if let Some(c) = candidates.first() {
                out.push(c.index);
            }
        }
    }

    fn root(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::Custom(Arc::new(DroppingRanker)));
        task.branch().then(doomed).then(impossible);
        task.branch().then(safe);
    }
    fn doomed(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    // The ranker only offered branch 0; the sanitizer appends branch 1, so
    // the search still reaches the safe plan.
    assert_eq!(plan_of(&mut planner, root, &state).task_names(), ["safe"]);
}

// ---------------------------------------------------------------------------
// Cost signals (recorded; inert under DepthFirst)
// ---------------------------------------------------------------------------

#[test]
fn cost_is_recorded_on_primitives() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(cheap);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(0.5)
            .cost_fn(|state: &PlanState| state.get::<Gold>(0).0.min(0).abs() as f32)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let bevy_bhtn::Task::Primitive(p) = domain.get_task("cheap").unwrap() else {
        panic!("cheap is primitive");
    };
    assert!(p.cost.is_some());

    // Inert under DepthFirst: the plan is unaffected.
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(plan_of(&mut planner, root, &state).task_names(), ["cheap"]);
}

// ---------------------------------------------------------------------------
// Strategies: fail-fast + per-agent override + Searcher trait
// ---------------------------------------------------------------------------

fn fail_fast_root(task: &mut TaskBuilder) {
    // Branch 1 passes ranking but fails *downstream* (after `prime`
    // applies, `impossible`'s precondition is false) — the case fail-fast
    // exists for.
    task.branch().then(prime).then(impossible);
    task.branch().then(works);
}
fn prime(task: &mut TaskBuilder) {
    task.effect(|gold: &mut Gold| gold.0 += 1);
}
fn impossible(task: &mut TaskBuilder) {
    task.precondition(|gold: &Gold| gold.0 > 100);
}
fn works(task: &mut TaskBuilder) {
    task.effect(|gold: &mut Gold| gold.0 = 1);
}
fn fail_fast_domain() -> HtnDomain {
    HtnDomain::from_root(fail_fast_root).build().unwrap()
}

/// Full backtracking recovers from the first branch's downstream failure...
#[test]
fn depth_first_backtracks_to_the_second_branch() {
    let domain = fail_fast_domain();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, fail_fast_root, &state).task_names(),
        ["works"]
    );
}

/// ...while fail-fast returns the partial plan immediately (the already-
/// decomposed `prime` step, without backtracking to try `works`).
#[test]
fn fail_fast_returns_partial_plan_on_first_failure() {
    let domain = fail_fast_domain();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_fail_fast(true);
    assert_eq!(
        plan_of(&mut planner, fail_fast_root, &state).task_names(),
        ["prime"],
        "fail-fast keeps the partial plan instead of backtracking"
    );
}

/// Regression: the MTR of a plan found *after* a backtrack records only the
/// chosen method per decomposition level. The frame's `mtr_len` snapshot must
/// precede its own method push, so the retry replaces the failed entry
/// instead of appending after it (which yielded `[0, 1]` for a plan that
/// only chose branch 1).
#[test]
fn mtr_after_backtrack_records_only_the_chosen_method() {
    let domain = fail_fast_domain();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_of(&mut planner, fail_fast_root, &state);
    assert_eq!(plan.task_names(), ["works"]);
    assert_eq!(
        plan.mtr().0,
        [1],
        "branch 0 was backtracked past; only the chosen branch 1 remains"
    );
}

/// Regression (nested case): a backtrack *below* the root must also replace
/// the failed method's entry at its own level — the root's choice stays at
/// its level and the retried compound's new choice replaces the failed one,
/// giving `[0, 1]`, not the stale `[0, 0, 1]`.
#[test]
fn mtr_after_nested_backtrack_records_only_chosen_methods() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(mid);
    }
    fn mid(task: &mut TaskBuilder) {
        // Branch 0 commits, then fails downstream (after `prime` applies,
        // `impossible`'s precondition is false).
        task.branch().then(prime).then(impossible);
        task.branch().then(works);
    }
    fn prime(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn works(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(plan.task_names(), ["works"]);
    assert_eq!(
        plan.mtr().0,
        [0, 1],
        "root chose branch 0; mid's failed branch 0 was replaced by branch 1"
    );
}

#[derive(Default)]
struct FixedSearcher;

impl Searcher for FixedSearcher {
    fn search(&self, domain: &HtnDomain, _state: &PlanState) -> Option<Plan> {
        // A "strategy" that always picks the second branch's single step.
        let idx = task_index_of(domain, works)?;
        Some(Plan {
            steps: vec![idx as u32],
            names: vec!["works".into()],
            mtr: bevy_bhtn::planner::Mtr(vec![1]),
            status: bevy_bhtn::planner::PlanStatus::Complete,
        })
    }
}

/// A `Custom` strategy replaces the planner entirely, globally and per-agent.
#[test]
fn custom_strategy_and_per_agent_override() {
    let mut world = World::new();
    // Global strategy: Custom.
    world.insert_resource(
        HtnConfig::new(fail_fast_domain())
            .with_strategy(HtnSearchStrategy::Custom(Arc::new(FixedSearcher))),
    );
    let global = world.spawn((Gold(0), HtnAgent::default())).id();
    // Per-agent override: a zero sanity budget (differs from the global
    // Custom strategy) — the agent plans nothing and stays idle.
    let overridden = world
        .spawn((
            Gold(0),
            HtnAgent::default(),
            SearchOverride {
                strategy: Some(HtnSearchStrategy::DepthFirst),
                sanity_limit: Some(0),
            },
        ))
        .id();

    htn_ai_system(&mut world);

    // The global-Custom agent ran FixedSearcher → the 1-step [works] plan
    // completed within the tick (plan already dropped), observable through
    // the committed effect.
    assert_eq!(world.get::<Gold>(global).unwrap().0, 1);
    assert!(world.get::<HtnAgent>(global).unwrap().plan.is_none());

    // The overridden agent's zero sanity budget yields an empty (planless)
    // plan: it stays idle while the global agent works — the override
    // demonstrably took effect.
    let agent = world.get::<HtnAgent>(overridden).unwrap();
    assert!(agent.plan.is_none());
    assert_eq!(agent.cursor, 0);
    assert_eq!(world.get::<Gold>(overridden).unwrap().0, 0);
}

// ---------------------------------------------------------------------------
// Decomposition trace
// ---------------------------------------------------------------------------

/// `plan_traced` reports Selected / PrecondFailed / Backtracked per commitment.
#[test]
fn plan_traced_reports_selection_decisions() {
    fn root(task: &mut TaskBuilder) {
        task.branch().named("doomed").then(prime).then(impossible);
        task.branch().named("safe").then(safe);
    }
    fn prime(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let mut trace = Vec::new();
    let plan = plan_traced_of(&mut planner, root, &state, &mut trace);

    assert_eq!(plan.task_names(), ["safe"]);
    assert!(
        trace
            .iter()
            .any(|t| t.branch_name == Some("doomed") && t.outcome == TraceOutcome::Selected),
        "the doomed branch was selected first"
    );
    assert!(
        trace
            .iter()
            .any(|t| t.branch_name == Some("safe") && t.outcome == TraceOutcome::Selected),
        "the safe branch was selected after backtracking"
    );
    assert!(
        trace.iter().any(|t| t.outcome == TraceOutcome::Backtracked),
        "the backtrack past the doomed branch is recorded"
    );
}

/// The driver bridges traces into `Messages<DecompositionTrace>` when
/// `debug_trace` is enabled.
#[test]
fn driver_forwards_traces_when_debug_trace_is_enabled() {
    fn root(task: &mut TaskBuilder) {
        task.branch().named("charge_cycle").then(gather);
    }
    fn gather(task: &mut TaskBuilder) {
        task.effect(|battery: &mut Battery| battery.0 += 1);
    }
    fn three_gold(task: &mut GoalBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 3);
    }
    let _ = three_gold;

    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(HtnDomain::from_root(root).build().unwrap()).with_debug_trace(true),
    );
    let entity = world.spawn((Battery(0), HtnAgent::default())).id();

    htn_ai_system(&mut world);

    let messages = world.resource::<bevy_ecs::message::Messages<DecompositionTrace>>();
    let mut reader = messages.get_cursor();
    let seen: Vec<_> = reader.read(messages).collect();
    assert!(seen
        .iter()
        .any(|t| t.branch_name == Some("charge_cycle") && t.outcome == TraceOutcome::Selected));
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);
}
