//! Pins for Phase 2: the `CostBounded` branch-and-bound strategy and the
//! `min_cost` summary it prunes with.
//!
//! The load-bearing guarantees pinned here:
//!
//! - **Choice**: CostBounded returns the cheapest *complete* plan, where
//!   DepthFirst returns the first one found — even when the cheap plan is
//!   declared last.
//! - **Pruning**: once a best plan exists, a commitment whose bake-time
//!   `min_cost` lower bound cannot strictly beat it is skipped *without
//!   recursing* — observable as plans found within budgets that exploring
//!   the pruned subtree would blow.
//! - **Anytime**: when the budget runs out mid-search, the best complete
//!   plan found so far is returned (never a half-expensive partial).
//! - **Soundness under rollback**: state is correctly restored between
//!   complete plans (a broken rollback makes later branches see phantom
//!   effects and either spuriously fail or spuriously succeed).
//! - **Dynamic costs**: `cost_fn` is evaluated against the scratchpad at
//!   plan time and drives the choice; statically it lower-bounds to 0.
//! - **Inert without annotations**: primitives without `cost`/`cost_fn`
//!   count 0, so CostBounded degenerates to exactly DepthFirst's choice.

use bevy_bhtn::domain::SelectionPolicy;
use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::selection::{HtnSearchStrategy, SearchOverride};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{TaskBuilder, TaskFn};
use bevy_bhtn::{HtnDomain, TaskSummary};
use bevy_ecs::prelude::*;

/// A task fn's item type cannot be named directly, so the lookup-by-type API
/// is reached through these inference helpers: the fn value pins `F` to the
/// fn item's unique type, resolved through the baked `TypeId` index.
fn plan_of<F: TaskFn>(planner: &mut HtnPlanner<'_>, _f: F, state: &PlanState) -> Plan {
    planner.plan(_f, state).expect("plan")
}

fn summary_of<F: TaskFn>(domain: &HtnDomain, _f: F) -> Option<&TaskSummary> {
    domain.task_summary(_f)
}

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(i32);
/// Mode flag read by a dynamic `cost_fn` (the only component in its domain,
/// so it occupies slot 0).
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct CheapMode(pub bool);

// ---------------------------------------------------------------------------
// min_cost summary
// ---------------------------------------------------------------------------

/// The `min_cost` summary mirrors `min_yield`: primitives contribute their
/// declared constant, compounds take the cheapest method's subtask sum,
/// dynamic `cost_fn` costs conservatively bound at 0, and non-terminating
/// tasks are infinite.
#[test]
fn min_cost_summary_is_inferred() {
    fn root(task: &mut TaskBuilder) {
        // 2.0 + 3.0 = 5.0 through this method.
        task.branch().then(priced_a).then(priced_b);
        // Dynamic cost: bake-time lower bound 0.
        task.branch().then(priced_fn);
    }
    fn priced_a(task: &mut TaskBuilder) {
        task.cost(2.0);
    }
    fn priced_b(task: &mut TaskBuilder) {
        task.cost(3.0);
    }
    fn priced_fn(task: &mut TaskBuilder) {
        task.cost_fn(|_: &PlanState| 100.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    assert_eq!(summary_of(&domain, priced_a).unwrap().min_cost, 2.0);
    assert_eq!(summary_of(&domain, priced_b).unwrap().min_cost, 3.0);
    assert_eq!(
        summary_of(&domain, priced_fn).unwrap().min_cost,
        0.0,
        "dynamic costs bound at 0"
    );
    assert_eq!(
        summary_of(&domain, root).unwrap().min_cost,
        0.0,
        "min over methods (the dynamic branch)"
    );
}

/// Non-terminating tasks have an infinite `min_cost` (no finite refinement
/// has a cost), and negative declared costs are clamped to 0 so the
/// branch-and-bound lower bounds stay sound.
#[test]
fn min_cost_is_infinite_for_non_terminating_and_clamps_negatives() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(spiral);
        task.branch().then(negative);
    }
    fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }
    fn negative(task: &mut TaskBuilder) {
        task.cost(-3.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    assert_eq!(
        summary_of(&domain, spiral).unwrap().min_cost,
        f32::INFINITY,
        "a task that can only refine forever has no finite cost"
    );
    assert_eq!(
        summary_of(&domain, negative).unwrap().min_cost,
        0.0,
        "negative costs are clamped (branch-and-bound requires non-negative steps)"
    );
}

/// `.cost(c)` followed by `.cost_fn(...)` is last-wins: the dynamic closure
/// replaces the constant, so the stale static cost must not survive into the
/// summary (a stale constant *above* the dynamic truth would be an unsound
/// lower bound and could prune the optimal plan).
#[test]
fn cost_fn_after_cost_clears_the_static_bound() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(overwritten);
    }
    fn overwritten(task: &mut TaskBuilder) {
        task.cost(5.0).cost_fn(|_: &PlanState| 0.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    assert_eq!(
        summary_of(&domain, overwritten).unwrap().min_cost,
        0.0,
        "the dynamic closure cleared the stale static cost"
    );
}

/// Regression (planner-level consequence of the stale-static-cost bug): a
/// branch declared with `.cost(5.0)` but actually costing 1.0 via a later
/// `.cost_fn` must still be explored after a cheaper-looking complete plan
/// exists. With the stale bound (5) surviving into `min_cost`, the
/// branch-and-bound prune (`0 + 5 >= 3`) would skip it and return the
/// statically-priced branch instead of the true optimum.
#[test]
fn stale_static_cost_cannot_prune_the_dynamic_optimum() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(pricey);
        task.branch().then(mislabeled);
    }
    fn pricey(task: &mut TaskBuilder) {
        task.cost(3.0);
    }
    fn mislabeled(task: &mut TaskBuilder) {
        // Declared 5.0, actually 1.0: the stale constant must not become the
        // bake-time lower bound.
        task.cost(5.0).cost_fn(|_: &PlanState| 1.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(
        plan.task_names(),
        ["mislabeled"],
        "the dynamic cost (1.0) beats the static price (3.0)"
    );
    assert_eq!(plan.mtr(), [1]);
}

// ---------------------------------------------------------------------------
// CostBounded: choice + optimality
// ---------------------------------------------------------------------------

/// Two complete plans: the declared-first one is expensive, the declared-last
/// one is cheap. DepthFirst returns the first; CostBounded keeps searching
/// and returns the cheapest.
#[test]
fn cost_bounded_finds_the_cheaper_complete_plan() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(expensive);
        task.branch().then(cheap);
    }
    fn expensive(task: &mut TaskBuilder) {
        task.cost(10.0).effect(|gold: &mut Gold| gold.0 = 100);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(1.0).effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mut dfs = HtnPlanner::new(&domain);
    let dfs_plan = plan_of(&mut dfs, root, &state);
    assert_eq!(dfs_plan.task_names(), ["expensive"], "first complete plan");
    assert_eq!(dfs_plan.mtr(), [0]);

    let mut bnb = HtnPlanner::new(&domain);
    bnb.set_strategy(HtnSearchStrategy::CostBounded);
    let bnb_plan = plan_of(&mut bnb, root, &state);
    assert_eq!(bnb_plan.task_names(), ["cheap"], "cheapest complete plan");
    assert_eq!(bnb_plan.mtr(), [1], "the second branch was taken");
}

/// Three alternatives with descending costs: branch-and-bound must settle on
/// the cheapest regardless of declaration position.
#[test]
fn cost_bounded_selects_the_optimum_among_three_branches() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(c10);
        task.branch().then(c5);
        task.branch().then(c1);
    }
    fn c10(task: &mut TaskBuilder) {
        task.cost(10.0);
    }
    fn c5(task: &mut TaskBuilder) {
        task.cost(5.0);
    }
    fn c1(task: &mut TaskBuilder) {
        task.cost(1.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(plan.task_names(), ["c1"]);
    assert_eq!(plan.mtr(), [2]);
}

/// Dynamic `cost_fn` costs are evaluated against the scratchpad at plan time
/// and drive the choice: the same domain flips its optimum with a component
/// value. (Also exercises pruning against dynamic costs: the static bound of
/// the expensive-at-runtime branch is 0, so the choice comes from comparing
/// complete-plan costs, not from pruning.)
#[test]
fn cost_fn_dynamic_costs_drive_the_choice() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(solo);
        task.branch().then(duo_a).then(duo_b);
    }
    fn solo(task: &mut TaskBuilder) {
        // A (trivially-true) precondition referencing CheapMode registers it as
        // the domain's slot 0 — `cost_fn` closures read the scratchpad raw and
        // register nothing themselves.
        task.precondition(|_: &CheapMode| true)
            .cost_fn(|state: &PlanState| {
                if state.get::<CheapMode>(0).0 {
                    2.0
                } else {
                    20.0
                }
            });
    }
    fn duo_a(task: &mut TaskBuilder) {
        task.cost(5.0);
    }
    fn duo_b(task: &mut TaskBuilder) {
        task.cost(5.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);

    // Cheap mode: solo (2) beats the duo (10).
    let cheap = PlanState::build(&domain.components)
        .set(CheapMode(true))
        .finish();
    assert_eq!(plan_of(&mut planner, root, &cheap).task_names(), ["solo"]);

    // Expensive mode: the duo (10) beats solo (20).
    let pricey = PlanState::build(&domain.components)
        .set(CheapMode(false))
        .finish();
    assert_eq!(
        plan_of(&mut planner, root, &pricey).task_names(),
        ["duo_a", "duo_b"]
    );
}

/// Without any cost annotations every primitive counts 0, every complete
/// plan costs 0, and CostBounded returns exactly what DepthFirst returns
/// (the first complete plan) — the documented degenerate case.
#[test]
fn unannotated_primitives_degenerate_to_depth_first() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(first);
        task.branch().then(second);
    }
    fn first(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 10);
    }
    fn second(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 20);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    let mut dfs = HtnPlanner::new(&domain);
    let mut bnb = HtnPlanner::new(&domain);
    bnb.set_strategy(HtnSearchStrategy::CostBounded);
    assert_eq!(
        plan_of(&mut bnb, root, &state).task_names(),
        plan_of(&mut dfs, root, &state).task_names(),
        "no costs → no bound → first complete plan wins"
    );
}

// ---------------------------------------------------------------------------
// CostBounded: pruning + anytime behavior
// ---------------------------------------------------------------------------

/// A 10-step expensive branch (total 50), a 40-step expensive branch (total
/// 200), and a 1-step cheap branch. Once the first plan (cost 50) is found,
/// the 40-step branch's bake-time bound (200 ≥ 50) must prune it *at the
/// commitment* — within a step budget far too small to explore it. Without
/// the prune, the budget would be exhausted mid-branch and the expensive
/// first plan returned instead.
#[test]
fn cost_bounded_prunes_subtrees_that_cannot_beat_the_best() {
    fn root(task: &mut TaskBuilder) {
        {
            let mut b = task.branch();
            for _ in 0..10 {
                b.then(expensive);
            }
        }
        {
            let mut b = task.branch();
            for _ in 0..40 {
                b.then(expensive);
            }
        }
        task.branch().then(cheap);
    }
    fn expensive(task: &mut TaskBuilder) {
        task.cost(5.0);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(1.0).effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    // Sanity: the bound really is 200 for the 40-step method (the prune's
    // premise), via the root summary: min(10×5, 40×5, 1) = 1.
    assert_eq!(summary_of(&domain, root).unwrap().min_cost, 1.0);

    // Plain DFS: the first complete plan (the 10×5 branch) is the answer —
    // it never even looks at the later branches.
    let mut dfs = HtnPlanner::new(&domain);
    dfs.set_lookahead(false).set_sanity_limit(20);
    assert_eq!(
        plan_of(&mut dfs, root, &state).task_names(),
        ["expensive"; 10]
    );

    // CostBounded within a budget of 20 pops: the 40-step branch is pruned
    // at its commitment (exploring it alone would consume the whole budget),
    // so the cheap branch is still reached and wins.
    let mut bnb = HtnPlanner::new(&domain);
    bnb.set_strategy(HtnSearchStrategy::CostBounded)
        .set_lookahead(false)
        .set_sanity_limit(20);
    let plan = plan_of(&mut bnb, root, &state);
    assert_eq!(
        plan.task_names(),
        ["cheap"],
        "the 40-step branch was pruned, not explored"
    );
    assert_eq!(plan.mtr(), [2]);
}

/// Anytime contract: when the budget runs out *mid-search* (after a complete
/// plan was found but before the cheaper alternative finished), the best
/// complete plan found so far is returned — never a truncated partial.
#[test]
fn cost_bounded_returns_the_best_complete_plan_within_the_budget() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(solo);
        task.branch().then(duo_a).then(duo_b);
    }
    fn solo(task: &mut TaskBuilder) {
        task.cost(10.0);
    }
    fn duo_a(task: &mut TaskBuilder) {
        task.cost(1.0);
    }
    fn duo_b(task: &mut TaskBuilder) {
        task.cost(1.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    // Budget 4: root + solo + root + duo_a — the budget dies right before
    // duo_b. The complete [solo] plan (cost 10) is returned, not the
    // half-built [duo_a].
    let mut tight = HtnPlanner::new(&domain);
    tight.set_strategy(HtnSearchStrategy::CostBounded).set_sanity_limit(4);
    assert_eq!(
        plan_of(&mut tight, root, &state).task_names(),
        ["solo"],
        "anytime: best complete plan within the budget"
    );

    // With room to finish, the cheaper duo plan wins.
    let mut roomy = HtnPlanner::new(&domain);
    roomy.set_strategy(HtnSearchStrategy::CostBounded).set_sanity_limit(10);
    assert_eq!(
        plan_of(&mut roomy, root, &state).task_names(),
        ["duo_a", "duo_b"]
    );
}

/// Regression: the scratchpad must be rolled back between complete plans.
/// The first (expensive) plan sets Gold to 100; the cheaper branch then
/// mutates Gold by +1 and gates on `Gold < 5`. If the search saw the phantom
/// 100, the gate would fail and the cheap plan would be unreachable.
#[test]
fn cost_bounded_rolls_state_back_between_complete_plans() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(set_high);
        task.branch().then(bump).then(gate);
    }
    fn set_high(task: &mut TaskBuilder) {
        task.cost(10.0).effect(|gold: &mut Gold| gold.0 = 100);
    }
    fn bump(task: &mut TaskBuilder) {
        task.cost(1.0).effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn gate(task: &mut TaskBuilder) {
        task.cost(1.0)
            .precondition(|gold: &Gold| gold.0 < 5)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).set(Gold(0)).finish();

    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(),
        ["bump", "gate"],
        "the second branch was evaluated against rolled-back state, not the first plan's effects"
    );
}

/// CostBounded composes with ranked selection policies: HighestUtility
/// explores the shiny-but-expensive branch first, branch-and-bound still
/// settles on the cheap one, and the ranked order is correctly resumed after
/// the stack-empty backtrack.
#[test]
fn cost_bounded_composes_with_ranked_selection() {
    fn root(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::HighestUtility);
        task.branch().named("shiny").utility(100.0).then(expensive);
        task.branch().named("dull").utility(1.0).then(cheap);
    }
    fn expensive(task: &mut TaskBuilder) {
        task.cost(10.0);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(1.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(plan.task_names(), ["cheap"]);
    assert_eq!(plan.mtr(), [1], "the dull branch won on cost");
}

/// A terminal empty branch yields an empty complete plan (cost 0); CostBounded
/// records it as the best and terminates cleanly instead of wedging.
#[test]
fn cost_bounded_handles_the_empty_complete_plan() {
    fn root(task: &mut TaskBuilder) {
        task.branch().precondition(|gold: &Gold| gold.0 >= 0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);
    assert!(plan_of(&mut planner, root, &state).is_empty());
}

// ---------------------------------------------------------------------------
// ECS driver integration
// ---------------------------------------------------------------------------

/// The driver runs the CostBounded strategy: the agent plans and executes the
/// cheap branch, not the declared-first expensive one.
#[test]
fn driver_runs_the_cost_bounded_strategy() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(expensive);
        task.branch().then(cheap);
    }
    fn expensive(task: &mut TaskBuilder) {
        task.cost(10.0).effect(|gold: &mut Gold| gold.0 = 100);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(1.0).effect(|gold: &mut Gold| gold.0 = 1);
    }

    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(HtnDomain::from_root(root).build().unwrap())
            .with_strategy(HtnSearchStrategy::CostBounded),
    );
    let entity = world.spawn((Gold(0), HtnAgent::default())).id();

    htn_ai_system(&mut world);

    assert_eq!(
        world.get::<Gold>(entity).unwrap().0,
        1,
        "the cheap branch was planned and executed"
    );
    assert!(world.get::<HtnAgent>(entity).unwrap().plan.is_none());
}

/// A per-agent `SearchOverride` can opt one agent into CostBounded while the
/// global strategy stays DepthFirst — the two agents plan differently from
/// the same domain in the same tick.
#[test]
fn per_agent_override_opts_into_cost_bounded() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(expensive);
        task.branch().then(cheap);
    }
    fn expensive(task: &mut TaskBuilder) {
        task.cost(10.0).effect(|gold: &mut Gold| gold.0 = 100);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(1.0).effect(|gold: &mut Gold| gold.0 = 1);
    }

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(HtnDomain::from_root(root).build().unwrap()));
    let global = world.spawn((Gold(0), HtnAgent::default())).id();
    let overridden = world
        .spawn((
            Gold(0),
            HtnAgent::default(),
            SearchOverride {
                strategy: Some(HtnSearchStrategy::CostBounded),
                sanity_limit: None,
            },
        ))
        .id();

    htn_ai_system(&mut world);

    assert_eq!(
        world.get::<Gold>(global).unwrap().0,
        100,
        "global DepthFirst took the first (expensive) branch"
    );
    assert_eq!(
        world.get::<Gold>(overridden).unwrap().0,
        1,
        "the override planned the cheap branch"
    );
}
