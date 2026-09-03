//! Certainty tests for the benchmarks: pin the **exact plans** the benches
//! produce, and the exact per-cycle behavior of their plan → execute → replan
//! loop.
//!
//! This file includes `benches/common/mod.rs` directly (`#[path]`), so it runs
//! against *exactly* the code the benches run — same function-defined fixture
//! domains, same `PlanState` constructors, same execution helper. Any change
//! that alters what a benchmark plans (domain edit, state seeding, execution
//! semantics, planner behavior) fails here first.

#[path = "../benches/common/mod.rs"]
mod bench_common;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::TaskFn;

use bench_common::*;

/// Helper: plan and render the exact task-name sequence. The root fn-item
/// type is inferred from the fn value (fn-item types cannot be named
/// directly in turbofish).
fn plan_of<F: TaskFn>(domain: &bevy_bhtn::HtnDomain, _root: F, state: &PlanState) -> Vec<String> {
    let mut planner = HtnPlanner::new(domain);
    planner
        .plan(_root, state)
        .task_names()
        .iter()
        .map(|n| n.to_string())
        .collect()
}

/// Helper: plan with an existing (possibly tuned) planner.
fn plan_of_with<F: TaskFn>(planner: &mut HtnPlanner, _root: F, state: &PlanState) -> Vec<String> {
    planner
        .plan(_root, state)
        .task_names()
        .iter()
        .map(|n| n.to_string())
        .collect()
}

/// Helper: read component `T` out of a scratchpad via the domain's registry.
fn scratch_get<T: bevy_bhtn::state::PlanComponent>(
    domain: &bevy_bhtn::HtnDomain,
    state: &PlanState,
) -> T
where
    T: Clone,
{
    state
        .get::<T>(
            domain
                .components
                .get::<T>()
                .expect("component is registered"),
        )
        .clone()
}

// ---------------------------------------------------------------------------
// Miner bench (`ai_throughput`): exact plans for the bench's state classes
// ---------------------------------------------------------------------------

#[test]
fn miner_bench_states_plan_exactly() {
    let domain = miner_domain();

    // Representative entities of the bench's spawn batch (`i`-th entity):
    // every residue class of gold (i%5), ore (i%3), metal (i%7), energy
    // (i%40), hunger (i%60).
    let expected: &[(usize, &[&str])] = &[
        // Starts holding metal (i%7==0): sells it, then mines to top up.
        (
            0,
            &[
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
        // Empty-handed: the full mine → smelt → sell loop, twice.
        (
            1,
            &[
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
        // One gold short: a single loop.
        (
            2,
            &[
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
        // Goal already met: no work at all.
        (3, &[]),
        // Two short: the loop, three times.
        (
            5,
            &[
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
        // Holding metal, one gold short: sell, done.
        (7, &["go_to_merchant", "sell_metal", "go_to_outside"]),
        (
            11,
            &[
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
        (13, &[]),
        (19, &[]),
        (29, &[]),
        (
            41,
            &[
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
        (
            97,
            &[
                "go_to_ore",
                "mine_ore",
                "go_to_smelter",
                "smelt_ore",
                "go_to_outside",
                "go_to_merchant",
                "sell_metal",
                "go_to_outside",
            ],
        ),
    ];

    for (i, want) in expected {
        let state = miner_scratch(&domain, *i);
        let got = plan_of(&domain, miner_tasks::earn_gold, &state);
        assert_eq!(&got, want, "miner entity {i} plan diverged");
    }
}

/// The look-ahead must never change *which* plan the miner bench finds.
#[test]
fn miner_bench_plans_identical_without_lookahead() {
    let domain = miner_domain();
    for i in [0usize, 1, 2, 3, 5, 7, 11, 13, 41, 97] {
        let state = miner_scratch(&domain, i);
        let with = plan_of(&domain, miner_tasks::earn_gold, &state);
        let mut planner = HtnPlanner::new(&domain);
        planner.set_lookahead(false);
        let without = plan_of_with(&mut planner, miner_tasks::earn_gold, &state);
        assert_eq!(
            with, without,
            "miner entity {i}: look-ahead changed the plan"
        );
    }
}

// ---------------------------------------------------------------------------
// Outpost bench (`deep_ai`): exact plans for the bench's state classes
// ---------------------------------------------------------------------------

#[test]
fn outpost_bench_states_plan_exactly() {
    let domain = outpost_domain();

    // Fresh actor: the full objective chain (march — fuel 5 < 8 — then watch,
    // reinforce via convoy, arm via supply run, clear the cache).
    assert_eq!(
        plan_of(
            &domain,
            outpost_tasks::secure_outpost,
            &outpost_scratch(&domain, fresh_outpost())
        ),
        [
            "hike",
            "watch_post",
            "convoy",
            "rally",
            "supply",
            "bolt_armor",
            "clear_cache"
        ]
    );

    // Marginal fuel: the "queue anyway" drive branch is eligible but its leaf
    // fails → the plan must detour through the rations march AND siphon fuel,
    // and rest at the end (morale 0 < 5).
    assert_eq!(
        plan_of(
            &domain,
            outpost_tasks::secure_outpost,
            &outpost_scratch(&domain, marginal_outpost())
        ),
        [
            "hike",
            "watch_post",
            "siphon",
            "convoy",
            "rally",
            "supply",
            "bolt_armor",
            "clear_cache",
            "rest"
        ]
    );

    // High fuel: the direct vehicle run leads the plan.
    assert_eq!(
        plan_of(
            &domain,
            outpost_tasks::secure_outpost,
            &outpost_scratch(&domain, high_fuel_outpost())
        ),
        [
            "drive",
            "watch_post",
            "convoy",
            "rally",
            "supply",
            "bolt_armor",
            "clear_cache",
            "rest"
        ]
    );
}

/// The look-ahead must never change *which* plan the outpost bench finds.
#[test]
fn outpost_bench_plans_identical_without_lookahead() {
    let domain = outpost_domain();
    for state in [
        outpost_scratch(&domain, fresh_outpost()),
        outpost_scratch(&domain, marginal_outpost()),
        outpost_scratch(&domain, high_fuel_outpost()),
    ] {
        let with = plan_of(&domain, outpost_tasks::secure_outpost, &state);
        let mut planner = HtnPlanner::new(&domain);
        planner.set_lookahead(false);
        let without = plan_of_with(&mut planner, outpost_tasks::secure_outpost, &state);
        assert_eq!(
            with, without,
            "outpost state {state:?}: look-ahead changed the plan"
        );
    }
}

// ---------------------------------------------------------------------------
// Look-ahead bench domains: exact plans
// ---------------------------------------------------------------------------

#[test]
fn lookahead_bench_domains_plan_exactly() {
    // Exponential: the doomed gate chain is refuted (gold 0 < 1000 and no
    // gate writes it) → the direct method.
    let domain = gate_domain();
    let state = PlanState::build(&domain.components).finish();
    assert_eq!(
        plan_of(&domain, gate_tasks::gate_root, &state),
        ["strike", "gate_final"]
    );

    // Doomed recursion: the non-terminating spiral is refuted → safe method.
    let domain = doomed_recursion_domain();
    let state = PlanState::build(&domain.components).finish();
    assert_eq!(plan_of(&domain, doomed_tasks::act, &state), ["safe"]);
}

// ---------------------------------------------------------------------------
// The benches' plan → execute → replan cycle (10 iterations)
// ---------------------------------------------------------------------------

/// Run exactly what the benches run per measured iteration: `cycles` rounds of
/// plan → execute-one-step → replan, against the working state. Returns every
/// cycle's plan and the final state.
fn run_replan_cycle<F: TaskFn + Copy>(
    domain: &bevy_bhtn::HtnDomain,
    _root: F,
    initial: &PlanState,
    cycles: usize,
) -> (Vec<Vec<String>>, PlanState) {
    let mut state = initial.clone();
    let mut planner = HtnPlanner::new(domain);
    let mut plans = Vec::with_capacity(cycles);
    for _ in 0..cycles {
        let plan = planner.plan(_root, &state);
        plans.push(plan.task_names().iter().map(|n| n.to_string()).collect());
        execute_plan_step(domain, &mut state, &plan);
    }
    (plans, state)
}

#[test]
fn miner_replan_cycle_progresses_deterministically() {
    let domain = miner_domain();
    let initial = miner_scratch(&domain, 0);
    let (plans, final_state) = run_replan_cycle(&domain, miner_tasks::earn_gold, &initial, 10);

    // The agent works through its plan one action per cycle: the plan shrinks
    // as state advances (sell the held metal, smelt, mine), then settles into
    // the steady mining loop (the fixture's `go_to_ore` has no preconditions,
    // so the head task is re-planned while standing at the ore patch).
    let lengths: Vec<usize> = plans.iter().map(|p| p.len()).collect();
    assert_eq!(lengths, [17, 16, 14, 13, 11, 10, 8, 8, 8, 8]);

    let heads: Vec<&str> = plans
        .iter()
        .map(|p| p.first().map(String::as_str).unwrap_or(""))
        .collect();
    assert_eq!(
        heads,
        [
            "go_to_merchant",
            "sell_metal",
            "go_to_smelter",
            "smelt_ore",
            "go_to_merchant",
            "sell_metal",
            "go_to_ore",
            "go_to_ore",
            "go_to_ore",
            "go_to_ore"
        ]
    );

    // Ten executed actions, deterministically: sold the held metal (gold 1),
    // smelted + sold it (gold 2), then walked to the ore patch.
    assert_eq!(scratch_get::<Gold>(&domain, &final_state).0, 2);
    assert!(!scratch_get::<HasMetal>(&domain, &final_state).0);
    assert_eq!(
        scratch_get::<Location>(&domain, &final_state),
        Location::Ore
    );
}

#[test]
fn outpost_replan_cycle_completes_all_objectives() {
    let domain = outpost_domain();
    let initial = outpost_scratch(&domain, fresh_outpost());
    let (plans, final_state) =
        run_replan_cycle(&domain, outpost_tasks::secure_outpost, &initial, 10);

    // One objective per cycle until the domain's terminal method takes over:
    // the plan shrinks to empty once every objective is secured.
    let lengths: Vec<usize> = plans.iter().map(|p| p.len()).collect();
    assert_eq!(lengths, [7, 6, 5, 4, 3, 2, 1, 0, 0, 0]);

    let heads: Vec<&str> = plans
        .iter()
        .map(|p| p.first().map(String::as_str).unwrap_or(""))
        .collect();
    assert_eq!(
        heads,
        [
            "hike",
            "watch_post",
            "convoy",
            "rally",
            "supply",
            "bolt_armor",
            "clear_cache",
            "",
            "",
            ""
        ]
    );

    // Executing the plans secures every objective — the outpost is done.
    assert!(scratch_get::<Perimeter>(&domain, &final_state).0);
    assert!(scratch_get::<Reinforced>(&domain, &final_state).0);
    assert!(scratch_get::<Armored>(&domain, &final_state).0);
    assert!(scratch_get::<Caches>(&domain, &final_state).0);
    assert_eq!(scratch_get::<Zone>(&domain, &final_state), Zone::Armory);
}
