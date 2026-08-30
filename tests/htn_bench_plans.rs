//! Certainty tests for the benchmarks: pin the **exact plans** the benches
//! produce, and the exact per-cycle behavior of their plan → execute → replan
//! loop.
//!
//! This file includes `benches/common/mod.rs` directly (`#[path]`), so it runs
//! against *exactly* the code the benches run — same `.htn` fixtures (the
//! top-level `htn/` directory), same state constructors, same execution
//! helper. Any change that alters what a benchmark plans (domain edit, state
//! seeding, execution semantics, planner behavior) fails here first.

#[path = "../benches/common/mod.rs"]
mod bench_common;

use bevy_bhtn::parse_htn;
use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::HtnState;
use bevy_reflect::{Reflect, TypeRegistry};

use bench_common::*;

fn miner_bed() -> (bevy_bhtn::HtnDomain, TypeRegistry) {
    let mut registry = TypeRegistry::default();
    register_miner(&mut registry);
    (
        parse_htn(MINER_HTN).expect("miner fixture parses"),
        registry,
    )
}

fn outpost_bed() -> (bevy_bhtn::HtnDomain, TypeRegistry) {
    let mut registry = TypeRegistry::default();
    register_outpost(&mut registry);
    (
        parse_htn(OUTPOST_HTN).expect("outpost fixture parses"),
        registry,
    )
}

/// Helper: plan and render the exact task-name sequence.
fn plan_of<S: HtnState>(
    domain: &bevy_bhtn::HtnDomain,
    registry: &TypeRegistry,
    root: &str,
    state: &S,
) -> Vec<String> {
    let mut planner = HtnPlanner::new(domain, registry);
    plan_of_with(&mut planner, domain, root, state)
}

fn plan_of_with<S: HtnState>(
    planner: &mut HtnPlanner,
    domain: &bevy_bhtn::HtnDomain,
    root: &str,
    state: &S,
) -> Vec<String> {
    planner
        .plan(root, state)
        .task_names()
        .iter()
        .map(|n| n.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Miner bench (`ai_throughput`): exact plans for the bench's state classes
// ---------------------------------------------------------------------------

#[test]
fn miner_bench_states_plan_exactly() {
    let (domain, registry) = miner_bed();

    // Representative entities of the bench's spawn batch (`i`-th entity):
    // every residue class of gold (i%5), ore (i%3), metal (i%7), energy
    // (i%40), hunger (i%60).
    let expected: &[(usize, &[&str])] = &[
        // Starts holding metal (i%7==0): sells it, then mines to top up.
        (
            0,
            &[
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
        // Empty-handed: the full mine → smelt → sell loop, twice.
        (
            1,
            &[
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
        // One gold short: a single loop.
        (
            2,
            &[
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
        // Goal already met: no work at all.
        (3, &[]),
        // Two short: the loop, three times.
        (
            5,
            &[
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
        // Holding metal, one gold short: sell, done.
        (7, &["GoToMerchant", "SellMetal", "GoToOutside"]),
        (
            11,
            &[
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
        (13, &[]),
        (19, &[]),
        (29, &[]),
        (
            41,
            &[
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
        (
            97,
            &[
                "GoToOre",
                "MineOre",
                "GoToSmelter",
                "SmeltOre",
                "GoToOutside",
                "GoToMerchant",
                "SellMetal",
                "GoToOutside",
            ],
        ),
    ];

    for (i, want) in expected {
        let state = initial_miner(*i);
        let got = plan_of(&domain, &registry, "EarnGold", &state);
        assert_eq!(&got, want, "miner entity {i} plan diverged");
    }
}

/// The look-ahead must never change *which* plan the miner bench finds.
#[test]
fn miner_bench_plans_identical_without_lookahead() {
    let (domain, registry) = miner_bed();
    for i in [0usize, 1, 2, 3, 5, 7, 11, 13, 41, 97] {
        let state = initial_miner(i);
        let with = plan_of(&domain, &registry, "EarnGold", &state);
        let mut planner = HtnPlanner::new(&domain, &registry);
        planner.set_lookahead(false);
        let without = plan_of_with(&mut planner, &domain, "EarnGold", &state);
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
    let (domain, registry) = outpost_bed();

    // Fresh actor: the full objective chain (march — fuel 5 < 8 — then watch,
    // reinforce via convoy, arm via supply run, clear the cache).
    assert_eq!(
        plan_of(&domain, &registry, "SecureOutpost", &fresh_outpost()),
        [
            "Hike",
            "WatchPost",
            "Convoy",
            "Rally",
            "Supply",
            "BoltArmor",
            "ClearCache"
        ]
    );

    // Marginal fuel: the "queue anyway" drive branch is eligible but its leaf
    // fails → the plan must detour through the rations march AND siphon fuel,
    // and rest at the end (morale 0 < 5).
    assert_eq!(
        plan_of(&domain, &registry, "SecureOutpost", &marginal_outpost()),
        [
            "Hike",
            "WatchPost",
            "Siphon",
            "Convoy",
            "Rally",
            "Supply",
            "BoltArmor",
            "ClearCache",
            "Rest"
        ]
    );

    // High fuel: the direct vehicle run leads the plan.
    assert_eq!(
        plan_of(&domain, &registry, "SecureOutpost", &high_fuel_outpost()),
        [
            "Drive",
            "WatchPost",
            "Convoy",
            "Rally",
            "Supply",
            "BoltArmor",
            "ClearCache",
            "Rest"
        ]
    );
}

/// The look-ahead must never change *which* plan the outpost bench finds.
#[test]
fn outpost_bench_plans_identical_without_lookahead() {
    let (domain, registry) = outpost_bed();
    for state in [fresh_outpost(), marginal_outpost(), high_fuel_outpost()] {
        let with = plan_of(&domain, &registry, "SecureOutpost", &state);
        let mut planner = HtnPlanner::new(&domain, &registry);
        planner.set_lookahead(false);
        let without = plan_of_with(&mut planner, &domain, "SecureOutpost", &state);
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
    let mut registry = TypeRegistry::default();
    register_gate(&mut registry);

    // Exponential: the doomed gate chain is refuted (gold 0 < 1000 and no
    // gate writes it) → the direct method.
    let domain = parse_htn(EXPONENTIAL_HTN).expect("exponential fixture parses");
    assert_eq!(
        plan_of(&domain, &registry, "Root", &GateState::default()),
        ["Strike", "Final"]
    );

    // Doomed recursion: the non-terminating spiral is refuted → safe method.
    let domain = parse_htn(DOOMED_RECURSION_HTN).expect("doomed fixture parses");
    assert_eq!(
        plan_of(&domain, &registry, "Act", &GateState::default()),
        ["Safe"]
    );
}

// ---------------------------------------------------------------------------
// The benches' plan → execute → replan cycle (10 iterations)
// ---------------------------------------------------------------------------

/// Run exactly what the benches run per measured iteration: `cycles` rounds of
/// plan → execute-one-step → replan, against the working state. Returns every
/// cycle's plan and the final state.
fn run_replan_cycle<S: HtnState>(
    domain: &bevy_bhtn::HtnDomain,
    registry: &TypeRegistry,
    root: &str,
    initial: &S,
    cycles: usize,
) -> (Vec<Vec<String>>, S) {
    let mut state = initial.clone();
    let mut planner = HtnPlanner::new(domain, registry);
    let mut plans = Vec::with_capacity(cycles);
    for _ in 0..cycles {
        let plan = planner.plan(root, &state);
        plans.push(plan_of_with(&mut planner, domain, root, &state));
        execute_plan_step(domain, registry, state.as_reflect_mut(), &plan);
    }
    (plans, state)
}

#[test]
fn miner_replan_cycle_progresses_deterministically() {
    let (domain, registry) = miner_bed();
    let (plans, final_state) =
        run_replan_cycle(&domain, &registry, "EarnGold", &initial_miner(0), 10);

    // The agent works through its plan one action per cycle: the plan shrinks
    // as state advances (sell the held metal, smelt, mine), then settles into
    // the steady mining loop (the fixture's `GoToOre` has no preconditions, so
    // the head task is re-planned while standing at the ore patch).
    let lengths: Vec<usize> = plans.iter().map(|p| p.len()).collect();
    assert_eq!(lengths, [17, 16, 14, 13, 11, 10, 8, 8, 8, 8]);

    let heads: Vec<&str> = plans
        .iter()
        .map(|p| p.first().map(String::as_str).unwrap_or(""))
        .collect();
    assert_eq!(
        heads,
        [
            "GoToMerchant",
            "SellMetal",
            "GoToSmelter",
            "SmeltOre",
            "GoToMerchant",
            "SellMetal",
            "GoToOre",
            "GoToOre",
            "GoToOre",
            "GoToOre"
        ]
    );

    // Ten executed actions, deterministically: sold the held metal (gold 1),
    // smelted + sold it (gold 2), then walked to the ore patch.
    assert_eq!(final_state.gold, 2);
    assert!(!final_state.has_metal);
    assert_eq!(final_state.location, Location::Ore);
}

#[test]
fn outpost_replan_cycle_completes_all_objectives() {
    let (domain, registry) = outpost_bed();
    let (plans, final_state) =
        run_replan_cycle(&domain, &registry, "SecureOutpost", &fresh_outpost(), 10);

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
            "Hike",
            "WatchPost",
            "Convoy",
            "Rally",
            "Supply",
            "BoltArmor",
            "ClearCache",
            "",
            "",
            ""
        ]
    );

    // Executing the plans secures every objective — the outpost is done.
    assert!(final_state.perimeter);
    assert!(final_state.reinforced);
    assert!(final_state.armored);
    assert!(final_state.caches);
    assert_eq!(final_state.position, Zone::Armory);
}
