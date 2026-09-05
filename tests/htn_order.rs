//! Pins for Phase 3 (Axis 2): partially-ordered subtask sets scheduled by the
//! DFS — `subtask` / `before` / `any_order` — plus the conservative summary
//! and look-ahead extensions.
//!
//! The load-bearing guarantees pinned here:
//!
//! - **Scheduling by backtracking**: when the declaration order fails, the
//!   planner retries the *same method* with its next topological order — a
//!   domain that is only planable in a non-declaration linearization plans
//!   (without this, the method would fail outright).
//! - **`before` constraints** restrict the explored orders even when the
//!   declaration order is *not* topological.
//! - **`then` tails** run after the whole unordered set.
//! - **MTR** records the method index only — the chosen permutation is not
//!   recorded, and a linearization retry leaves exactly one entry.
//! - **Summaries** are order-independent (min-yield sums; required fields use
//!   the conservative "read by a member nobody can write" rule).
//! - **Look-ahead** refutes dead sets under set semantics.
//! - **Builder validation**: cyclic `before` constraints and >64-member sets
//!   are build errors.

use bevy_bhtn::domain::SelectionPolicy;
use bevy_bhtn::selection::LookaheadMode;
use bevy_bhtn::domain::Task;
use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
use bevy_bhtn::order::SubtaskOrder;
use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::selection::HtnSearchStrategy;
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
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Key(bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Wood(i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Stone(i32);

// ---------------------------------------------------------------------------
// Scheduling semantics
// ---------------------------------------------------------------------------

/// An `any_order` set runs every member exactly once; the default
/// linearization is the declaration order.
#[test]
fn any_order_runs_every_member_in_declaration_order_by_default() {
    fn root(task: &mut TaskBuilder) {
        task.branch()
            .any_order((gather_wood, gather_stone, sell_spoils));
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }
    fn sell_spoils(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(&domain),
        ["gather_wood", "gather_stone", "sell_spoils"],
        "order 0 is the declaration order"
    );
}

/// THE scheduling test: the declaration order fails (the door is locked when
/// tried first) but the reversed linearization succeeds. The planner must
/// retry the same method with its next topological order — without that, the
/// method would fail outright and the fallback would run.
#[test]
fn unordered_set_backtracks_to_a_valid_linearization() {
    fn root(task: &mut TaskBuilder) {
        // Declared door-first: fails in linearization 0.
        task.branch()
            .named("set")
            .any_order((unlock_door, find_key));
        task.branch().named("fallback").then(give_up);
    }
    fn unlock_door(task: &mut TaskBuilder) {
        task.precondition(|key: &Key| key.0)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn find_key(task: &mut TaskBuilder) {
        task.effect(|key: &mut Key| key.0 = true);
    }
    fn give_up(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = -1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(
        plan.task_names(&domain),
        ["find_key", "unlock_door"],
        "the reversed linearization was found by backtracking"
    );
    assert_eq!(
        plan.mtr(),
        [0],
        "the MTR records the method index only — no permutation entry"
    );
}

/// `before` constraints restrict the explored orders even when the
/// declaration order is *not* topological: the first topological order (not
/// the declaration order) is committed.
#[test]
fn before_constraints_force_the_first_topological_order() {
    fn root(task: &mut TaskBuilder) {
        let mut b = task.branch();
        // Declared cook-first, but the constraint says fetch first.
        let cook = b.subtask(cook_meal);
        let fetch = b.subtask(fetch_ingredient);
        b.before(fetch, cook);
    }
    fn fetch_ingredient(task: &mut TaskBuilder) {
        task.effect(|key: &mut Key| key.0 = true);
    }
    fn cook_meal(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(&domain),
        ["fetch_ingredient", "cook_meal"],
        "the constraint DAG, not declaration order, decides"
    );
}

/// A `then` member declared after unordered members runs after the whole
/// set; the structural order count reflects the constraint DAG.
#[test]
fn then_tail_runs_after_the_unordered_set() {
    fn root(task: &mut TaskBuilder) {
        let mut b = task.branch();
        b.subtask(gather_wood);
        b.subtask(gather_stone);
        b.then(sell_spoils);
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }
    fn sell_spoils(task: &mut TaskBuilder) {
        task.precondition(|wood: &Wood| wood.0 > 0)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(&domain),
        ["gather_wood", "gather_stone", "sell_spoils"]
    );

    // Structural: the two unordered members admit 2! = 2 linearizations
    // (the tail is constrained after both).
    let Task::Compound(c) = domain.get_task("root").unwrap() else {
        panic!("root is compound");
    };
    match &c.methods[0].order {
        SubtaskOrder::Partial { orders, .. } => assert_eq!(*orders, 2),
        SubtaskOrder::Total => panic!("the branch should be partially ordered"),
    }
}

/// A compound member inside an unordered set decomposes normally, and the
/// rest of the set runs after its decomposition completes.
#[test]
fn compound_members_decompose_inside_unordered_sets() {
    fn root(task: &mut TaskBuilder) {
        task.branch().any_order((do_chores, unlock_door));
    }
    fn do_chores(task: &mut TaskBuilder) {
        task.branch().then(gather_wood).then(gather_stone);
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }
    fn unlock_door(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(&domain),
        ["gather_wood", "gather_stone", "unlock_door"],
        "the compound member's full decomposition precedes the next member"
    );
}

// ---------------------------------------------------------------------------
// Summaries (conservative extensions)
// ---------------------------------------------------------------------------

/// `min_yield` of a partially-ordered method is the order-independent sum
/// over its members.
#[test]
fn partial_method_min_yield_sums_members() {
    fn root(task: &mut TaskBuilder) {
        task.branch().any_order((gather_wood, gather_stone));
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    assert_eq!(summary_of(&domain, root).unwrap().min_yield, 2);
}

/// `required_fields` uses the conservative set rule: a component is required
/// only if some member reads it and **no** member can write it. A component
/// that another member may write is not required (some linearization writes
/// it before the reader runs).
#[test]
fn partial_method_required_fields_are_conservative() {
    fn root(task: &mut TaskBuilder) {
        task.branch().any_order((read_gold, write_key));
    }
    fn read_gold(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 >= 0);
    }
    fn write_key(task: &mut TaskBuilder) {
        task.effect(|key: &mut Key| key.0 = true);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let required = &summary_of(&domain, root).unwrap().required_fields;
    let gold = domain.components.slot_of::<Gold>().unwrap();
    let key = domain.components.slot_of::<Key>().unwrap();
    assert!(
        required.contains(gold),
        "Gold is read by a member and no member can write it — required"
    );
    assert!(
        !required.contains(key),
        "Key is never read at all — not required"
    );
}

/// The look-ahead sweep refutes a dead set under set semantics: a member
/// whose precondition definitely fails (and that no member can fix) kills
/// the whole method at the frame. Differentiated by the sanity budget: with
/// the sweep, the fallback is reached within a tiny budget; without it, the
/// doomed set is explored and the budget dies mid-branch.
#[test]
fn lookahead_refutes_dead_sets() {
    fn root(task: &mut TaskBuilder) {
        task.branch().any_order((impossible, busywork));
        task.branch().then(works);
    }
    fn impossible(task: &mut TaskBuilder) {
        // Reads Gold; nothing in the set (or anywhere) writes it.
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn busywork(task: &mut TaskBuilder) {
        task.effect(|key: &mut Key| key.0 = true);
    }
    fn works(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();

    // Look-ahead on: the dead set is refuted at the frame; the fallback is
    // planned within a 2-step budget.
    let mut on = HtnPlanner::new(&domain);
    on.set_sanity_limit(2);
    assert_eq!(
        plan_of(&mut on, root, &state).task_names(&domain),
        ["works"],
        "the dead set was refuted without exploring it"
    );

    // Look-ahead off: the set is explored — impossible fails, the reversed
    // linearization runs busywork before failing again — and the 4-step
    // budget dies mid-set with only busywork committed.
    let mut off = HtnPlanner::new(&domain);
    off.set_lookahead_mode(LookaheadMode::Off).set_sanity_limit(4);
    assert_eq!(
        plan_of(&mut off, root, &state).task_names(&domain),
        ["busywork"],
        "without the sweep, the budget is spent inside the doomed set"
    );
}

// ---------------------------------------------------------------------------
// CostBounded × partial orders
// ---------------------------------------------------------------------------

/// The branch-and-bound prune uses the order-independent member sum: a
/// partially-ordered method whose members cost 5+5 is pruned against a
/// cheaper alternative exactly like a total-order method.
#[test]
fn cost_bounded_prunes_partial_methods_by_their_member_sum() {
    fn root(task: &mut TaskBuilder) {
        task.branch().any_order((costly_a, costly_b));
        task.branch().then(cheap);
    }
    fn costly_a(task: &mut TaskBuilder) {
        task.cost(5.0);
    }
    fn costly_b(task: &mut TaskBuilder) {
        task.cost(5.0);
    }
    fn cheap(task: &mut TaskBuilder) {
        task.cost(1.0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_strategy(HtnSearchStrategy::CostBounded);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(plan.task_names(&domain), ["cheap"]);
    assert_eq!(plan.mtr(), [1]);
}

// ---------------------------------------------------------------------------
// Builder validation
// ---------------------------------------------------------------------------

/// Cyclic `before` constraints (including a self-constraint) are build-time
/// errors.
#[test]
fn cyclic_before_constraints_are_build_errors() {
    fn cyclic(task: &mut TaskBuilder) {
        let mut b = task.branch();
        let a = b.subtask(gather_wood);
        let c = b.subtask(gather_stone);
        b.before(a, c).before(c, a);
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }
    assert!(HtnDomain::from_root(cyclic).build().is_err());

    fn self_loop(task: &mut TaskBuilder) {
        let mut b = task.branch();
        let a = b.subtask(gather_wood);
        b.before(a, a);
    }
    assert!(HtnDomain::from_root(self_loop).build().is_err());
}

/// A partially-ordered branch with more than 64 members is rejected (the
/// predecessor bitmasks are u64).
#[test]
fn partial_methods_over_64_members_are_rejected() {
    fn wide(task: &mut TaskBuilder) {
        let mut b = task.branch();
        for _ in 0..65 {
            b.subtask(gather_wood);
        }
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    let err = HtnDomain::from_root(wide).build().unwrap_err();
    assert!(
        err.to_string().contains("64"),
        "error names the limit: {err}"
    );

    // 64 members exactly: accepted.
    fn at_limit(task: &mut TaskBuilder) {
        let mut b = task.branch();
        for _ in 0..64 {
            b.subtask(gather_wood);
        }
    }
    assert!(HtnDomain::from_root(at_limit).build().is_ok());
}

// ---------------------------------------------------------------------------
// Wide-set envelope (documented limitation)
// ---------------------------------------------------------------------------

/// Linearizations enumerate deterministically (declaration order first) and
/// are capped at `LINEARIZATION_CAP` (64). A wide unconstrained set whose
/// dependency is buried early — member 0 needs member 1, and 7! = 5040
/// orders share the member-0 prefix — cannot reach a valid order within the
/// cap. This pins the envelope: unordered sets with cross-member
/// dependencies are reliable up to ~4 members (4! = 24 ≤ 64); wider or
/// data-driven member sets belong in recursion over components (measured in
/// `benches/wide_sets.rs`).
#[test]
fn wide_set_envelope_buried_dependencies() {
    fn locked(task: &mut TaskBuilder) {
        task.precondition(|key: &Key| key.0)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn find_key(task: &mut TaskBuilder) {
        task.effect(|key: &mut Key| key.0 = true);
    }
    fn wide8(task: &mut TaskBuilder) {
        task.branch().any_order((
            locked,
            find_key,
            gather_wood,
            gather_stone,
            noop_e,
            noop_f,
            noop_g,
            noop_h,
        ));
        task.branch().then(fallback);
    }
    fn wide4(task: &mut TaskBuilder) {
        task.branch()
            .any_order((locked, find_key, gather_wood, gather_stone));
        task.branch().then(fallback);
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }
    fn noop_e(_task: &mut TaskBuilder) {}
    fn noop_f(_task: &mut TaskBuilder) {}
    fn noop_g(_task: &mut TaskBuilder) {}
    fn noop_h(_task: &mut TaskBuilder) {}
    fn fallback(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = -1);
    }

    // 4 members: 4! = 24 ≤ cap — every linearization is reachable, and the
    // key-first order is found by backtracking.
    let d4 = HtnDomain::from_root(wide4).build().unwrap();
    let s4 = PlanState::build(&d4.components).finish();
    let mut p4 = HtnPlanner::new(&d4);
    assert_eq!(
        plan_of(&mut p4, wide4, &s4).task_names(&d4),
        ["find_key", "locked", "gather_wood", "gather_stone"],
        "within the cap, the valid linearization is reached"
    );

    // 8 members: beyond the cap the method fails. Within the default sanity
    // budget the retry storm consumes the budget and the planner returns an
    // empty partial; with a raised budget the cap exhausts cleanly and the
    // fallback plans.
    let d8 = HtnDomain::from_root(wide8).build().unwrap();
    let s8 = PlanState::build(&d8.components).finish();
    let mut p8 = HtnPlanner::new(&d8);
    assert!(
        plan_of(&mut p8, wide8, &s8).is_empty(),
        "default budget: the retry storm consumes the sanity limit"
    );
    let mut p8b = HtnPlanner::new(&d8);
    p8b.set_sanity_limit(10_000);
    assert_eq!(
        plan_of(&mut p8b, wide8, &s8b_state(&d8)).task_names(&d8),
        ["fallback"],
        "raised budget: the cap exhausts and the fallback plans"
    );
}

fn s8b_state(domain: &HtnDomain) -> PlanState {
    PlanState::build(&domain.components).finish()
}

// ---------------------------------------------------------------------------
// ECS driver integration
// ---------------------------------------------------------------------------

/// The driver plans and executes a partially-ordered plan step by step; the
/// compiled program and executor are unchanged.
#[test]
fn driver_executes_partially_ordered_plans() {
    fn root(task: &mut TaskBuilder) {
        task.branch().any_order((gather_wood, gather_stone));
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }

    let mut world = World::new();
    world.insert_resource(HtnConfig::new(HtnDomain::from_root(root).build().unwrap()));
    let entity = world.spawn((Wood(0), Stone(0), HtnAgent::default())).id();

    htn_ai_system(&mut world);
    assert_eq!(world.get::<Wood>(entity).unwrap().0, 1, "first step ran");
    assert!(world.get::<HtnAgent>(entity).unwrap().plan().is_some());

    htn_ai_system(&mut world);
    assert_eq!(world.get::<Stone>(entity).unwrap().0, 1, "second step ran");
    assert!(world.get::<HtnAgent>(entity).unwrap().plan().is_none());
}

// ---------------------------------------------------------------------------
// Regression: linearization retries must not corrupt the search state
// ---------------------------------------------------------------------------

/// A ranked compound member inside an unordered set: the set's scheduling
/// machinery composes with selection policies — the compound decomposes via
/// its ranked branches and the remaining member runs after it.
#[test]
fn ranked_compound_inside_unordered_set() {
    fn root(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::HighestUtility);
        task.branch().any_order((prepare, unlock_door));
        task.branch().then(give_up);
    }
    fn prepare(task: &mut TaskBuilder) {
        task.branch().named("good").utility(10.0).then(find_key);
        task.branch().named("bad").utility(1.0).then(give_up);
    }
    fn find_key(task: &mut TaskBuilder) {
        task.effect(|key: &mut Key| key.0 = true);
    }
    fn unlock_door(task: &mut TaskBuilder) {
        task.precondition(|key: &Key| key.0)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn give_up(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = -1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, root, &state).task_names(&domain),
        ["find_key", "unlock_door"],
        "the compound member decomposed via its highest-utility branch"
    );
}

/// Regression: a linearization retry after members already ran must roll the
/// scratchpad back to commitment time — the retried order re-runs every
/// member from the original state, not on top of the failed attempt's
/// effects.
#[test]
fn linearization_retry_rolls_back_partial_member_effects() {
    fn root(task: &mut TaskBuilder) {
        // Declaration order: gather (gold += 1) then gate (needs gold == 0).
        // The gate fails in linearization 0; the retry must see the ORIGINAL
        // gold (0), not the post-gather value (1) — and re-run gather.
        task.branch().any_order((gather, gate));
    }
    fn gather(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn gate(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 == 0)
            .effect(|key: &mut Key| key.0 = true);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).set(Gold(0)).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = plan_of(&mut planner, root, &state);
    assert_eq!(
        plan.task_names(&domain),
        ["gate", "gather"],
        "the reversed linearization was retried from rolled-back state"
    );
    // Both members ran exactly once across both attempts.
    assert_eq!(plan.len(), 2);
}

/// The `then`/`subtask` kinds build distinct constraint edges: a `subtask`
/// member after a `then` member is constrained after it, so a single-
/// linearization branch whose order equals the declaration order normalizes
/// to the total-order fast path.
#[test]
fn then_before_subtask_normalizes_to_total_when_fully_constrained() {
    fn root(task: &mut TaskBuilder) {
        let mut b = task.branch();
        b.then(gather_wood);
        b.subtask(gather_stone);
    }
    fn gather_wood(task: &mut TaskBuilder) {
        task.effect(|wood: &mut Wood| wood.0 += 1);
    }
    fn gather_stone(task: &mut TaskBuilder) {
        task.effect(|stone: &mut Stone| stone.0 += 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let Task::Compound(c) = domain.get_task("root").unwrap() else {
        panic!("root is compound");
    };
    assert!(
        matches!(c.methods[0].order, SubtaskOrder::Total),
        "a single linearization equal to the declaration order normalizes to Total"
    );
}
