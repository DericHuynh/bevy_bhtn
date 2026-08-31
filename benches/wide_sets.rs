//! Wide-subtask benchmark for `bevy_bhtn`: the cost of unordered member sets
//! (`any_order`) versus the two alternatives a CDDA-style game would choose
//! between, plus the wide-set cap behavior.
//!
//! All groups run the **plan → execute-one-step → replan cycle** per measured
//! iteration (state reset per iteration) and call the planner directly,
//! isolating scheduling cost from ECS overhead.
//!
//! - **`set_vs_chain_8`** — the same 8 unconditional primitives declared as an
//!   `any_order` set vs a `then` chain. Order 0 (declaration order) succeeds
//!   for both, so this measures the pure machinery overhead: the set's
//!   commitment pushes the baked first topological order, and its look-ahead
//!   sweep runs in set semantics (pre-unioned optimistic writes, no
//!   sequential effect application, no state clone) — often *cheaper* than
//!   the chain's exact sequential sweep.
//! - **`set_retry_vs_chain_2`** — a 2-member set whose declaration order
//!   fails (a locked door before the key) vs a 2-member chain that succeeds
//!   first try: the cost of one linearization retry (rollback to commitment
//!   state + re-commit with the next topological order).
//! - **`recursion_vs_set_fetch`** — fetching 16 items modeled the two ways a
//!   CDDA-style game would: a **recursive** task over a counter component
//!   (data-driven, no arity limit, one commitment per item) vs an 8-member
//!   unordered set. This is the modeling guidance in numbers: authored
//!   unordered sets are for small independent member sets; data-driven
//!   repetition (loot N items, craft with M ingredients) belongs in
//!   recursion over components.
//! - **`wide_set_buried_dependency`** — an 8-member unconstrained set whose
//!   declaration order fails because member 0 needs member 1: the
//!   lexicographic linearization enumeration (cap 64) cannot reach an order
//!   where member 1 leads (7! = 5040 orders share the member-0 prefix), so
//!   the method fails and the fallback plans. The 4-member version (4! = 24 ≤
//!   64) succeeds. This pins the documented envelope: unordered sets with
//!   cross-member dependencies are reliable up to ~4 members; wider sets
//!   belong in recursion.

mod common;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::TaskBuilder;
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Key(bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Needs(i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Touched(i32);

fn step_a(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 1);
}
fn step_b(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 2);
}
fn step_c(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 4);
}
fn step_d(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 8);
}
fn step_e(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 16);
}
fn step_f(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 32);
}
fn step_g(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 64);
}
fn step_h(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 += 128);
}
/// Unconditional when the key is present; the buried dependency's dependent.
fn locked(task: &mut TaskBuilder) {
    task.precondition(|k: &Key| k.0)
        .effect(|t: &mut Touched| t.0 += 1);
}
/// The buried dependency's enabler.
fn find_key(task: &mut TaskBuilder) {
    task.effect(|k: &mut Key| k.0 = true);
}
fn fallback(task: &mut TaskBuilder) {
    task.effect(|t: &mut Touched| t.0 = -1);
}
fn fetch_one(task: &mut TaskBuilder) {
    task.precondition(|n: &Needs| n.0 > 0)
        .effect(|n: &mut Needs| n.0 -= 1)
        .effect(|t: &mut Touched| t.0 += 1);
}

/// Plan + execute one step + replan `REPLAN_CYCLES` times per iteration.
fn plan_cycle(domain: &HtnDomain, root: &str, state: &PlanState, planner: &mut HtnPlanner) {
    let mut state = state.clone();
    for _ in 0..REPLAN_CYCLES {
        let plan = planner.plan(root, black_box(&state));
        // Execute one step against the scratchpad (the shared helper needs a
        // World; for direct-planner measurement committing the step's effects
        // to the scratchpad is the equivalent work).
        if let Some(idx) = plan.step_task(0) {
            if let bevy_bhtn::tasks::Task::Primitive(p) = &domain.tasks[idx] {
                p.apply_effects(&mut state);
            }
        }
        black_box(plan.task_names().len());
    }
}

const REPLAN_CYCLES: usize = 1;

fn bench_wide_sets(c: &mut Criterion) {
    // --- set_vs_chain_8 -----------------------------------------------------
    {
        fn chain8(task: &mut TaskBuilder) {
            let mut b = task.branch();
            b.then(step_a).then(step_b).then(step_c).then(step_d);
            b.then(step_e).then(step_f).then(step_g).then(step_h);
        }
        fn set8(task: &mut TaskBuilder) {
            task.branch().any_order((
                step_a, step_b, step_c, step_d, step_e, step_f, step_g, step_h,
            ));
        }
        let d_chain = HtnDomain::from_root(chain8).build().unwrap();
        let d_set = HtnDomain::from_root(set8).build().unwrap();
        let s_chain = PlanState::build(&d_chain.components).finish();
        let s_set = PlanState::build(&d_set.components).finish();

        let mut group = c.benchmark_group("set_vs_chain_8");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, domain, state, root) in [
            ("chain", &d_chain, &s_chain, "chain8"),
            ("any_order", &d_set, &s_set, "set8"),
        ] {
            let mut planner = HtnPlanner::new(domain);
            group.bench_function(label, |b| {
                b.iter(|| plan_cycle(domain, root, state, &mut planner))
            });
        }
        group.finish();
    }

    // --- set_retry_vs_chain_2 -------------------------------------------------
    {
        fn chain2(task: &mut TaskBuilder) {
            task.branch().then(find_key).then(locked);
        }
        fn set2(task: &mut TaskBuilder) {
            // Declaration order fails; the retry reverses it.
            task.branch().any_order((locked, find_key));
            task.branch().then(fallback);
        }
        let d_chain = HtnDomain::from_root(chain2).build().unwrap();
        let d_set = HtnDomain::from_root(set2).build().unwrap();
        let s_chain = PlanState::build(&d_chain.components).finish();
        let s_set = PlanState::build(&d_set.components).finish();

        let mut group = c.benchmark_group("set_retry_vs_chain_2");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, domain, state, root) in [
            ("chain", &d_chain, &s_chain, "chain2"),
            ("one_retry", &d_set, &s_set, "set2"),
        ] {
            let mut planner = HtnPlanner::new(domain);
            group.bench_function(label, |b| {
                b.iter(|| plan_cycle(domain, root, state, &mut planner))
            });
        }
        group.finish();
    }

    // --- recursion_vs_set_fetch ---------------------------------------------
    {
        fn fetch_all(task: &mut TaskBuilder) {
            task.branch().precondition(|n: &Needs| n.0 <= 0);
            task.branch().then(fetch_one).then(fetch_all);
        }
        fn set8(task: &mut TaskBuilder) {
            task.branch().any_order((
                step_a, step_b, step_c, step_d, step_e, step_f, step_g, step_h,
            ));
        }
        let d_rec = HtnDomain::from_root(fetch_all).build().unwrap();
        let d_set = HtnDomain::from_root(set8).build().unwrap();
        let s_rec = PlanState::build(&d_rec.components).set(Needs(16)).finish();
        let s_set = PlanState::build(&d_set.components).finish();

        let mut group = c.benchmark_group("recursion_vs_set_fetch");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, domain, state, root) in [
            ("recursive_16_items", &d_rec, &s_rec, "fetch_all"),
            ("set_8_members", &d_set, &s_set, "set8"),
        ] {
            let mut planner = HtnPlanner::new(domain);
            group.bench_function(label, |b| {
                b.iter(|| plan_cycle(domain, root, state, &mut planner))
            });
        }
        group.finish();
    }

    // --- wide_set_buried_dependency ------------------------------------------
    {
        fn wide8(task: &mut TaskBuilder) {
            task.branch().any_order((
                locked, find_key, step_a, step_b, step_c, step_d, step_e, step_f,
            ));
            task.branch().then(fallback);
        }
        fn wide4(task: &mut TaskBuilder) {
            task.branch().any_order((locked, find_key, step_a, step_b));
            task.branch().then(fallback);
        }
        let d8 = HtnDomain::from_root(wide8).build().unwrap();
        let d4 = HtnDomain::from_root(wide4).build().unwrap();
        let s8 = PlanState::build(&d8.components).finish();
        let s4 = PlanState::build(&d4.components).finish();

        let mut group = c.benchmark_group("wide_set_buried_dependency");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, domain, state, root) in [
            ("members_8_beyond_cap", &d8, &s8, "wide8"),
            ("members_4_within_cap", &d4, &s4, "wide4"),
        ] {
            let mut planner = HtnPlanner::new(domain);
            group.bench_function(label, |b| {
                b.iter(|| plan_cycle(domain, root, state, &mut planner))
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_wide_sets);
criterion_main!(benches);
