//! Look-ahead pruning A/B benchmark for `cdda_htn`.
//!
//! Measures what the look-ahead sweep (`src/lookahead.rs`, Olz & Bercher SoCS
//! 2023) buys — and what it costs — by planning the **same domains** (the
//! shared `htn/` fixtures) with [`HtnPlanner::set_lookahead`] on vs off, each
//! running the same **plan → execute → replan cycle 10 times** per measured
//! iteration (state reset per iteration, so every iteration does identical
//! work):
//!
//! - **`exponential_backtrack`** — a doomed method whose dead end is only
//!   detectable via optimistic propagation: a chain of 12 binary-choice gates
//!   where *no* choice writes `gold`, followed by a task requiring
//!   `gold > 1000`. With the sweep, the doomed method is refuted in one pass.
//!   Without it, plain MTR backtracking must enumerate all 2^12 gate
//!   combinations before abandoning the method (sanity limit raised so the
//!   blowup is visible rather than capped).
//! - **`doomed_recursion`** — the sanity-limit case: the first method recurses
//!   without terminating before an impossible tail task. With the sweep the
//!   recursion is never entered; without it the planner burns its whole step
//!   budget (the default 100, i.e. the realistic setting) and returns a
//!   partial plan.
//! - **`outpost_deep`** — the realistic `htn/outpost.htn` domain (depth >= 5,
//!   dozens of methods, genuine in-branch backtracking). This is the honesty
//!   check: on a healthy domain the sweep should be roughly neutral (small
//!   overhead or small gain), not a regression.
//!
//! Unlike `ai_throughput` / `deep_ai`, this bench calls the planner directly
//! instead of through a Bevy `Schedule`: it isolates the *algorithmic* win of
//! pruning from ECS overhead, which is the quantity the on/off comparison is
//! about. The through-ECS path is covered by the other two benches.

mod common;

use bevy_bhtn::parse_htn;
use bevy_bhtn::planner::HtnPlanner;
use bevy_reflect::Reflect;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use common::{
    execute_plan_step, fresh_outpost, register_gate, register_outpost, GateState,
    DOOMED_RECURSION_HTN, EXPONENTIAL_HTN, OUTPOST_HTN,
};

/// How many plan → execute → replan cycles each case runs per measured
/// iteration.
const REPLAN_CYCLES: usize = 10;

fn bench_lookahead(c: &mut Criterion) {
    // --- exponential_backtrack: 2^12 leaf failures vs one sweep ------------
    {
        let domain = parse_htn(EXPONENTIAL_HTN).expect("exponential fixture parses");
        let mut registry = bevy_reflect::TypeRegistry::default();
        register_gate(&mut registry);
        let initial = GateState::default();

        let mut group = c.benchmark_group("exponential_backtrack");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, on) in [("off", false), ("on", true)] {
            let mut planner = HtnPlanner::new(&domain, &registry);
            planner.set_lookahead(on);
            // Raise the budget so the off case shows the real exponential
            // enumeration instead of stopping at the default sanity limit.
            planner.set_sanity_limit(1_000_000);
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut state = initial.clone();
                    for _ in 0..REPLAN_CYCLES {
                        let plan = planner.plan("Root", black_box(&state));
                        execute_plan_step(&domain, &registry, state.as_reflect_mut(), &plan);
                        black_box(plan.task_names().len());
                    }
                })
            });
        }
        group.finish();
    }

    // --- doomed_recursion: sanity-limit burn vs one sweep -------------------
    {
        let domain = parse_htn(DOOMED_RECURSION_HTN).expect("doomed fixture parses");
        let mut registry = bevy_reflect::TypeRegistry::default();
        register_gate(&mut registry);
        let initial = GateState::default();

        let mut group = c.benchmark_group("doomed_recursion");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, on) in [("off", false), ("on", true)] {
            let mut planner = HtnPlanner::new(&domain, &registry);
            planner.set_lookahead(on);
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut state = initial.clone();
                    for _ in 0..REPLAN_CYCLES {
                        let plan = planner.plan("Act", black_box(&state));
                        execute_plan_step(&domain, &registry, state.as_reflect_mut(), &plan);
                        black_box(plan.task_names().len());
                    }
                })
            });
        }
        group.finish();
    }

    // --- outpost_deep: realistic domain, sweep overhead/gain check ----------
    {
        let domain = parse_htn(OUTPOST_HTN).expect("outpost fixture parses");
        let mut registry = bevy_reflect::TypeRegistry::default();
        register_outpost(&mut registry);
        let initial = fresh_outpost();

        let mut group = c.benchmark_group("outpost_deep");
        group.throughput(criterion::Throughput::Elements(REPLAN_CYCLES as u64));
        for (label, on) in [("off", false), ("on", true)] {
            let mut planner = HtnPlanner::new(&domain, &registry);
            planner.set_lookahead(on);
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut state = initial.clone();
                    for _ in 0..REPLAN_CYCLES {
                        let plan = planner.plan("SecureOutpost", black_box(&state));
                        execute_plan_step(&domain, &registry, state.as_reflect_mut(), &plan);
                        black_box(plan.task_names().len());
                    }
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_lookahead);
criterion_main!(benches);
