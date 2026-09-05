//! Look-ahead pruning A/B benchmark for `bevy_bhtn`.
//!
//! Measures what the look-ahead sweep (`src/lookahead.rs`, Olz & Bercher SoCS
//! 2023) buys — and what it costs — by planning the **same domains** (the
//! shared function-defined fixtures) with [`HtnPlanner::set_lookahead_mode`] across
//! the Off / Adaptive / Always modes, each running the same **plan → execute-the-full-plan** episode per
//! measured iteration (state reset per iteration, so every iteration does
//! identical work):
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
//! - **`outpost_deep`** — the realistic outpost domain (depth >= 5, dozens of
//!   methods, genuine in-branch backtracking). This is the honesty check: on a
//!   healthy domain the sweep should be roughly neutral (small overhead or
//!   small gain), not a regression.
//!
//! Unlike `ai_throughput` / `deep_ai`, this bench calls the planner directly
//! instead of through a Bevy `Schedule`: it isolates the *algorithmic* win of
//! pruning from ECS overhead, which is the quantity the on/off comparison is
//! about. The through-ECS path is covered by the other two benches.

mod common;

use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::selection::LookaheadMode;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::TaskFn;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use common::doomed_tasks::act;
use common::gate_tasks::gate_root;
use common::outpost_tasks::secure_outpost;
use common::{
    doomed_recursion_domain, execute_plan, fresh_outpost, gate_domain, outpost_domain,
    outpost_scratch,
};

/// Plan from `root` — `F` is inferred from the function item, so the lookup
/// uses the same `TypeId` the domain recorded at bake time.
fn plan_root<F: TaskFn>(planner: &mut HtnPlanner, _root: F, state: &PlanState) -> Plan {
    planner.plan(_root, state).expect("plan")
}

fn bench_lookahead(c: &mut Criterion) {
    // --- exponential_backtrack: 2^12 leaf failures vs one sweep ------------
    {
        let domain = gate_domain();
        let initial = PlanState::build(&domain.components).finish();

        let mut group = c.benchmark_group("exponential_backtrack");
        group.throughput(criterion::Throughput::Elements(1));
        for (label, mode) in [
            ("off", LookaheadMode::Off),
            ("adaptive", LookaheadMode::Adaptive),
            ("on", LookaheadMode::Always),
        ] {
            let mut planner = HtnPlanner::new(&domain);
            planner.set_lookahead_mode(mode);
            // Raise the budget so the off case shows the real exponential
            // enumeration instead of stopping at the default sanity limit.
            planner.set_sanity_limit(1_000_000);
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut state = initial.clone();
                    let plan = plan_root(&mut planner, gate_root, black_box(&state));
                    execute_plan(&domain, &mut state, &plan);
                    black_box(plan.task_names(&domain).len());
                })
            });
        }
        group.finish();
    }

    // --- doomed_recursion: sanity-limit burn vs one sweep -------------------
    {
        let domain = doomed_recursion_domain();
        let initial = PlanState::build(&domain.components).finish();

        let mut group = c.benchmark_group("doomed_recursion");
        group.throughput(criterion::Throughput::Elements(1));
        for (label, mode) in [
            ("off", LookaheadMode::Off),
            ("adaptive", LookaheadMode::Adaptive),
            ("on", LookaheadMode::Always),
        ] {
            let mut planner = HtnPlanner::new(&domain);
            planner.set_lookahead_mode(mode);
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut state = initial.clone();
                    let plan = plan_root(&mut planner, act, black_box(&state));
                    execute_plan(&domain, &mut state, &plan);
                    black_box(plan.task_names(&domain).len());
                })
            });
        }
        group.finish();
    }

    // --- outpost_deep: realistic domain, sweep overhead/gain check ----------
    {
        let domain = outpost_domain();
        let initial = outpost_scratch(&domain, fresh_outpost());

        let mut group = c.benchmark_group("outpost_deep");
        group.throughput(criterion::Throughput::Elements(1));
        for (label, mode) in [
            ("off", LookaheadMode::Off),
            ("adaptive", LookaheadMode::Adaptive),
            ("on", LookaheadMode::Always),
        ] {
            let mut planner = HtnPlanner::new(&domain);
            planner.set_lookahead_mode(mode);
            group.bench_function(label, |b| {
                b.iter(|| {
                    let mut state = initial.clone();
                    let plan = plan_root(&mut planner, secure_outpost, black_box(&state));
                    execute_plan(&domain, &mut state, &plan);
                    black_box(plan.task_names(&domain).len());
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_lookahead);
criterion_main!(benches);
