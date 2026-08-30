//! Throughput benchmark for the `cdda_htn` planner, run **through Bevy ECS**.
//!
//! Stresses the planner the way a real CDDA frame would: a population of AI
//! actors living as **real Bevy entities**, each carrying the shared
//! `htn/miner.htn` domain (the canonical `bevy_htn` miner example). A Bevy
//! system iterates the entity query every "tick" and, per entity, runs a
//! **plan → execute → replan cycle 10 times**: the plan's effects are applied
//! to the actor's state (via [`common::execute_plan_step`], the same execution
//! semantics the integration tests pin), the mutated state is re-planned, and
//! the final plan is written back into the `Plan` component. States are reset
//! to their spawn-time seed at the start of every measured iteration, so each
//! iteration does identical, deterministic work.
//!
//! This is upstream of the reference `bevy_htn` example, which drives only a
//! handful of `Dude`s; here we scale to **200k** simultaneous entities.

mod common;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;
use bevy_ecs::system::Res;
use bevy_reflect::Reflect;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use common::{
    execute_plan_step, initial_miner, register_miner, MinerState, PlanComponent, MINER_HTN,
};

/// How many plan → execute → replan cycles each actor runs per measured
/// iteration (the mid-execution replanning loop a real AI tick performs).
const REPLAN_CYCLES: usize = 10;

/// The AI system: for every miner entity, run the replan cycle and write the
/// final plan into its `Plan` component.
///
/// Runs the query in **parallel** via [`Query::par_iter_mut`], the per-frame AI
/// cost. Each closure builds its own [`HtnPlanner`] (an immutable
/// `domain + registry` view, no shared mutable state), so the population plans
/// concurrently; the `par_iter_mut` batch is scheduled across the
/// multi-threaded `ComputeTaskPool`.
fn run_ai(
    resources: Res<HtnResources>,
    mut q: Query<(&mut MinerState, &mut PlanComponent)>,
    processed: Res<AiProcessed>,
) {
    processed.0.store(0, std::sync::atomic::Ordering::Relaxed);
    q.par_iter_mut().for_each(|(mut state, mut plan)| {
        let mut planner = HtnPlanner::new(&resources.domain, &resources.registry);
        let mut planned = planner.plan("EarnGold", &*state);
        for _ in 0..REPLAN_CYCLES {
            common::execute_plan_step(
                &resources.domain,
                &resources.registry,
                state.as_reflect_mut(),
                &planned,
            );
            planned = planner.plan("EarnGold", &*state);
        }
        plan.0 = planned.task_names().to_vec();
        processed
            .0
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
}

/// Reset every actor's state to its spawn-time seed. Runs at the head of the
/// schedule so each measured iteration starts from identical states (the
/// replan cycle mutates them).
fn reset_states(mut q: Query<(&mut MinerState, &MinerSeed)>) {
    q.par_iter_mut()
        .for_each(|(mut state, seed)| *state = seed.0.clone());
}

/// The spawn-time state each entity replays on reset.
#[derive(Component, Clone)]
struct MinerSeed(MinerState);

/// Running count of actors planned so far, written by [`run_ai`]. Using a single
/// atomics resource keeps the `par_iter_mut` closure free of commands; the
/// planner work per entity dominates the (rare) cross-thread contention on this
/// counter, so a thread-local accumulation pass is not worth its setup cost.
#[derive(Resource, Default)]
struct AiProcessed(std::sync::atomic::AtomicUsize);

/// Immutable domain+registry shared by every AI system run. Both derive
/// `Resource` and are registered once.
#[derive(Resource)]
struct HtnResources {
    domain: HtnDomain,
    registry: bevy_reflect::TypeRegistry,
}

impl HtnResources {
    fn new() -> Self {
        let mut registry = bevy_reflect::TypeRegistry::default();
        register_miner(&mut registry);
        let domain = bevy_bhtn::parse_htn(MINER_HTN).expect("parse miner HTN");
        Self { domain, registry }
    }
}

/// Spawn `n` miner entities with varied state (plus their reset seeds), then
/// return a `(world, schedule)` ready to run the AI system — mirroring how a
/// production Bevy app runs it via the multi-threaded [`Schedule`] executor
/// (which initializes the `ComputeTaskPool`).
fn spawn_world(n: usize) -> (World, Schedule) {
    let mut res = World::new();
    res.insert_resource(HtnResources::new());
    res.insert_resource(AiProcessed::default());
    res.spawn_batch((0..n).map(|i| {
        let initial = initial_miner(i);
        (
            initial.clone(),
            MinerSeed(initial),
            // Pre-insert the output component so the AI system can write into
            // it directly in parallel (steady-state: every actor has a plan).
            PlanComponent(Vec::new()),
        )
    }))
    .count();

    let mut schedule = Schedule::default();
    schedule.add_systems((reset_states, run_ai));
    (res, schedule)
}

pub fn miner_planner(c: &mut Criterion) {
    let single_state = common::initial_miner(0);

    let mut group = c.benchmark_group("cdda_htn_bevy_ecs");

    // Full-frame throughput: run the AI system over the whole entity population
    // through a Bevy `Schedule`, exactly as production does.
    for n in [10_000usize, 50_000, 200_000] {
        let (mut world, mut schedule) = spawn_world(n);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_function(format!("frame_{n}_miner_entities"), |b| {
            b.iter(|| {
                schedule.run(&mut world);
                let processed = world
                    .resource::<AiProcessed>()
                    .0
                    .load(std::sync::atomic::Ordering::Relaxed);
                warn_if_processed(processed, n);
            });
        });
    }

    // Single-actor latency done straight through the planner (no ECS
    // overhead): the same 10-cycle replan loop, state reset per iteration.
    let resources = HtnResources::new();
    group.bench_function("plan_one_actor_latency", |b| {
        b.iter(|| {
            let mut state = single_state.clone();
            let mut planner = HtnPlanner::new(&resources.domain, &resources.registry);
            for _ in 0..REPLAN_CYCLES {
                let plan = planner.plan("EarnGold", &state);
                execute_plan_step(
                    &resources.domain,
                    &resources.registry,
                    state.as_reflect_mut(),
                    &plan,
                );
                black_box(&plan);
            }
        });
    });

    group.finish();
}

#[track_caller]
fn warn_if_processed(actual: usize, expected: usize) {
    assert_eq!(actual, expected, "AI system missed entities");
}

criterion_group!(benches, miner_planner);
criterion_main!(benches);
