//! Throughput benchmark for the `bevy_bhtn` planner, run **through Bevy ECS**.
//!
//! Stresses the planner the way a real CDDA frame would: a population of AI
//! actors living as **real Bevy entities**, each carrying the shared miner
//! domain (the canonical HTN miner example) as a dense [`PlanState`]
//! scratchpad component. A Bevy system iterates the entity query every "tick"
//! and, per entity, runs a **plan → execute → replan cycle 10 times**: the
//! plan's effects are applied to the actor's scratchpad (via
//! [`common::execute_plan_step`], the same execution semantics the integration
//! tests pin), the mutated state is re-planned, and the final plan is written
//! back into the actor's `PlanOutput` component. States are reset to their
//! spawn-time seed at the start of every measured iteration, so each
//! iteration does identical, deterministic work.
//!
//! This is upstream of the reference examples, which drive only a handful of
//! actors; here we scale to **200k** simultaneous entities.

mod common;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::HtnDomain;
use bevy_ecs::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use ustr::Ustr;

use common::{
    execute_plan_step, miner_domain, miner_scratch, Energy, Gold, HasMetal, HasOre, Hunger,
    Location,
};

/// How many plan → execute → replan cycles each actor runs per measured
/// iteration (the mid-execution replanning loop a real AI tick performs).
const REPLAN_CYCLES: usize = 1;

/// The root task of the miner domain (the task function's name).
const ROOT: &str = "earn_gold";

/// The per-entity planning scratchpad: a dense snapshot of the miner
/// components the domain's closures read and write. [`PlanState`] is
/// `Clone + Default + Send + Sync`, so it lives directly on the entity.
#[derive(Component, Default)]
struct Scratch(PlanState);

/// The spawn-time scratchpad each entity replays on reset.
#[derive(Component, Clone)]
struct MinerSeed(PlanState);

/// The output of the planner, written back onto each entity.
#[derive(Component, Debug, Default)]
struct PlanOutput(Vec<Ustr>);

/// Running count of actors planned so far, written by [`run_ai`]. Using a single
/// atomics resource keeps the `par_iter_mut` closure free of commands; the
/// planner work per entity dominates the (rare) cross-thread contention on this
/// counter, so a thread-local accumulation pass is not worth its setup cost.
#[derive(Resource, Default)]
struct AiProcessed(AtomicUsize);

/// Immutable baked domain shared by every AI system run.
#[derive(Resource)]
struct HtnResources {
    domain: HtnDomain,
}

/// The AI system: for every miner entity, run the replan cycle over its
/// scratchpad and write the final plan into its `PlanOutput` component.
///
/// Runs the query in **parallel** via [`Query::par_iter_mut`], the per-frame AI
/// cost. Each closure builds its own [`HtnPlanner`] (an immutable `domain`
/// view, no shared mutable state), so the population plans concurrently; the
/// `par_iter_mut` batch is scheduled across the multi-threaded
/// `ComputeTaskPool`.
fn run_ai(
    resources: Res<HtnResources>,
    processed: Res<AiProcessed>,
    mut q: Query<(&mut Scratch, &mut PlanOutput)>,
) {
    processed.0.store(0, Ordering::Relaxed);
    q.par_iter_mut().for_each(|(mut scratch, mut output)| {
        let domain = &resources.domain;
        let mut planner = HtnPlanner::new(domain);
        let mut planned = planner.plan(ROOT, &scratch.0);
        for _ in 0..REPLAN_CYCLES {
            execute_plan_step(domain, &mut scratch.0, &planned);
            planned = planner.plan(ROOT, &scratch.0);
        }
        output.0 = planned.task_names().to_vec();
        processed.0.fetch_add(1, Ordering::Relaxed);
    });
}

/// Reset every actor's scratchpad to its spawn-time seed. Runs at the head of
/// the schedule so each measured iteration starts from identical states (the
/// replan cycle mutates them).
fn reset_states(mut q: Query<(&mut Scratch, &MinerSeed)>) {
    q.par_iter_mut().for_each(|(mut scratch, seed)| {
        scratch.0 = seed.0.clone();
    });
}

/// The miner components as a **concrete** tuple. `common::miner_components`
/// returns an opaque `impl Bundle`, which bevy 0.18 cannot spawn through (the
/// opaque's `DynamicBundle::Effect` bound does not propagate), so the bench
/// reads the same seed values back out of the fixture's `miner_scratch` — the
/// formulas stay in `benches/common/mod.rs` as the single source of truth.
fn miner_bundle(
    domain: &HtnDomain,
    state: &PlanState,
) -> (Gold, HasOre, HasMetal, Energy, Hunger, Location) {
    let reg = &domain.components;
    (
        state.get::<Gold>(reg.get::<Gold>().unwrap()).clone(),
        state.get::<HasOre>(reg.get::<HasOre>().unwrap()).clone(),
        state
            .get::<HasMetal>(reg.get::<HasMetal>().unwrap())
            .clone(),
        state.get::<Energy>(reg.get::<Energy>().unwrap()).clone(),
        state.get::<Hunger>(reg.get::<Hunger>().unwrap()).clone(),
        state
            .get::<Location>(reg.get::<Location>().unwrap())
            .clone(),
    )
}

/// Spawn `n` miner entities with varied components (plus their reset seeds and
/// scratchpads), then return a `(world, schedule)` ready to run the AI system —
/// mirroring how a production Bevy app runs it via the multi-threaded
/// [`Schedule`] executor (which initializes the `ComputeTaskPool`).
fn spawn_world(n: usize) -> (World, Schedule) {
    let domain = miner_domain();
    let mut res = World::new();
    let ids: Vec<Entity> = res
        .spawn_batch((0..n).map(|i| {
            let seed = miner_scratch(&domain, i);
            (miner_bundle(&domain, &seed), MinerSeed(seed))
        }))
        .collect();
    for e in ids {
        res.entity_mut(e).insert((
            // Pre-insert the scratchpad and output components so the AI system
            // can write into them directly in parallel (steady-state: every
            // actor has a plan).
            Scratch::default(),
            PlanOutput::default(),
        ));
    }
    res.insert_resource(HtnResources { domain });
    res.insert_resource(AiProcessed::default());

    let mut schedule = Schedule::default();
    // Explicit chain: bevy_ecs 0.18 tuples no longer imply ordering, and the
    // reset must complete before the AI system reads the scratchpads.
    schedule.add_systems((reset_states, run_ai).chain());
    (res, schedule)
}

pub fn miner_planner(c: &mut Criterion) {
    let domain = miner_domain();
    let single_state = miner_scratch(&domain, 0);

    let mut group = c.benchmark_group("cdda_htn_bevy_ecs");

    // Full-frame throughput: run the AI system over the whole entity population
    // through a Bevy `Schedule`, exactly as production does.
    for n in [10_000usize, 50_000, 200_000] {
        let (mut world, mut schedule) = spawn_world(n);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_function(format!("frame_{n}_miner_entities"), |b| {
            b.iter(|| {
                schedule.run(&mut world);
                let processed = world.resource::<AiProcessed>().0.load(Ordering::Relaxed);
                warn_if_processed(processed, n);
            });
        });
    }

    // Single-actor latency done straight through the planner (no ECS
    // overhead): the same 10-cycle replan loop, state reset per iteration.
    group.bench_function("plan_one_actor_latency", |b| {
        b.iter(|| {
            let mut state = single_state.clone();
            let mut planner = HtnPlanner::new(&domain);
            for _ in 0..REPLAN_CYCLES {
                let plan = planner.plan(ROOT, &state);
                execute_plan_step(&domain, &mut state, &plan);
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
