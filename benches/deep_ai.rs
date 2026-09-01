//! Deeper, relationship-driven throughput benchmark for `bevy_bhtn`.
//!
//! The miner benchmark measures flat-planning throughput. This one stresses the
//! planner on a **deep** domain (the outpost domain, depth >= 5 with dozens of
//! method options) **and** makes every entity carry a *real Bevy Relationship*
//! (`ServesCache` / `CacheMembers`) instead of a flat struct: each colonist
//! points at a supply-cache entity and its plan scratchpad is seeded from cache
//! resources read through the relationship, so the per-entity AI cost includes
//! a relationship traversal (the exact pattern a squad/inventory system in the
//! game would use).
//!
//! Like the miner bench, this runs through a real Bevy `Schedule` with
//! `Query::par_iter_mut` over the multi-threaded executor at 10k / 50k / 200k
//! entities, plus single-actor overhead cases — and per actor it runs one
//! **complete AI episode**: plan against the cache-seeded working scratchpad,
//! then execute **every step** of the plan (the scratchpad is rebuilt from
//! the colonist's immutable seed every run, so every measured iteration
//! starts from identical states without a reset pass).

mod common;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::HtnDomain;
use bevy_ecs::component::Component;
use bevy_ecs::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use ustr::Ustr;

use common::{
    execute_plan, outpost_domain, outpost_scratch, Ammo, Food, Fuel, Health, Morale, Zone,
};

/// The root task of the outpost domain (the task function's name).
const ROOT: &str = "secure_outpost";

/// The supply cache a colonist draws from. Owned by the *cache* entity; read by
/// the planner through the [`ServesCache`] relationship when seeding a colonist's
/// plan state.
#[derive(Component, Clone, Debug, Default)]
struct CacheResources {
    fuel: i32,
    food: i32,
    ammo: i32,
}

/// Relationship: a colonist (source) is provisioned by a supply-cache entity
/// (target). Reading this is the cross-entity hop in the benchmark.
#[derive(Component)]
#[relationship(relationship_target = CacheMembers)]
struct ServesCache(#[entities] pub Entity);

/// Relationship target stored on the cache: the members that feed from it.
#[derive(Component)]
#[relationship_target(relationship = ServesCache)]
struct CacheMembers(Vec<Entity>);

/// The per-entity planning scratchpad: a dense snapshot of the outpost
/// components the domain's closures read and write.
#[derive(Component, Default)]
struct Scratch(PlanState);

/// The colonist's own (pre-cache) scratchpad seed, immutable across runs.
#[derive(Component, Clone)]
struct ColonistSeed(PlanState);

/// The output of the planner, written back onto each colonist.
#[derive(Component, Debug, Default)]
struct PlanOutput(Vec<Ustr>);

/// Counter written by [`run_ai`].
#[derive(Resource, Default)]
struct AiProcessed(AtomicUsize);

/// Immutable baked domain shared by every AI system run.
#[derive(Resource)]
struct HtnResources {
    domain: HtnDomain,
}

/// The supply caches, keyed by entity, so the AI can read cache resources via the
/// relationship (a real cache/connection store the game would keep).
#[derive(Resource, Default)]
struct CacheStore(HashMap<Entity, CacheResources>);

/// Seed a colonist's scratchpad from its own base components (no cache
/// contribution yet — that is applied per run through the relationship).
fn colonist_seed(domain: &HtnDomain, i: usize) -> PlanState {
    PlanState::build(&domain.components)
        .set(Fuel(1 + (i % 8) as i32))
        .set(Food(1 + (i % 30) as i32))
        .set(Health(50 + (i % 60) as i32))
        .set(Morale(5 + (i % 90) as i32))
        .set(Ammo(1 + (i % 12) as i32))
        .set(Zone::Outside)
        .finish()
}

/// Apply the cache contribution to a working scratchpad (a quarter share of
/// each cache resource, the same split the former monolithic-state bench used).
fn apply_cache(domain: &HtnDomain, state: &mut PlanState, cache: &CacheResources) {
    let reg = &domain.components;
    // Unwrap: every component below is touched by the outpost domain's
    // closures, so it is guaranteed registered.
    state.get_mut::<Fuel>(reg.get::<Fuel>().unwrap()).0 += cache.fuel / 4;
    state.get_mut::<Food>(reg.get::<Food>().unwrap()).0 += cache.food / 4;
    state.get_mut::<Ammo>(reg.get::<Ammo>().unwrap()).0 += cache.ammo / 4;
}

/// The AI system: for each colonist, use its `ServesCache` relationship to find
/// its supply cache, seed a working scratchpad from its immutable seed plus the
/// cache share, then plan and execute the full plan. The seed stays immutable
/// (the cache-seeded scratchpad is ephemeral), so measured iterations need no
/// reset pass.
fn run_ai(
    resources: Res<HtnResources>,
    caches: Res<CacheStore>,
    processed: Res<AiProcessed>,
    mut q: Query<(&ColonistSeed, &ServesCache, &mut Scratch, &mut PlanOutput)>,
) {
    processed.0.store(0, Ordering::Relaxed);
    q.par_iter_mut()
        .for_each(|(seed, serves, mut scratch, mut output)| {
            // The relationship traversal: resolve the cache entity -> resources.
            let cache = caches
                .0
                .get(&serves.0)
                .cloned()
                .unwrap_or(CacheResources::default());
            let domain = &resources.domain;
            scratch.0 = seed.0.clone();
            apply_cache(domain, &mut scratch.0, &cache);
            let mut planner = HtnPlanner::new(domain);
            let planned = planner.plan(ROOT, &scratch.0);
            execute_plan(domain, &mut scratch.0, &planned);
            output.0 = planned.task_names().to_vec();
            processed.0.fetch_add(1, Ordering::Relaxed);
        });
}

/// Spawn `n` colonists, each provisioned by one of `n / SQUAD_MIN` cache
/// entities via a `ServesCache` relationship.
fn spawn_world(n: usize, squad: usize) -> (World, Schedule) {
    let mut res = World::new();
    let mut caches = CacheStore::default();

    // Spawn the supply caches first (one per up-to-`squad` colonists).
    let cache_entities: Vec<Entity> = (0..n.div_ceil(squad))
        .map(|i| {
            let e = res
                .spawn(CacheResources {
                    fuel: 20 + ((i * 7) % 20) as i32,
                    food: 30 + ((i * 5) % 30) as i32,
                    ammo: 10 + ((i * 3) % 25) as i32,
                })
                .id();
            caches
                .0
                .insert(e, res.entity(e).get::<CacheResources>().unwrap().clone());
            e
        })
        .collect();
    res.insert_resource(caches);

    let domain = outpost_domain();
    let mut idx = 0usize;
    res.spawn_batch((0..n).map(|i| {
        let cache = cache_entities[idx / squad];
        idx += 1;
        (
            // Cross-entity relationship: provisioned by `cache`.
            ServesCache(cache),
            ColonistSeed(colonist_seed(&domain, i)),
            // Pre-insert the scratchpad and output components so the AI system
            // writes into them directly in parallel (steady-state: every actor
            // already has a plan).
            Scratch::default(),
            PlanOutput::default(),
        )
    }))
    .count();

    res.insert_resource(HtnResources { domain });
    res.insert_resource(AiProcessed::default());

    let mut schedule = Schedule::default();
    schedule.add_systems(run_ai);
    (res, schedule)
}

pub fn deep_planner(c: &mut Criterion) {
    let domain = outpost_domain();
    let single_state = outpost_scratch(&domain, common::fresh_outpost());

    let mut group = c.benchmark_group("cdda_htn_deep_bevy_ecs");

    // Full-frame throughput over the entity population.
    let squad = 8usize;
    for n in [10_000usize, 50_000, 200_000] {
        let (mut world, mut schedule) = spawn_world(n, squad);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_function(format!("frame_{n}_deep_rel_entities"), |b| {
            b.iter(|| {
                schedule.run(&mut world);
            });
        });
    }

    // Single-actor deep latency: the same work shape as `run_ai` — one plan,
    // then execute the full plan — with the working state reset per iteration.
    // Throughput pinned to 1 element (the group-wide setting still holds 200k
    // from the frame loop above).
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("deep_plan_one_actor_overhead", |b| {
        b.iter(|| {
            let mut state = single_state.clone();
            let mut planner = HtnPlanner::new(&domain);
            let plan = planner.plan(ROOT, &state);
            execute_plan(&domain, &mut state, &plan);
            black_box(&plan);
        });
    });
    // Also a seed with relation resources to keep the deep benchmark honest.
    let seeded_state = {
        let mut state = single_state.clone();
        apply_cache(
            &domain,
            &mut state,
            &CacheResources {
                fuel: 20,
                food: 40,
                ammo: 8,
            },
        );
        state
    };
    group.bench_function("deep_plan_related_latency", |b| {
        b.iter(|| {
            let mut state = seeded_state.clone();
            let mut planner = HtnPlanner::new(&domain);
            let plan = planner.plan(ROOT, &state);
            execute_plan(&domain, &mut state, &plan);
            black_box(&plan);
        });
    });

    group.finish();
}

criterion_group!(benches, deep_planner);
criterion_main!(benches);
