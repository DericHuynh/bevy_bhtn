//! Deeper, relationship-driven throughput benchmark for `cdda_htn`.
//!
//! The miner benchmark measures flat-planning throughput. This one stresses the
//! planner on a **deep** domain (`htn/outpost.htn`, depth >= 5 with dozens
//! of method options) **and** makes every entity carry a *real Bevy
//! Relationship* (`ServesCache` / `CacheMembers`) instead of a flat struct: each
//! colonist points at a supply-cache entity and its plan state is seeded from
//! cache resources read through the relationship, so the per-entity AI cost
//! includes a relationship traversal (the exact pattern a squad/inventory system
//! in the game would use).
//!
//! Like the miner bench, this runs through a real Bevy `Schedule` with
//! `Query::par_iter_mut` over the multi-threaded executor at 10k / 50k / 200k
//! entities, plus single-actor latency cases — and per actor it runs a
//! **plan → execute → replan cycle 10 times** against the cache-seeded working
//! state (the colonist component itself stays immutable, so every measured
//! iteration starts from identical states without a reset pass).

mod common;

use bevy_ecs::component::Component;
use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_reflect::TypeRegistry;
use cdda_htn::planner::HtnPlanner;
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use ustr::Ustr;

use common::{execute_plan_step, register_outpost, OutpostState, OUTPOST_HTN};

/// How many plan → execute → replan cycles each actor runs per measured
/// iteration.
const REPLAN_CYCLES: usize = 10;

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

/// The output of the planner, written back onto each colonist.
#[derive(Component, Debug, Default)]
struct Plan(Vec<Ustr>);

/// Counter written by [`run_ai`].
#[derive(Resource, Default)]
struct AiProcessed(std::sync::atomic::AtomicUsize);

/// Immutable domain + registry + pre-built relationship lookup.
#[derive(Resource)]
struct HtnResources {
    domain: cdda_htn::HtnDomain,
    registry: TypeRegistry,
}

impl HtnResources {
    fn new() -> Self {
        let mut registry = TypeRegistry::default();
        register_outpost(&mut registry);
        let domain = cdda_htn::parse_htn(OUTPOST_HTN).expect("parse outpost HTN");
        Self { domain, registry }
    }
}

/// The supply caches, keyed by entity, so the AI can read cache resources via the
/// relationship (a real cache/connection store the game would keep).
#[derive(Resource, Default)]
struct CacheStore(HashMap<Entity, CacheResources>);

fn build_plan_state(colonist: &OutpostState, cache: &CacheResources) -> OutpostState {
    OutpostState {
        fuel: colonist.fuel + cache.fuel / 4,
        food: colonist.food + cache.food / 4,
        ammo: colonist.ammo + cache.ammo / 4,
        // Intentionally spread out per colony so the deep planner's branches
        // differ across entities (and the plan is never a trivial "done").
        health: colonist.health,
        morale: colonist.morale,
        perimeter: colonist.perimeter,
        reinforced: colonist.reinforced,
        armored: colonist.armored,
        caches: colonist.caches,
        position: colonist.position,
    }
}

/// The AI system: for each colonist, use its `ServesCache` relationship to find
/// its supply cache, seed a plan state from it, then run the 10-cycle
/// plan → execute → replan loop against that working state and write the final
/// plan. The colonist component stays immutable (the cache-seeded state is
/// ephemeral), so measured iterations need no reset pass.
fn run_ai(
    resources: Res<HtnResources>,
    caches: Res<CacheStore>,
    processed: Res<AiProcessed>,
    mut q: Query<(&OutpostState, &ServesCache, &mut Plan)>,
) {
    processed.0.store(0, std::sync::atomic::Ordering::Relaxed);
    q.par_iter_mut().for_each(|(colonist, serves, mut plan)| {
        // The relationship traversal: resolve the cache entity -> resources.
        let cache = caches
            .0
            .get(&serves.0)
            .cloned()
            .unwrap_or(CacheResources::default());
        let mut start = build_plan_state(colonist, &cache);
        let mut planner = HtnPlanner::new(&resources.domain, &resources.registry);
        let mut planned = planner.plan("SecureOutpost", &start);
        for _ in 0..REPLAN_CYCLES {
            common::execute_plan_step(
                &resources.domain,
                &resources.registry,
                start.as_reflect_mut(),
                &planned,
            );
            planned = planner.plan("SecureOutpost", &start);
        }
        plan.0 = planned.task_names().to_vec();
        processed
            .0
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
}

/// Spawn `n` colonists, each provisioned by one of `n / SQUAD_MIN` cache
/// entities via a `ServesCache` relationship.
fn spawn_world(n: usize, squad: usize) -> (World, Schedule) {
    let mut res = World::new();
    res.insert_resource(HtnResources::new());
    res.insert_resource(AiProcessed::default());
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

    let mut idx = 0usize;
    res.spawn_batch((0..n).map(|i| {
        let cache = cache_entities[idx / squad];
        idx += 1;
        (
            OutpostState {
                fuel: 1 + (i % 8) as i32,
                food: 1 + (i % 30) as i32,
                health: 50 + (i % 60) as i32,
                morale: 5 + (i % 90) as i32,
                ammo: 1 + (i % 12) as i32,
                ..Default::default()
            },
            // Cross-entity relationship: provisioned by `cache`.
            ServesCache(cache),
            // Pre-insert the output component so the AI system writes into it
            // directly in parallel (steady-state: every actor already has a plan).
            Plan(Vec::new()),
        )
    }))
    .count();

    let mut schedule = Schedule::default();
    schedule.add_systems(run_ai);
    (res, schedule)
}

pub fn deep_planner(c: &mut Criterion) {
    let single_state = OutpostState::default();

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

    // Single-actor deep latency: the same 10-cycle replan loop, working state
    // rebuilt per iteration.
    let resources = HtnResources::new();
    group.bench_function("deep_plan_one_actor_latency", |b| {
        b.iter(|| {
            let mut start = single_state.clone();
            let mut planner = HtnPlanner::new(&resources.domain, &resources.registry);
            for _ in 0..REPLAN_CYCLES {
                let plan = planner.plan("SecureOutpost", &start);
                execute_plan_step(
                    &resources.domain,
                    &resources.registry,
                    start.as_reflect_mut(),
                    &plan,
                );
                black_box(&plan);
            }
        });
    });
    // Also a seed with relation resources to keep the deep benchmark honest.
    let seeded_cache = CacheResources {
        fuel: 20,
        food: 40,
        ammo: 8,
    };
    let seeded_state = build_plan_state(&single_state, &seeded_cache);
    group.bench_function("deep_plan_related_latency", |b| {
        b.iter(|| {
            let mut start = seeded_state.clone();
            let mut planner = HtnPlanner::new(&resources.domain, &resources.registry);
            for _ in 0..REPLAN_CYCLES {
                let plan = planner.plan("SecureOutpost", &start);
                execute_plan_step(
                    &resources.domain,
                    &resources.registry,
                    start.as_reflect_mut(),
                    &plan,
                );
                black_box(&plan);
            }
        });
    });

    group.finish();
}

criterion_group!(benches, deep_planner);
criterion_main!(benches);
