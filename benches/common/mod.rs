//! Shared bench/test scaffolding for `cdda_htn`.
//!
//! Single source of truth for everything the benchmarks plan: the `.htn`
//! fixtures (top-level [`htn/` directory]), the per-actor state types, the
//! initial-state constructors, and the plan-**execution** helper. The
//! integration tests include this file directly (`#[path =
//! "../benches/common.rs"]`) so the "benchmarks produce correct plans" pins
//! run against *exactly* the code the benches run — same states, same domains,
//! same execution semantics.

// Each bench/test target includes this module and uses only its slice; the
// rest is intentionally shared.
#![allow(dead_code)]

use bevy_ecs::component::Component;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::{Reflect, TypeRegistry};
use cdda_htn::planner::Plan;
use cdda_htn::{HtnDomain, Task};
use ustr::Ustr;

// ---------------------------------------------------------------------------
// Fixtures — the one true `.htn` source, shared by benches and tests
// ---------------------------------------------------------------------------

/// Flat miner domain (the canonical `bevy_htn`/`bevy_dogoap` miner example).
pub const MINER_HTN: &str = include_str!("../../htn/miner.htn");
/// Deep outpost domain (depth >= 5, dozens of methods, genuine backtracking).
pub const OUTPOST_HTN: &str = include_str!("../../htn/outpost.htn");
/// Doomed 12-gate binary-choice chain (look-ahead A/B: exponential case).
pub const EXPONENTIAL_HTN: &str = include_str!("../../htn/exponential.htn");
/// Non-terminating recursion before an impossible tail (look-ahead A/B).
pub const DOOMED_RECURSION_HTN: &str = include_str!("../../htn/doomed_recursion.htn");

// ---------------------------------------------------------------------------
// Miner domain types
// ---------------------------------------------------------------------------

/// The world map location a miner can occupy — mirrors `Location` in the
/// reference miner example.
#[derive(Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Default)]
pub enum Location {
    #[default]
    Outside,
    House,
    Ore,
    Smelter,
    Mushroom,
    Merchant,
}

/// The per-actor AI plan state, stored as a Bevy [`Component`].
#[derive(Reflect, Component, Clone, Debug, Default)]
pub struct MinerState {
    pub gold: i32,
    pub has_ore: bool,
    pub has_metal: bool,
    pub energy: i32,
    pub hunger: i32,
    pub location: Location,
}

/// The output of the planner: written back onto each entity (like the
/// reference `bevy_bae`/`bevy_htn` `Plan` component). Carries just the
/// (interned) task names so the benchmark forces a real component write while
/// exercising the `ustr`-based plan path.
#[derive(Component, Debug, Default, Clone)]
pub struct PlanComponent(pub Vec<Ustr>);

/// The bench's deterministic per-entity initial state (`i`-th entity of the
/// spawn batch).
pub fn initial_miner(i: usize) -> MinerState {
    MinerState {
        gold: (i % 5) as i32,
        has_ore: i % 3 == 0,
        has_metal: i % 7 == 0,
        energy: 80 - (i % 40) as i32,
        hunger: 20 + (i % 60) as i32,
        location: Location::Outside,
    }
}

pub fn register_miner(registry: &mut TypeRegistry) {
    registry.register::<MinerState>();
    registry.register::<Location>();
}

// ---------------------------------------------------------------------------
// Outpost domain types
// ---------------------------------------------------------------------------

/// Where a colonist is physically posted. Mirrors `Zone` in `outpost.htn`.
#[derive(Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Default)]
pub enum Zone {
    #[default]
    Outside,
    Posting,
    Rally,
    Armory,
}

/// Per-colonist plan state — mirrors `OutpostState` in `htn_deep`.
#[derive(Reflect, Component, Clone, Debug, Default)]
pub struct OutpostState {
    pub fuel: i32,
    pub food: i32,
    pub health: i32,
    pub morale: i32,
    pub ammo: i32,
    pub perimeter: bool,
    pub reinforced: bool,
    pub armored: bool,
    pub caches: bool,
    pub position: Zone,
}

/// The bench's fresh-actor state (plenty of everything).
pub fn fresh_outpost() -> OutpostState {
    OutpostState {
        fuel: 5,
        food: 30,
        health: 80,
        morale: 50,
        ammo: 12,
        ..Default::default()
    }
}

/// The bench's marginal-fuel state (the "queue anyway" trap is eligible but
/// its `Drive` leaf fails — genuine in-branch backtracking).
pub fn marginal_outpost() -> OutpostState {
    OutpostState {
        fuel: 3,
        food: 30,
        health: 1,
        ..Default::default()
    }
}

/// The bench's high-fuel state (the direct drive branch).
pub fn high_fuel_outpost() -> OutpostState {
    OutpostState {
        fuel: 10,
        food: 0,
        health: 90,
        ..Default::default()
    }
}

pub fn register_outpost(registry: &mut TypeRegistry) {
    registry.register::<OutpostState>();
    registry.register::<Zone>();
}

// ---------------------------------------------------------------------------
// Look-ahead A/B domain types
// ---------------------------------------------------------------------------

/// Binary-choice gate chain state. `gold` is only ever written by `Strike`;
/// the gates only flip `noise` fields, which is what makes the doomed method's
/// dead end provable by optimistic propagation.
#[derive(Reflect, Default, Clone, Debug)]
#[reflect(Default)]
pub struct GateState {
    pub gold: i32,
    pub noise: bool,
}

pub fn register_gate(registry: &mut TypeRegistry) {
    registry.register::<GateState>();
}

// ---------------------------------------------------------------------------
// Plan execution
// ---------------------------------------------------------------------------

/// Execute a plan: apply each planned primitive task's `effects` to `state`,
/// in order. This is the *execution* semantics the integration tests pin
/// (`effects` only — `expected_effects` are planning-only hopes).
pub fn execute_plan(
    domain: &HtnDomain,
    registry: &TypeRegistry,
    state: &mut dyn Reflect,
    plan: &Plan,
) {
    for name in plan.task_names() {
        apply_task_effects(domain, registry, state, name);
    }
}

/// Execute **one step** of a plan (the first planned task's effects) — the
/// agent-tick semantics the benches' plan → execute → replan cycle uses: each
/// cycle advances the world by one action, so every replan sees real, changed
/// state instead of a goal that was already completed by a full-plan execution.
pub fn execute_plan_step(
    domain: &HtnDomain,
    registry: &TypeRegistry,
    state: &mut dyn Reflect,
    plan: &Plan,
) {
    if let Some(first) = plan.task_names().first() {
        apply_task_effects(domain, registry, state, first);
    }
}

fn apply_task_effects(
    domain: &HtnDomain,
    registry: &TypeRegistry,
    state: &mut dyn Reflect,
    name: &str,
) {
    match domain.get_task(name) {
        Some(Task::Primitive(p)) => {
            for e in &p.effects {
                e.apply_dyn(state, registry);
            }
        }
        _ => {} // defensive: plans are primitive sequences
    }
}
