//! Shared test bed for `bevy_bhtn` integration tests.
//!
//! A lightweight wrapper that owns a function-defined [`HtnDomain`] and
//! exposes the planner surfaces the tests pin: forward planning, backward
//! (goal-state) planning, and structural domain access. The fixture domains,
//! component types, and execution helpers live in `benches/common/mod.rs` —
//! the single source of truth shared with the benchmarks — and are re-exported
//! here so tests and benches plan *exactly* the same domains.

#![allow(dead_code, unused_imports)]

#[path = "../../benches/common/mod.rs"]
pub mod bench_common;

pub use bench_common::{
    doomed_recursion_domain, doomed_tasks, execute_plan, execute_plan_step, fresh_outpost,
    gate_domain, gate_tasks, high_fuel_outpost, marginal_outpost, miner_domain, miner_scratch,
    miner_tasks, outpost_domain, outpost_scratch, outpost_tasks, Ammo, Armored, Caches, Energy,
    Food, Fuel, GateGold, Gold, HasMetal, HasOre, Health, Hunger, Location, Morale, Noise,
    Perimeter, Position, Reinforced, Zone,
};

use bevy_bhtn::back_planner::BackPlanner;
use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::{GoalBuilder, HtnDomain, HtnResult};
use ustr::Ustr;

/// A ready-to-use HTN test arena.
pub struct HtnTestBed {
    domain: HtnDomain,
    root: String,
}

impl HtnTestBed {
    /// Build a bed from a function-defined domain and its root task name.
    pub fn new(domain: HtnDomain, root: impl Into<String>) -> Self {
        Self {
            domain,
            root: root.into(),
        }
    }

    /// The domain (for structural assertions on tasks/methods).
    pub fn domain(&self) -> &HtnDomain {
        &self.domain
    }

    /// Forward-plan the root task against a scratchpad state.
    pub fn plan_forward(&self, state: &PlanState) -> Vec<Ustr> {
        let mut planner = HtnPlanner::new(&self.domain);
        planner.plan(&self.root, state).task_names().to_vec()
    }

    /// Forward-plan with the look-ahead sweep explicitly on or off (A/B).
    pub fn plan_forward_lookahead(&self, state: &PlanState, lookahead: bool) -> Vec<Ustr> {
        let mut planner = HtnPlanner::new(&self.domain);
        planner.set_lookahead(lookahead);
        planner.plan(&self.root, state).task_names().to_vec()
    }

    /// Backward-plan toward a named goal task from a scratchpad state.
    pub fn plan_backward(&self, goal: &str, state: &PlanState) -> HtnResult<Vec<Ustr>> {
        let mut planner = BackPlanner::new(&self.domain);
        planner.plan(goal, state).map(|p| p.task_names().to_vec())
    }
}
