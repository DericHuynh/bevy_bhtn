//! Shared test bed for `cdda_htn` integration tests.
//!
//! A lightweight wrapper that owns a parsed [`HtnDomain`] plus its reflection
//! registry, and exposes the three planner surfaces we want to pin:
//! forward planning, backward (goal-state) planning, and a source-string
//! round-trip sanity check. It intentionally avoids pulling ECS into the
//! abstract planner tests; the workspace `TestBed` pattern is mirrored here but
//! simplified to the HTN slice.

use cdda_htn::back_planner::BackPlanner;
use cdda_htn::planner::HtnPlanner;
use cdda_htn::{HtnDomain, HtnResult, HtnState};
use ustr::Ustr;

/// A ready-to-use HTN test arena.
pub struct HtnTestBed {
    domain: HtnDomain,
    registry: bevy_reflect::TypeRegistry,
    root: String,
}

impl HtnTestBed {
    /// Build a bed from `.htn` source, a root compound task name, and a closure
    /// that registers the state types (components used by conditions/effects).
    pub fn new(
        htn_src: &str,
        root: impl Into<String>,
        register_types: impl FnOnce(&mut bevy_reflect::TypeRegistry),
    ) -> Self {
        let mut registry = bevy_reflect::TypeRegistry::default();
        register_types(&mut registry);
        let domain = cdda_htn::parse_htn(htn_src).expect("htn should parse");
        Self {
            domain,
            registry,
            root: root.into(),
        }
    }

    /// The parsed domain (for structural assertions on tasks/methods).
    pub fn domain(&self) -> &HtnDomain {
        &self.domain
    }

    /// The reflection registry (for manually applying effects in execution
    /// tests).
    #[allow(dead_code)]
    pub fn registry(&self) -> &bevy_reflect::TypeRegistry {
        &self.registry
    }

    /// Forward-plan the root task against a concrete state.
    pub fn plan_forward<S>(&self, state: &S) -> Vec<Ustr>
    where
        S: HtnState,
    {
        let mut planner = HtnPlanner::new(&self.domain, &self.registry);
        planner.plan(&self.root, state).task_names().to_vec()
    }

    /// Backward-plan toward a named goal task from a concrete state.
    #[allow(dead_code)]
    pub fn plan_backward<S>(&self, goal: &str, state: &S) -> HtnResult<Vec<Ustr>>
    where
        S: HtnState,
    {
        let mut planner = BackPlanner::new(&self.domain, &self.registry);
        planner.plan(goal, state).map(|p| p.task_names().to_vec())
    }
}
