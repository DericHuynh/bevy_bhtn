//! Backward (goal-state) planner.
//!
//! Forward planning (see [`crate::planner`]) starts from a task and decomposes
//! toward a fixed goal encoded in the task graph. Back-planning instead starts
//! from a `goal_task` — a list of desired [`Effect`]s — and works backwards over
//! the domain's primitive tasks to find a dependency-ordered plan that, when
//! executed from the initial state, makes every goal effect true.
//!
//! # Algorithm
//!
//! 1. Register every goal field (the `field` of each goal [`Effect`]).
//! 2. In reverse (last goal effect processed first — though any order works for
//!    an *unconditional* search), repeatedly pick a primitive whose **effects**
//!    include a `Set*`/`Increment*` on a currently-unfulfilled goal field, greed
//!    choose it, prepend it to the plan, and apply its effects to a working copy
//!    of the state.
//! 3. Stop when all goal fields are satisfied; if the search plateaus, return
//!    [`HtnError::NoPlan`].
//!
//! This is deliberately **greedy reverse chaining** (a cheap stand-in for a full
//! backward search that explores every operator once per step); it prefers
//! operators that make the most progress. For full correctness on domains with
//! mutually-dependent goals, callers can fall back to forward planning.

use std::collections::HashSet;

use bevy_reflect::{Reflect, TypeRegistry};
use ustr::Ustr;

use crate::domain::HtnDomain;
use crate::effects::Effect;
use crate::error::{HtnError, HtnResult};
use crate::planner::Plan;
use crate::tasks::Task;
use crate::HtnState;

/// A backward planner that reaches a named goal task's effects.
pub struct BackPlanner<'a> {
    domain: &'a HtnDomain,
    registry: &'a TypeRegistry,
}

impl<'a> BackPlanner<'a> {
    /// Create a back-planner over `domain` using `registry` for effect
    /// application and condition evaluation.
    pub fn new(domain: &'a HtnDomain, registry: &'a TypeRegistry) -> Self {
        Self { domain, registry }
    }

    /// Plan from `initial_state` toward the effects of the goal task `goal_name`.
    ///
    /// Returns a [`Plan`] of primitive task names in execution order. The plan's
    /// MTR is empty (MTR is a forward-only concept).
    pub fn plan<S: HtnState>(&mut self, goal_name: &str, initial_state: &S) -> HtnResult<Plan> {
        let Some(goal) = self.domain.goal(goal_name) else {
            return Err(HtnError::UnknownTask {
                name: goal_name.to_string(),
            });
        };
        if goal.effects.is_empty() {
            return Err(HtnError::NoPlan);
        }

        let mut state = initial_state.clone();
        let mut needed: HashSet<Ustr> = goal
            .effects
            .iter()
            .map(Effect::field)
            .map(Ustr::from)
            .collect();

        let mut plan: Vec<Ustr> = Vec::new();
        let mut search_limit = 200;

        while !needed.is_empty() && search_limit > 0 {
            search_limit -= 1;
            let erased = state.as_reflect();
            match self.pick_one(&needed, erased) {
                Some(task_name) => {
                    // Apply the chosen task's effects to working state.
                    if let Some(Task::Primitive(p)) = self.domain.get_task(&task_name) {
                        p.apply_effects(state.as_reflect_mut(), self.registry);
                        // The goal field may now be set by an effect, so drop it.
                        // (We approximate "satisfied" by the effect having been
                        // applied; a fully order-robust check would re-verify by
                        // conditions + goal predicates.)
                        for e in p.effects.iter() {
                            if needed.remove(&Ustr::from(e.field())) {
                                break;
                            }
                        }
                    }
                    plan.push(task_name);
                }
                None => {
                    // Nothing can advance a needed field.
                    return Err(HtnError::NoPlan);
                }
            }
        }

        if needed.is_empty() {
            Ok(Plan {
                tasks: plan,
                mtr: crate::planner::Mtr::default(),
            })
        } else {
            Err(HtnError::NoPlan)
        }
    }

    /// Choose a single primitive task whose effects produce a value for a
    /// currently-needed goal field, preferring the one that covers the most
    /// needed fields (a cheap, deterministic heuristic).
    fn pick_one(&self, needed: &HashSet<Ustr>, state: &dyn Reflect) -> Option<Ustr> {
        let mut best: Option<(usize, Ustr)> = None;
        for name in self.domain.primitive_names() {
            let Some(Task::Primitive(p)) = self.domain.get_task(name.as_str()) else {
                continue;
            };
            let produced: HashSet<Ustr> = p
                .effects
                .iter()
                .map(Effect::field)
                .map(Ustr::from)
                .filter(|f| needed.contains(f))
                .collect();
            if produced.is_empty() {
                continue;
            }
            // Only consider tasks whose preconditions currently hold.
            if !p.preconditions.iter().all(|c| c.evaluate(state)) {
                continue;
            }
            let score = produced.len();
            if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                best = Some((score, name));
            }
        }
        best.map(|(_, name)| name)
    }
}
