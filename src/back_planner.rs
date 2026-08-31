//! Backward (goal-state) planner.
//!
//! Forward planning (see [`crate::planner`]) starts from a task and decomposes
//! toward a fixed goal encoded in the task graph. Back-planning instead starts
//! from a goal task — a list of desired [`Effect`]s — and works backwards over
//! the domain's primitive tasks to find a dependency-ordered plan that, when
//! executed from the initial state, makes every goal effect true.
//!
//! # Algorithm
//!
//! 1. Register every goal slot (the components each goal [`Effect`] writes).
//! 2. Repeatedly pick a primitive whose **effects** write a currently-needed
//!    slot, greedily choose the one covering the most needed slots, prepend it
//!    to the plan, and apply its effects to a working copy of the state.
//! 3. Stop when all goal slots are satisfied; if the search plateaus, return
//!    [`HtnError::NoPlan`].
//!
//! This is deliberately **greedy reverse chaining** (a cheap stand-in for a
//! full backward search that explores every operator once per step); it
//! prefers operators that make the most progress. For full correctness on
//! domains with mutually-dependent goals, callers can fall back to forward
//! planning.

use std::collections::HashSet;

use crate::domain::HtnDomain;
use crate::error::{HtnError, HtnResult};
use crate::planner::Plan;
use crate::state::PlanState;
use crate::tasks::Task;

/// A backward planner that reaches a named goal task's effects.
pub struct BackPlanner<'a> {
    domain: &'a HtnDomain,
}

impl<'a> BackPlanner<'a> {
    /// Create a back-planner over `domain`.
    pub fn new(domain: &'a HtnDomain) -> Self {
        Self { domain }
    }

    /// Plan from `initial_state` toward the effects of the goal task `goal_name`.
    ///
    /// `initial_state` is only read: the planner works on its own clone of the
    /// scratchpad. Returns a [`Plan`] of primitive task names in execution
    /// order. The plan's MTR is empty (MTR is a forward-only concept).
    pub fn plan(&mut self, goal_name: &str, initial_state: &PlanState) -> HtnResult<Plan> {
        let Some(goal) = self.domain.goal(goal_name) else {
            return Err(HtnError::UnknownTask {
                name: goal_name.to_string(),
            });
        };
        if goal.effects.is_empty() {
            return Err(HtnError::NoPlan);
        }

        let mut state = initial_state.clone();
        let mut needed: HashSet<usize> = goal.write_slots().collect();

        let mut steps: Vec<u32> = Vec::new();
        let mut search_limit = 200;

        while !needed.is_empty() && search_limit > 0 {
            search_limit -= 1;
            match self.pick_one(&needed, &state) {
                Some(task_idx) => {
                    // Apply the chosen task's effects to the working state.
                    if let Task::Primitive(p) = &self.domain.tasks[task_idx] {
                        p.apply_effects(&mut state);
                        // The goal slots may now be written, so drop them.
                        // (We approximate "satisfied" by the effect having
                        // been applied; a fully order-robust check would
                        // re-verify by preconditions + goal predicates.)
                        for e in &p.effects {
                            let mut removed = false;
                            for &w in e.writes() {
                                if needed.remove(&w) {
                                    removed = true;
                                }
                            }
                            if removed {
                                break;
                            }
                        }
                    }
                    steps.push(task_idx as u32);
                }
                None => {
                    // Nothing can advance a needed slot.
                    return Err(HtnError::NoPlan);
                }
            }
        }

        if needed.is_empty() {
            Ok(Plan {
                names: steps
                    .iter()
                    .map(|&s| self.domain.tasks[s as usize].name().into())
                    .collect(),
                steps,
                mtr: crate::planner::Mtr::default(),
            })
        } else {
            Err(HtnError::NoPlan)
        }
    }

    /// Choose a single primitive task whose effects produce a value for a
    /// currently-needed goal slot, preferring the one that covers the most
    /// needed slots (a cheap, deterministic heuristic).
    fn pick_one(&self, needed: &HashSet<usize>, state: &PlanState) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (i, task) in self.domain.tasks.iter().enumerate() {
            let Task::Primitive(p) = task else {
                continue;
            };
            let produced: HashSet<usize> = p
                .guaranteed_slots()
                .filter(|w| needed.contains(w))
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
                best = Some((score, i));
            }
        }
        best.map(|(_, i)| i)
    }
}
