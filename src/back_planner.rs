//! Backward (goal-state) planner.
//!
//! Forward planning (see [`crate::planner`]) starts from a task and decomposes
//! toward a fixed goal encoded in the task graph. Back-planning instead starts
//! from a goal task — a list of desired [`Effect`](crate::tasks::Effect)s —
//! and greedily builds a
//! plan that, when executed from the initial state, makes every goal effect
//! true.
//!
//! # Algorithm
//!
//! 1. Register every goal slot (the components each goal
//!    [`Effect`](crate::tasks::Effect) writes).
//! 2. Repeatedly pick the candidate covering the most currently-needed slots
//!    and whose preconditions hold against the simulated state, then commit
//!    it:
//!    - a **primitive** task contributes itself;
//!    - a **compound** task contributes one method's whole subtask sequence,
//!      chosen by the same coverage heuristic and expanded recursively (so
//!      compound tasks participate in reverse chaining through their
//!      per-method *guaranteed* writes — the under-approximation that every
//!      refinement of the method writes those slots).
//! 3. Applied effects satisfy their slots; stop when all goal slots are
//!    satisfied. If no candidate expands, return [`HtnError::NoPlan`].
//!
//! This is deliberately **greedy** (a cheap stand-in for a full backward
//! search): candidates are tried best-first with state restoration, but the
//! chosen method's subtasks are expanded without backtracking. For full
//! correctness on domains with mutually-dependent goals, callers can fall
//! back to forward planning.

use std::collections::HashSet;

use crate::domain::HtnDomain;
use crate::domain::Task;
use crate::error::{HtnError, HtnResult};
use crate::planner::Plan;
use crate::state::PlanState;
use crate::tasks::GoalFn;

/// One candidate commitment: a primitive task, or a compound task via one of
/// its methods (whose whole subtask sequence would be committed).
#[derive(Clone, Copy, Debug)]
enum Candidate {
    Primitive(usize),
    Compound(usize, usize),
}

/// Nesting cap for compound expansion (recursive domains would otherwise
/// recurse forever — the greedy expansion has no backtracking to unwind).
const MAX_EXPANSION_DEPTH: usize = 64;

/// The default total-primitive-steps budget of a back-planning run (bounds
/// pathological compound chains; configurable via
/// [`BackPlanner::with_budget`]).
pub const DEFAULT_BACK_PLANNING_BUDGET: usize = 200;

/// A backward planner that reaches a named goal task's effects.
pub struct BackPlanner<'a> {
    domain: &'a HtnDomain,
    budget: usize,
}

impl<'a> BackPlanner<'a> {
    /// Create a back-planner over `domain` with the default expansion budget
    /// ([`DEFAULT_BACK_PLANNING_BUDGET`] primitive steps).
    pub fn new(domain: &'a HtnDomain) -> Self {
        Self {
            domain,
            budget: DEFAULT_BACK_PLANNING_BUDGET,
        }
    }

    /// Set the total-primitive-steps budget of a planning run (CDDA-scale
    /// domains with long dependency chains legitimately need more than the
    /// default). Exhausting the budget is reported as
    /// [`HtnError::NoPlan`] — the same error as a genuine dead end; the
    /// greedy expansion has no partial-prefix value to return.
    #[must_use]
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// Plan from `initial_state` toward the effects of the goal function
    /// `goal` (passed by value; resolved by its `TypeId` through the baked
    /// type index — names are display-only).
    ///
    /// `initial_state` is only read: the planner works on its own clone of the
    /// scratchpad. Returns a [`Plan`] of primitive task names in execution
    /// order. The plan's MTR is empty (MTR is a forward-only concept).
    pub fn plan<F: GoalFn>(&mut self, goal: F, initial_state: &PlanState) -> HtnResult<Plan> {
        let Some(goal_task) = self.domain.goal(goal) else {
            return Err(HtnError::UnregisteredTask {
                type_name: std::any::type_name::<F>().to_string(),
            });
        };
        if goal_task.effects.is_empty() {
            return Err(HtnError::NoPlan);
        }

        let mut state = initial_state.clone();
        let mut needed: HashSet<usize> = goal_task.write_slots().collect();

        let mut steps: Vec<u32> = Vec::new();
        // Total primitive steps the plan may contain (bounds pathological
        // compound chains; the old loop count, widened to per-step cost).
        let mut budget: usize = self.budget;

        while !needed.is_empty() {
            if budget == 0 {
                return Err(HtnError::NoPlan);
            }
            // Try candidates best-first; a failed expansion (a nested task
            // whose preconditions do not hold) is rolled back and the next
            // candidate tried. Nothing expandable: plateau.
            let candidates = self.rank_candidates(&needed, &state);
            let mut picked = false;
            for (_, cand) in candidates {
                let steps_len = steps.len();
                let snapshot = state.clone();
                let needed_before = needed.clone();
                if self.expand(&cand, &mut needed, &mut state, &mut steps, &mut budget, 0) {
                    picked = true;
                    break;
                }
                state.copy_from(&snapshot);
                needed = needed_before;
                steps.truncate(steps_len);
            }
            if !picked {
                return Err(HtnError::NoPlan);
            }
        }

        Ok(Plan::compiled(
            steps,
            Vec::new(),
            // Reverse chaining runs to completion or errors — never a
            // truncated prefix.
            crate::planner::PlanStatus::Complete,
            // Pause markers shape forward plans only; backward chaining has
            // no method commitments to truncate.
            None,
        ))
    }

    /// Rank every applicable candidate by how many currently-needed slots it
    /// covers: primitives by their guaranteed writes, compound methods by
    /// their per-method guaranteed writes (under-approximation). Sorted
    /// best-first; ties break by task index, then method index.
    fn rank_candidates(
        &self,
        needed: &HashSet<usize>,
        state: &PlanState,
    ) -> Vec<(usize, Candidate)> {
        let mut out: Vec<(usize, Candidate)> = Vec::new();
        for (i, task) in self.domain.tasks.iter().enumerate() {
            match task {
                Task::Primitive(p) => {
                    if !p.preconditions_met(state) {
                        continue;
                    }
                    let score = p.guaranteed_slots().filter(|w| needed.contains(w)).count();
                    if score > 0 {
                        out.push((score, Candidate::Primitive(i)));
                    }
                }
                Task::Compound(c) => {
                    // Only terminating compounds can contribute a finite
                    // refinement.
                    if !self.domain.summaries[i].terminating {
                        continue;
                    }
                    for (mi, m) in c.methods.iter().enumerate() {
                        if !m.applicable(state) {
                            continue;
                        }
                        let score = m
                            .guaranteed_writes
                            .indices()
                            .filter(|w| needed.contains(w))
                            .count();
                        if score > 0 {
                            out.push((score, Candidate::Compound(i, mi)));
                        }
                    }
                }
                Task::Goal(_) => {}
            }
        }
        out.sort_by(|(sa, a), (sb, b)| {
            sb.cmp(sa)
                .then_with(|| (task_of(a), method_of(a)).cmp(&(task_of(b), method_of(b))))
        });
        out
    }

    /// Commit a candidate: append its primitive sequence to `steps` in
    /// execution order, applying each step's effects to `state` and clearing
    /// the slots they satisfy from `needed`. Compound candidates expand their
    /// method's whole subtask sequence, choosing nested methods by the same
    /// coverage heuristic (greedy — no backtracking inside an expansion).
    /// Returns `false` (with garbage in `steps`/`state`) when any committed
    /// step's preconditions fail or the budget runs out; the caller restores.
    fn expand(
        &self,
        cand: &Candidate,
        needed: &mut HashSet<usize>,
        state: &mut PlanState,
        steps: &mut Vec<u32>,
        budget: &mut usize,
        depth: usize,
    ) -> bool {
        if depth > MAX_EXPANSION_DEPTH {
            return false;
        }
        match *cand {
            Candidate::Primitive(i) => self.commit_primitive(i, needed, state, steps, budget),
            Candidate::Compound(ci, mi) => {
                let Task::Compound(c) = &self.domain.tasks[ci] else {
                    return false;
                };
                let Some(m) = c.methods.get(mi) else {
                    return false;
                };
                for &sub in &m.subtasks {
                    match &self.domain.tasks[sub as usize] {
                        Task::Primitive(_) => {
                            if !self.commit_primitive(sub as usize, needed, state, steps, budget) {
                                return false;
                            }
                        }
                        Task::Compound(inner) => {
                            if !self.domain.summaries[sub as usize].terminating {
                                return false;
                            }
                            // Nested method choice: most needed-slot coverage
                            // first (the sequence is mandatory, so a method
                            // covering nothing still runs — first applicable).
                            let mut best: Option<(usize, usize)> = None;
                            for (mi2, m2) in inner.methods.iter().enumerate() {
                                if !m2.applicable(state) {
                                    continue;
                                }
                                let score = m2
                                    .guaranteed_writes
                                    .indices()
                                    .filter(|w| needed.contains(w))
                                    .count();
                                if best.is_none_or(|(bs, _)| score > bs) {
                                    best = Some((score, mi2));
                                }
                            }
                            let Some((_, mi2)) = best else {
                                return false;
                            };
                            if !self.expand(
                                &Candidate::Compound(sub as usize, mi2),
                                needed,
                                state,
                                steps,
                                budget,
                                depth + 1,
                            ) {
                                return false;
                            }
                        }
                        Task::Goal(_) => return false,
                    }
                }
                true
            }
        }
    }

    /// Append one primitive: preconditions must hold on the simulated state,
    /// then its effects are applied and the slots they satisfy leave `needed`.
    fn commit_primitive(
        &self,
        idx: usize,
        needed: &mut HashSet<usize>,
        state: &mut PlanState,
        steps: &mut Vec<u32>,
        budget: &mut usize,
    ) -> bool {
        let Task::Primitive(p) = &self.domain.tasks[idx] else {
            return false;
        };
        if !p.preconditions_met(state) {
            return false;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        p.apply_effects(state);
        for e in &p.effects {
            for &w in e.writes() {
                needed.remove(&w);
            }
        }
        steps.push(idx as u32);
        true
    }
}

/// Accessors for candidate tie-breaking.
fn task_of(c: &Candidate) -> usize {
    match c {
        Candidate::Primitive(i) | Candidate::Compound(i, _) => *i,
    }
}

fn method_of(c: &Candidate) -> usize {
    match c {
        Candidate::Primitive(_) => 0,
        Candidate::Compound(_, m) => *m,
    }
}
