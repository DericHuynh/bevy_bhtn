//! Look-ahead pruning for the forward planner.
//!
//! Implements the Olz & Bercher (SoCS 2023) look-ahead technique, adapted to
//! this crate's component-state planner: before the planner commits to a
//! method, a linear sweep over that **method's subtask sequence** proves —
//! without decomposing anything — which refinements can possibly succeed:
//!
//! - **Dead-end**: a primitive's preconditions definitely fail, a compound
//!   task has no method that can possibly apply, or the sequence's minimum
//!   yield cannot fit the planner's remaining step budget — all under an
//!   *optimistic* state (the current state plus every write any surviving
//!   refinement might make). No refinement of the chosen method can reach a
//!   plan → the planner skips straight to the next method instead of
//!   recursing into a doomed subtree.
//! - **Inevitable refinement (pin)**: a compound task in the sequence has
//!   exactly one method that can possibly apply → the planner commits to it
//!   when it reaches that task, never trying the methods proven infeasible.
//!   Pins are **occurrence-scoped**: they attach to the specific subtask
//!   instance pushed by the deriving commitment (recursive domains decompose
//!   the same task many times, each under a different state, so a pin is only
//!   valid for the occurrence whose position it was derived at).
//!
//! Soundness rests on the optimistic state being a *superset* approximation of
//! every state reachable by any refinement that reaches a given position:
//! compound tasks contribute their inferred `possible_writes` (components
//! become "unknown"); primitives apply their effects exactly (they must
//! execute for the sequence to continue) — but because effect closures are
//! opaque, every component they write is conservatively marked unknown.
//! Preconditions on known components are evaluated exactly; preconditions
//! touching unknown components are treated as "maybe" and never prune. A
//! dead-end verdict therefore means *every* refinement fails, and a pin means
//! *every other* method fails — under any reachable state.
//!
//! The sweep is scoped to the committed method's own subtasks (not the whole
//! remaining queue): the sequence executes immediately after commitment, so
//! the working state is exact at its start, and the per-commitment cost stays
//! bounded by the method's width instead of the whole plan. Cross-frame dead
//! ends are refuted when their own frame commits.

use crate::domain::HtnDomain;
use crate::state::PlanState;
use crate::summaries::FieldSet;
use crate::tasks::{Precondition, Task};

/// The verdict of one look-ahead sweep.
pub(crate) enum Lookahead {
    /// The sequence can possibly succeed; contains the inevitable refinements
    /// found along it as `(position in the swept sequence, method index)` —
    /// the caller attaches them to the corresponding subtask occurrences.
    Refine(Vec<(usize, usize)>),
    /// No refinement of the sequence can succeed from the current state.
    DeadEnd,
}

/// Sweep `sequence` (task indices, in execution order) against `state`.
///
/// `state` is the planner's *working* state; the sweep runs on a lazily
/// created private clone plus an "unknown components" overlay, so the caller's
/// state is untouched. `unknown` and `pins` are caller-owned scratch buffers,
/// cleared and reused across sweeps to keep the per-commitment cost
/// allocation-light. `budget` is the planner's remaining step allowance:
/// sequences whose minimum yield cannot fit it are refuted outright.
pub(crate) fn sweep(
    domain: &HtnDomain,
    state: &PlanState,
    sequence: &[usize],
    budget: usize,
    unknown: &mut FieldSet,
    pins: &mut Vec<(usize, usize)>,
) -> Lookahead {
    // The sweep is only sound when the inferred summaries are present (they
    // define what "possibly written" and "terminating" mean).
    if domain.summaries.len() != domain.tasks.len() {
        return Lookahead::Refine(Vec::new());
    }

    unknown.clear();
    pins.clear();
    // The private state clone is only needed once a primitive's effects must
    // be applied; compound-only prefixes evaluate against the caller's state.
    let mut owned: Option<PlanState> = None;
    // Lower bound on the decomposition steps the sequence still needs (sum of
    // per-task minimum yields). Exceeding the budget proves the method cannot
    // finish within it — the same contract as dead-end refutation.
    let mut min_steps = 0usize;

    for (seq_position, &idx) in sequence.iter().enumerate() {
        // Budget: every task needs at least `min_yield` decomposition steps.
        min_steps = min_steps.saturating_add(domain.summaries[idx].min_yield);
        if min_steps > budget {
            return Lookahead::DeadEnd;
        }

        match &domain.tasks[idx] {
            Task::Primitive(p) => {
                for c in &p.preconditions {
                    if definitely_false(c, current(state, &owned), unknown) {
                        return Lookahead::DeadEnd;
                    }
                }
                if seq_position + 1 < sequence.len()
                    && (!p.effects.is_empty() || !p.expected_effects.is_empty())
                {
                    // Only apply effects when a later task could observe them;
                    // a trailing primitive's writes are invisible to the sweep
                    // (and this avoids cloning the state for tail-effect
                    // methods, the common shape).
                    let s = ensure_owned(state, &mut owned);
                    for e in p.effects.iter().chain(p.expected_effects.iter()) {
                        e.apply(s);
                        // Opaque closures: every written component becomes
                        // unknown (its new value may depend on the old one).
                        for &w in e.writes() {
                            unknown.insert(w);
                        }
                    }
                }
            }
            Task::Compound(c) => {
                // (Non-terminating tasks are caught by the budget check above:
                // their `min_yield` is `usize::MAX`, which exceeds any finite
                // budget — a task that can only refine forever can never
                // complete, so any method whose sequence contains it is
                // refuted outright.)
                let mut surviving = 0usize;
                let mut unique: Option<usize> = None;
                for (mi, m) in c.methods.iter().enumerate() {
                    if definitely_false_all(&m.preconditions, current(state, &owned), unknown) {
                        continue;
                    }
                    surviving += 1;
                    unique = Some(mi);
                    if surviving > 1 {
                        unique = None;
                        break;
                    }
                }
                if surviving == 0 {
                    return Lookahead::DeadEnd;
                }
                if surviving == 1 {
                    pins.push((seq_position, unique.expect("surviving == 1 implies unique")));
                }
                // Optimistic propagation: every component any surviving
                // method's refinement might write becomes "unknown".
                for m in c.methods.iter() {
                    if definitely_false_all(&m.preconditions, current(state, &owned), unknown) {
                        continue;
                    }
                    unknown.union_with(&m.possible_writes);
                }
            }
            // The forward planner stops at a goal task; mirror that.
            Task::Goal(_) => break,
        }
    }

    Lookahead::Refine(std::mem::take(pins))
}

/// The state to evaluate against: the private clone once it exists, else the
/// caller's working state.
fn current<'a>(state: &'a PlanState, owned: &'a Option<PlanState>) -> &'a PlanState {
    owned.as_ref().unwrap_or(state)
}

/// Materialize the private clone (first write) and return it.
fn ensure_owned<'a>(state: &'a PlanState, owned: &'a mut Option<PlanState>) -> &'a mut PlanState {
    owned.get_or_insert_with(|| state.clone())
}

/// Whether `c` definitely fails: it reads only known components and evaluates
/// false against them. Preconditions touching unknown components are "maybe".
fn definitely_false(c: &Precondition, state: &PlanState, unknown: &FieldSet) -> bool {
    for &r in c.reads() {
        if unknown.contains(r) {
            return false;
        }
    }
    !c.evaluate(state)
}

/// Whether any precondition of a list definitely fails.
fn definitely_false_all(
    preconditions: &[Precondition],
    state: &PlanState,
    unknown: &FieldSet,
) -> bool {
    preconditions
        .iter()
        .any(|c| definitely_false(c, state, unknown))
}
