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
use crate::domain::Task;
use crate::state::FieldSet;
use crate::state::PlanState;
use crate::tasks::Precondition;

/// The verdict of one look-ahead sweep.
///
/// On [`Lookahead::Refine`], the inevitable refinements are left in the
/// caller's `pins` scratch buffer (same discipline as `unknown` and
/// `surviving_buf`) — taking ownership would swap the caller's populated
/// `Vec` for a fresh empty one and defeat the reuse.
pub(crate) enum Lookahead {
    /// The sequence can possibly succeed; the caller's `pins` buffer holds
    /// the inevitable refinements found along it as `(position in the swept
    /// sequence, method index)` — the caller attaches them to the
    /// corresponding subtask occurrences.
    Refine,
    /// No refinement of the sequence can succeed from the current state.
    DeadEnd,
}

/// How much analysis one look-ahead sweep performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SweepDepth {
    /// Full analysis: compound-survivor checks (dead ends + pins + optimistic
    /// propagation), primitive-precondition checks, budget refutation.
    Full,
    /// Refutation-only: budget refutation and primitive-precondition checks
    /// — no compound-survivor analysis (no pins, no compound dead ends, no
    /// optimistic propagation). A fraction of the cost on wide selectors,
    /// where the survivor analysis re-evaluates every branch.
    RefutationOnly,
}

/// Sweep `sequence` (task indices, in execution order) against `state`.
///
/// `state` is the planner's *working* state; the sweep runs on a lazily
/// created private clone plus an "unknown components" overlay, so the caller's
/// state is untouched. `unknown` and `pins` are caller-owned scratch buffers,
/// cleared and reused across sweeps to keep the per-commitment cost
/// allocation-light. `budget` is the planner's remaining step allowance:
/// sequences whose minimum yield cannot fit it are refuted outright.
///
/// `set_semantics` marks the sequence as a **partially-ordered member set**
/// (each member runs exactly once, in a search-chosen order): every member's
/// possible writes are optimistic-unknown before any check, and no effects
/// are applied sequentially. Pruning weakens but stays sound — a dead-end
/// verdict means every member fails in every linearization, and a pin means
/// every other method fails in every linearization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sweep(
    domain: &HtnDomain,
    state: &PlanState,
    owned_scratch: &mut Option<PlanState>,
    sequence: &[usize],
    budget: usize,
    unknown: &mut FieldSet,
    pins: &mut Vec<(usize, usize)>,
    surviving_buf: &mut Vec<usize>,
    set_semantics: bool,
    depth: SweepDepth,
) -> Lookahead {
    // The sweep is only sound when the inferred summaries are present (they
    // define what "possibly written" and "terminating" mean).
    if domain.summaries.len() != domain.tasks.len() {
        return Lookahead::Refine;
    }

    unknown.clear();
    pins.clear();
    // Set semantics: the members run in a search-chosen order, so every
    // member's possible writes are optimistic-unknown before any check, and
    // no member's effects are applied sequentially (the private state clone
    // is never needed).
    if set_semantics {
        for &idx in sequence {
            unknown.union_with(&domain.summaries[idx].possible_writes);
        }
    }
    // The private state clone is only needed once a primitive's effects must
    // be applied; compound-only prefixes evaluate against the caller's state.
    // The scratch is owned by the planner and reused across sweeps
    // (`copy_from` — no re-allocation, just per-slot deep clones).
    let mut owned = false;
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
                let cur: &PlanState = if owned {
                    owned_scratch.as_ref().expect("owned scratch materialized")
                } else {
                    state
                };
                for c in &p.preconditions {
                    if definitely_false(c, cur, unknown) {
                        return Lookahead::DeadEnd;
                    }
                }
                if !set_semantics
                    && seq_position + 1 < sequence.len()
                    && (!p.effects.is_empty() || !p.expected_effects.is_empty())
                {
                    // Only apply effects when a later task could observe them;
                    // a trailing primitive's writes are invisible to the sweep
                    // (and this avoids cloning the state for tail-effect
                    // methods, the common shape).
                    if !owned {
                        match owned_scratch {
                            Some(scratch) => scratch.copy_from(state),
                            None => *owned_scratch = Some(state.clone()),
                        }
                        owned = true;
                    }
                    let scratch = owned_scratch.as_mut().expect("just materialized");
                    for e in p.effects.iter().chain(p.expected_effects.iter()) {
                        e.apply(scratch);
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
                // Refutation-only sweeps skip the survivor analysis: no
                // pins, no compound dead ends, no optimistic propagation —
                // on wide selectors that analysis re-evaluates every branch
                // and dominates the sweep cost.
                if depth == SweepDepth::RefutationOnly {
                    continue;
                }
                // Single pass: evaluate each method's preconditions once,
                // collecting survivors; then pin (unique survivor) and
                // propagate optimistic writes.
                let cur: &PlanState = if owned {
                    owned_scratch.as_ref().expect("owned scratch materialized")
                } else {
                    state
                };
                surviving_buf.clear();
                for (mi, m) in c.methods.iter().enumerate() {
                    if !definitely_false_all(&m.preconditions, cur, unknown) {
                        surviving_buf.push(mi);
                    }
                }
                if surviving_buf.is_empty() {
                    return Lookahead::DeadEnd;
                }
                if surviving_buf.len() == 1 {
                    pins.push((seq_position, surviving_buf[0]));
                }
                // Optimistic propagation: every component any surviving
                // method's refinement might write becomes "unknown".
                for &mi in surviving_buf.iter() {
                    unknown.union_with(&c.methods[mi].possible_writes);
                }
            }
            // The forward planner stops at a goal task; mirror that.
            Task::Goal(_) => break,
        }
    }

    Lookahead::Refine
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
