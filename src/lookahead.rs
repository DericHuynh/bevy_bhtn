//! Look-ahead pruning for the forward planner.
//!
//! Implements the Olz & Bercher (SoCS 2023) look-ahead technique, adapted to
//! this crate's typed-state planner: before the planner commits to a method, a
//! linear sweep over that **method's subtask sequence** proves — without
//! decomposing anything — which refinements can possibly succeed:
//!
//! - **Dead-end**: a primitive's preconditions definitely fail, a compound
//!   task has no method that can possibly apply, a task in the sequence has no
//!   finite refinement (it can only refine forever), or the sequence's minimum
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
//! compound tasks contribute their inferred `possible_writes` (fields become
//! "unknown"); primitives apply their effects exactly (they must execute for
//! the sequence to continue). Conditions on known fields are evaluated
//! exactly; conditions touching unknown fields are treated as "maybe" and
//! never prune. A dead-end verdict therefore means *every* refinement fails,
//! and a pin means *every other* method fails — under any reachable state.
//!
//! The sweep is scoped to the committed method's own subtasks (not the whole
//! remaining queue): the sequence executes immediately after commitment, so
//! the working state is exact at its start, and the per-commitment cost stays
//! bounded by the method's width instead of the whole plan. Cross-frame dead
//! ends are refuted when their own frame commits.

use bevy_reflect::TypeRegistry;
use ustr::Ustr;

use crate::conditions::HtnCondition;
use crate::domain::HtnDomain;
use crate::effects::Effect;
use crate::summaries::FieldSet;
use crate::tasks::Task;
use crate::HtnState;

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
/// created private clone plus an "unknown fields" overlay, so the caller's
/// state is untouched. `unknown` and `pins` are caller-owned scratch buffers,
/// cleared and reused across sweeps to keep the per-commitment cost
/// allocation-light. `budget` is the planner's remaining step allowance:
/// sequences whose minimum yield cannot fit it are refuted outright.
pub(crate) fn sweep<S: HtnState>(
    domain: &HtnDomain,
    registry: &TypeRegistry,
    state: &S,
    sequence: &[usize],
    budget: usize,
    unknown: &mut FieldSet,
    pins: &mut Vec<(usize, usize)>,
) -> Lookahead {
    // The sweep is only sound when the inferred summaries are present (they
    // define what "possibly written" and "terminating" mean). Hand-built
    // domains without a parse-time index rebuild skip the look-ahead entirely.
    if domain.summaries.len() != domain.tasks.len() {
        return Lookahead::Refine(Vec::new());
    }

    unknown.clear();
    pins.clear();
    // The private state clone is only needed once a primitive's effects must
    // be applied; compound-only prefixes evaluate against the caller's state.
    let mut owned: Option<S> = None;
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
                for (c, reads) in p.preconditions.iter().zip(p.prec_reads.iter()) {
                    if definitely_false(c, reads, current(state, &owned).as_reflect(), unknown) {
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
                    for e in p.effects.iter().chain(p.expected_effects.iter()) {
                        let s = ensure_owned(state, &mut owned);
                        apply_sweep_effect(e, s.as_reflect_mut(), registry, unknown, domain);
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
                    if definitely_false_all(
                        &m.preconditions,
                        &m.prec_reads,
                        current(state, &owned).as_reflect(),
                        unknown,
                    ) {
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
                // Optimistic propagation: every field any surviving method's
                // refinement might write becomes "unknown".
                for m in c.methods.iter() {
                    if definitely_false_all(
                        &m.preconditions,
                        &m.prec_reads,
                        current(state, &owned).as_reflect(),
                        unknown,
                    ) {
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
fn current<'a, S: HtnState>(state: &'a S, owned: &'a Option<S>) -> &'a S {
    owned.as_ref().unwrap_or(state)
}

/// Materialize the private clone (first write) and return it.
fn ensure_owned<'a, S: HtnState>(state: &'a S, owned: &'a mut Option<S>) -> &'a mut S {
    owned.get_or_insert_with(|| state.clone())
}

/// Whether `c` definitely fails: it reads only known fields and evaluates
/// false against them. Conditions touching unknown fields are "maybe".
/// `reads` holds the condition's precomputed field indices (`[primary,
/// comparison]`); a `None` slot means the field is absent from the domain's
/// field table and is therefore never "unknown".
fn definitely_false(
    c: &HtnCondition,
    reads: &[Option<usize>; 2],
    state: &dyn bevy_reflect::Reflect,
    unknown: &FieldSet,
) -> bool {
    for slot in reads.iter().flatten() {
        if unknown.contains(*slot) {
            return false;
        }
    }
    !c.evaluate(state)
}

/// Whether any condition of a precondition list definitely fails.
fn definitely_false_all(
    preconditions: &[HtnCondition],
    prec_reads: &[[Option<usize>; 2]],
    state: &dyn bevy_reflect::Reflect,
    unknown: &FieldSet,
) -> bool {
    preconditions
        .iter()
        .zip(prec_reads.iter())
        .any(|(c, reads)| definitely_false(c, reads, state, unknown))
}

/// Apply one effect to the sweep's private state and update the known/unknown
/// overlay for the written field.
fn apply_sweep_effect(
    e: &Effect,
    state: &mut dyn bevy_reflect::Reflect,
    registry: &TypeRegistry,
    unknown: &mut FieldSet,
    domain: &HtnDomain,
) {
    let fidx = domain.field_index.get(&Ustr::from(e.field())).copied();
    let was_unknown = fidx.map(|i| unknown.contains(i)).unwrap_or(false);
    e.apply_dyn(state, registry);
    let Some(i) = fidx else { return };
    match e {
        // Literal writes are deterministic regardless of the prior value.
        Effect::SetBool { .. }
        | Effect::SetInt { .. }
        | Effect::SetFloat { .. }
        | Effect::SetEnum { .. }
        | Effect::SetNone { .. } => unknown.remove(i),
        // Copies inherit the source's trustworthiness.
        Effect::SetIdentifier { field_source, .. } => {
            let src_known = domain
                .field_index
                .get(&Ustr::from(field_source))
                .map(|&s| !unknown.contains(s))
                .unwrap_or(true);
            if src_known {
                unknown.remove(i);
            }
        }
        // Relative writes are only deterministic when the prior value was.
        Effect::IncrementInt { .. } | Effect::IncrementFloat { .. } => {
            if !was_unknown {
                unknown.remove(i);
            }
        }
    }
}
