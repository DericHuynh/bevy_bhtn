//! Forward HTN planner with MTR (Method Traversal Record) backtracking.
//!
//! Starting from the domain's root task, the planner repeatedly decomposes the
//! top of a task stack. Compound tasks pick the first method whose preconditions
//! hold (at or after a per-decomposition `skip`), push their subtasks, and
//! record the method index into the MTR. Primitive tasks whose preconditions
//! hold are appended to the plan and their (expected) effects are applied to a
//! working copy of the state. When a task can't be satisfied, the planner
//! backtracks to the most recent decomposition, tries the next method, and
//! restores the plan/MTR.
//!
//! # Performance
//!
//! The hot loop works on **`usize` task indices**, never names. The working
//! stack and the backtracking frames hold `usize` indices into
//! [`HtnDomain::tasks`]. Domains intern task names as [`Ustr`] keys in a
//! precomputed `name -> index` map, so subtask resolution is O(1), not a linear
//! scan.
//!
//! `plan` and `mtr` are **append-only**, so backtracking frames store only the
//! two **lengths** (not cloned `Vec`s) — on backtrack a `truncate` restores the
//! exact prefix, which is provably identical to restoring a full clone but
//! avoids ~2 heap allocations + O(n) copies per recursion level. This is the
//! dominant win for domains that recursively decompose a root toward the sanity
//! limit (e.g. the miner benchmark). Task names are only materialized as
//! [`Ustr`]s when the final [`Plan`] is constructed.

use std::collections::VecDeque;

use bevy_reflect::TypeRegistry;
use ustr::Ustr;

use crate::domain::HtnDomain;
use crate::lookahead::{self, Lookahead};
use crate::tasks::Task;
use crate::HtnState;

/// The method traversal record of a completed plan: the index of the chosen
/// method at each decomposition level. Used to compare plans by priority
/// (lower index = higher priority) and for debugging.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Mtr(pub Vec<usize>);

impl std::fmt::Display for Mtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(".")
        )
    }
}

/// A completed forward plan: an ordered list of primitive task names (interned
/// [`Ustr`]s, so it's cheap to copy, compare, and hand around) plus the MTR
/// describing how each compound was decomposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Primitive task names, in execution order.
    pub tasks: Vec<Ustr>,
    /// Method indices chosen at each decomposition level.
    pub mtr: Mtr,
}

impl Plan {
    /// The ordered primitive task names (interned handles; deref to `&str`).
    pub fn task_names(&self) -> &[Ustr] {
        &self.tasks
    }

    /// The MTR for this plan.
    pub fn mtr(&self) -> &Mtr {
        &self.mtr
    }

    /// Order two plans by MTR priority (lower first).
    pub fn is_preferred_over(&self, other: &Self) -> bool {
        for (a, b) in self.mtr.0.iter().zip(other.mtr.0.iter()) {
            match a.cmp(b) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord == std::cmp::Ordering::Less,
            }
        }
        // Prefer the shorter MTR when they share a prefix.
        self.mtr.0.len() < other.mtr.0.len()
    }
}

/// Planner state used during one decomposition site for backtracking.
///
/// Stores **task indices** (not names) so frames stay tiny and copy-free.
///
/// # Backtracking via lengths
///
/// [`HtnPlanner`] builds `plan` and `mtr` **append-only** during a monotonic
/// recursive descent (primitives and method indices are only ever `push`ed).
/// So instead of deep-cloning both `Vec`s into every frame (which costs ~2n
/// allocations + O(n) copies per recursion level — catastrophic when a domain
/// recursively decomposes its root toward the sanity limit), we snapshot just
/// the two lengths. On backtrack we `truncate` back to those lengths, which is
/// provably identical to restoring a clone because the prefix of an append-only
/// Vec never changes. `skip_next` alone traces the search branch, so this stays
/// a fully correct DFS MTR backtrack.
#[derive(Debug)]
struct DecompositionFrame {
    /// The compound task index being decomposed.
    task: usize,
    /// `plan.len()` before this decomposition's subtasks were entered.
    plan_len: usize,
    /// The number of methods to skip (index+1 of the one just tried).
    skip_next: usize,
    /// `mtr.len()` before adding this decomposition's method index.
    mtr_len: usize,
    /// The method index this task occurrence was pinned to by an ancestor
    /// look-ahead, if any. When a pinned task's method fails, every other
    /// method was already proven infeasible at pin time — so backtracking
    /// skips them entirely.
    pinned: Option<usize>,
    /// The task-queue suffix at commitment time (everything queued behind this
    /// decomposition, each entry carrying its own occurrence pin). The queue
    /// is consumed front-to-back during search, so a backtrack that re-queues
    /// this task must also restore what followed it — truncating lengths alone
    /// would lose siblings already popped after a failed later subtask (or
    /// leave stale ones from the abandoned choice). Empty (and allocation-free)
    /// for the common last-subtask case.
    stack: Vec<(usize, Option<usize>)>,
}

/// A forward planner over a parsed [`HtnDomain`].
///
/// Planning mutates no external state: it clones the initial state and works on
/// its own copy, so it can be called repeatedly and cheaply across turns.
pub struct HtnPlanner<'a> {
    domain: &'a HtnDomain,
    registry: &'a TypeRegistry,
    /// Whether the look-ahead sweep runs before each method commitment
    /// (default: enabled).
    lookahead: bool,
    /// Decomposition-step budget before the best partial plan is returned
    /// (default: 100).
    sanity_limit: usize,
}

impl<'a> HtnPlanner<'a> {
    /// Create a planner bound to a domain and a type registry (for reflection
    /// evaluation / effect application).
    pub fn new(domain: &'a HtnDomain, registry: &'a TypeRegistry) -> Self {
        Self {
            domain,
            registry,
            lookahead: true,
            sanity_limit: 100,
        }
    }

    /// The domain this planner reads.
    pub fn domain(&self) -> &'a HtnDomain {
        self.domain
    }

    /// Enable or disable look-ahead pruning (default: enabled). Disabling
    /// falls back to plain MTR backtracking; useful for A/B benchmarking and
    /// for domains where the sweep's per-commitment cost outweighs its
    /// pruning (e.g. shallow domains with no dead ends).
    pub fn set_lookahead(&mut self, enabled: bool) -> &mut Self {
        self.lookahead = enabled;
        self
    }

    /// Set the decomposition-step budget after which the best partial plan is
    /// returned (default: 100). Raise it for domains that legitimately need
    /// deep searches; the look-ahead usually refutes doomed branches long
    /// before the budget matters.
    pub fn set_sanity_limit(&mut self, limit: usize) -> &mut Self {
        self.sanity_limit = limit;
        self
    }

    /// Decompose `root` into a [`Plan`]. Even if no task satisfies, the search
    /// terminates after exhausting backtracking and returns the best partial
    /// plan found (with an empty task list if nothing was decomposable).
    ///
    /// Before committing each method choice, a look-ahead sweep
    /// ([`lookahead`]) proves the remaining sequence can possibly succeed;
    /// doomed methods are skipped at the frame and inevitable refinements
    /// (unique surviving methods) are pinned for when the planner reaches them.
    pub fn plan<S: HtnState>(&mut self, root: &str, initial_state: &S) -> Plan {
        let sanity_limit = self.sanity_limit;
        let mut count = 0;
        let mut stack: VecDeque<(usize, Option<usize>)> = VecDeque::with_capacity(16);
        let mut decomp_stack: Vec<DecompositionFrame> = Vec::with_capacity(8);
        let mut mtr: Vec<usize> = Vec::with_capacity(8);
        let mut plan: Vec<usize> = Vec::with_capacity(8);
        // Reusable look-ahead scratch: the sweep's "unknown fields" overlay
        // and its inevitable-refinement output, cleared per sweep.
        let mut sweep_unknown = crate::summaries::FieldSet::new(self.domain.fields.len());
        let mut sweep_pins: Vec<(usize, usize)> = Vec::with_capacity(4);
        // Reusable scratch for the commitment's resolved subtask list and the
        // sweep's sequence view of it.
        let mut resolved_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
        let mut seq_buf: Vec<usize> = Vec::with_capacity(8);
        let mut skip = 0;
        let mut state = initial_state.clone();

        let tasks = &self.domain.tasks;

        let root = Ustr::from(root);
        let Some(&root_idx) = self.domain.index_of.get(&root) else {
            return Plan {
                tasks: Vec::new(),
                mtr: Mtr(Vec::new()),
            };
        };
        stack.push_back((root_idx, None));

        let registry = self.registry;

        'search: while let Some((current, occurrence_pin)) = stack.pop_front() {
            count += 1;
            if count > sanity_limit {
                return Plan {
                    tasks: materialize_names(tasks, &plan),
                    mtr: Mtr(mtr),
                };
            }

            let task = &tasks[current];

            match task {
                Task::Compound(compound) => {
                    // An ancestor look-ahead may have proven all methods but
                    // one infeasible for this occurrence under every reachable
                    // state: commit to (or exhaust) that method only.
                    let pin = occurrence_pin;
                    loop {
                        let eligible = match pin {
                            Some(pm) if pm >= skip => compound
                                .methods
                                .get(pm)
                                .filter(|m| {
                                    m.preconditions
                                        .iter()
                                        .all(|c| c.evaluate(state.as_reflect()))
                                })
                                .map(|m| (m, pm)),
                            // The pinned method was already tried and failed;
                            // every other method was proven infeasible at pin
                            // time, so this task is exhausted.
                            Some(_) => None,
                            None => compound.find_method(state.as_reflect(), skip),
                        };
                        let Some((method, idx)) = eligible else {
                            // No eligible method: unwind to the most recent
                            // decomposition and try its next choice.
                            if !backtrack(
                                &mut decomp_stack,
                                &mut plan,
                                &mut mtr,
                                &mut stack,
                                &mut skip,
                            ) {
                                break 'search;
                            }
                            continue 'search;
                        };

                        // Resolve this method's subtasks to (position, index)
                        // — needed to push the queue regardless of the sweep.
                        resolved_buf.clear();
                        for (pos, sub) in method.subtasks.iter().enumerate() {
                            if let Some(idx) = self.domain.task_index(*sub) {
                                resolved_buf.push((pos, idx));
                            }
                        }
                        // Look-ahead: can any refinement of this choice
                        // possibly succeed? Scoped to the method's own
                        // subtasks — they execute immediately after this
                        // commitment, against the exact current state. (The
                        // sweep skips the last task's effects, so single-
                        // subtask methods never clone the state.)
                        let verdict = if self.lookahead {
                            seq_buf.clear();
                            seq_buf.extend(resolved_buf.iter().map(|&(_, idx)| idx));
                            lookahead::sweep(
                                self.domain,
                                registry,
                                &state,
                                &seq_buf,
                                sanity_limit.saturating_sub(count),
                                &mut sweep_unknown,
                                &mut sweep_pins,
                            )
                        } else {
                            Lookahead::Refine(Vec::new())
                        };
                        match verdict {
                            Lookahead::DeadEnd => {
                                // Proven doomed without recursing: try the
                                // next method at this site.
                                skip = idx + 1;
                                continue;
                            }
                            Lookahead::Refine(pins_found) => {
                                mtr.push(idx);
                                let frame = DecompositionFrame {
                                    task: current,
                                    plan_len: plan.len(),
                                    skip_next: idx + 1,
                                    // Snapshot *after* the push so restoring
                                    // truncates back to a world that includes
                                    // this method choice.
                                    mtr_len: mtr.len(),
                                    pinned: pin,
                                    stack: stack.iter().copied().collect(),
                                };
                                decomp_stack.push(frame);
                                // Push subtask occurrences in reverse so the
                                // first pops first, attaching each one's
                                // inevitable-refinement pin (if the sweep
                                // proved its other methods infeasible).
                                for (pos, sub_idx) in resolved_buf.iter().rev() {
                                    let sub_pin =
                                        pins_found.iter().find(|(p, _)| p == pos).map(|&(_, m)| m);
                                    stack.push_front((*sub_idx, sub_pin));
                                }
                                skip = 0;
                                continue 'search;
                            }
                        }
                    }
                }
                Task::Primitive(primitive) => {
                    if primitive.preconditions_met(state.as_reflect()) {
                        plan.push(current);
                        for e in primitive.effects.iter() {
                            e.apply_dyn(state.as_reflect_mut(), registry);
                        }
                        for e in primitive.expected_effects.iter() {
                            e.apply_dyn(state.as_reflect_mut(), registry);
                        }
                        skip = 0;
                        continue;
                    }
                    if !backtrack(
                        &mut decomp_stack,
                        &mut plan,
                        &mut mtr,
                        &mut stack,
                        &mut skip,
                    ) {
                        break 'search;
                    }
                    continue;
                }
                Task::Goal(_) => {
                    break;
                }
            }
        }

        Plan {
            tasks: materialize_names(tasks, &plan),
            mtr: Mtr(mtr),
        }
    }
}

/// Unwind one decomposition frame (or, for a pinned task whose only viable
/// method failed, keep unwinding past it). Restores the append-only prefixes
/// by truncation and re-queues the frame's task for its next method choice.
/// Returns `false` when the search is exhausted.
fn backtrack(
    decomp_stack: &mut Vec<DecompositionFrame>,
    plan: &mut Vec<usize>,
    mtr: &mut Vec<usize>,
    stack: &mut VecDeque<(usize, Option<usize>)>,
    skip: &mut usize,
) -> bool {
    loop {
        match decomp_stack.pop() {
            Some(frame) => {
                plan.truncate(frame.plan_len);
                mtr.truncate(frame.mtr_len);
                // Restore the queue to its state at commitment time: the
                // failed subtree's remnants go, the suffix (with its
                // occurrence pins) comes back.
                stack.clear();
                stack.extend(frame.stack.iter().copied());
                if frame.pinned.is_some() {
                    // This task was pinned to a single method and it failed:
                    // all alternatives were proven infeasible at pin time, so
                    // keep unwinding instead of retrying them.
                    continue;
                }
                *skip = frame.skip_next;
                stack.push_front((frame.task, frame.pinned));
                return true;
            }
            None => return false,
        }
    }
}

/// Convert a plan of task indices into interned task-name [`Ustr`]s in the same
/// order.
fn materialize_names(tasks: &[Task], plan: &[usize]) -> Vec<Ustr> {
    plan.iter().map(|&i| tasks[i].name().into()).collect()
}
