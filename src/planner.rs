//! Forward HTN planner with MTR (Method Traversal Record) backtracking.
//!
//! Starting from the domain's root task, the planner repeatedly decomposes the
//! top of a task stack. Compound tasks pick the first method whose preconditions
//! hold (at or after a per-decomposition `skip`), push their subtasks, and
//! record the method index into the MTR. Primitive tasks whose preconditions
//! hold are appended to the plan and their (expected) effects are applied to
//! the working [`PlanState`] scratchpad. When a task can't be satisfied, the
//! planner backtracks to the most recent decomposition, tries the next method,
//! and restores the plan/MTR/state.
//!
//! # Performance
//!
//! The hot loop works on **`usize` task indices**, never names. The working
//! stack and the backtracking frames hold `usize` indices into
//! [`HtnDomain::tasks`]. Task names are interned as [`Ustr`] keys in a
//! precomputed `name -> index` map, so root resolution is O(1).
//!
//! `plan` and `mtr` are **append-only**, so backtracking frames store only the
//! two **lengths** (not cloned `Vec`s) — on backtrack a `truncate` restores
//! the exact prefix, which is provably identical to restoring a full clone but
//! avoids ~2 heap allocations + O(n) copies per recursion level. State
//! rollback is similarly allocation-light: before a primitive's effects run,
//! only the slots they write are snapshotted onto a pre-allocated rollback
//! stack; backtracking pops and restores exactly those. Task names are only
//! materialized as [`Ustr`]s when the final [`Plan`] is constructed.

use std::collections::VecDeque;

use smallvec::SmallVec;
use ustr::Ustr;

use crate::domain::HtnDomain;
use crate::lookahead::{self, Lookahead};
use crate::state::PlanState;
use crate::tasks::Task;

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

/// A completed forward plan: a **compiled step program** plus the MTR.
///
/// `steps` holds task indices into [`HtnDomain::tasks`](crate::domain::HtnDomain)
/// in execution order, so executing a plan is a flat array walk — the driver
/// indexes the baked task array directly, with no name lookups and no string
/// comparisons on the hot path. The interned names are kept in parallel for
/// display and introspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Baked step program: task indices, in execution order.
    pub steps: Vec<u32>,
    /// Interned task names, parallel to [`Self::steps`] (display/introspection;
    /// execution reads [`Self::steps`] only).
    pub names: Vec<Ustr>,
    /// Method indices chosen at each decomposition level.
    pub mtr: Mtr,
}

impl Plan {
    /// The ordered primitive task names (interned handles; deref to `&str`).
    pub fn task_names(&self) -> &[Ustr] {
        &self.names
    }

    /// The domain task index of the step at `cursor` (the compiled program
    /// entry the executor jumps to).
    pub fn step_task(&self, cursor: usize) -> Option<usize> {
        self.steps.get(cursor).map(|&s| s as usize)
    }

    /// The number of steps in the compiled program.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the program has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
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

/// Rollback journal for the search: snapshotted slot bytes plus the
/// `(slot, size)` ops that produced them, both append-only. Restoring pops
/// both stacks in lockstep — a bit-identical move of each old value back into
/// its slot (drop current, `memcpy` old bytes), allocation-free after warmup.
struct Rollback {
    bytes: Vec<u8>,
    ops: Vec<(usize, usize)>,
}

impl Rollback {
    fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(256),
            ops: Vec::with_capacity(16),
        }
    }

    /// The number of snapshotted ops (the value frames store).
    fn len(&self) -> usize {
        self.ops.len()
    }

    fn snapshot(&mut self, state: &PlanState, idx: usize) {
        state.snapshot_slot(idx, &mut self.bytes);
        self.ops.push((idx, state.slot_size(idx)));
    }

    fn restore_to(&mut self, len: usize, state: &mut PlanState) {
        while self.ops.len() > len {
            let (idx, size) = self.ops.pop().expect("len invariant");
            let start = self.bytes.len() - size;
            unsafe {
                state.restore_slot(idx, self.bytes[start..].as_ptr());
            }
            self.bytes.truncate(start);
        }
    }
}

/// Planner state used during one decomposition site for backtracking.
///
/// Stores **task indices** (not names) so frames stay tiny and copy-free.
/// `plan` and `mtr` are append-only within one `plan()` call — recursive
/// descent (primitives and method indices are only ever `push`ed). So instead
/// of deep-cloning both `Vec`s into every frame (which costs ~2n allocations
/// + O(n) copies per recursion level — catastrophic when a domain recursively
/// decomposes its root toward the sanity limit), we snapshot just the two
/// lengths. On backtrack we `truncate` back to those lengths, which is
/// provably identical to restoring a clone because the prefix of an
/// append-only Vec never changes. `skip_next` alone traces the search branch,
/// so this stays a fully correct DFS MTR backtrack.
/// Storage width for **task indices** in the search's transient structures,
/// chosen once per [`HtnPlanner::plan`] call from the domain's task count:
/// `u8` for domains of up to 256 tasks (the overwhelmingly common case),
/// `u16` up to 65 536, `u32` beyond. The search is monomorphized per width —
/// no branching in the hot loop — so a typical domain's queue entries shrink
/// from 24 to 8 bytes and decomposition frames shrink proportionally.
///
/// Method indices stay `usize`/`u32` (method counts are tiny and pins carry
/// them); only task indices narrow.
trait NarrowIdx: Copy + 'static {
    /// The exclusive upper bound of representable task counts.
    const MAX_TASKS: usize;

    fn new(idx: usize) -> Self;

    fn get(self) -> usize;
}

impl NarrowIdx for u8 {
    const MAX_TASKS: usize = u8::MAX as usize + 1;
    fn new(idx: usize) -> Self {
        u8::try_from(idx).expect("task index within dispatch width")
    }
    fn get(self) -> usize {
        self as usize
    }
}

impl NarrowIdx for u16 {
    const MAX_TASKS: usize = u16::MAX as usize + 1;
    fn new(idx: usize) -> Self {
        u16::try_from(idx).expect("task index within dispatch width")
    }
    fn get(self) -> usize {
        self as usize
    }
}

impl NarrowIdx for u32 {
    const MAX_TASKS: usize = u32::MAX as usize;
    fn new(idx: usize) -> Self {
        u32::try_from(idx).expect("task index within dispatch width")
    }
    fn get(self) -> usize {
        self as usize
    }
}

struct DecompositionFrame<I: NarrowIdx> {
    /// The compound task index being decomposed.
    task: I,
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
    pinned: Option<u32>,
    /// The task-queue suffix at commitment time (everything queued behind this
    /// decomposition, each entry carrying its own occurrence pin). The queue
    /// is consumed front-to-back during search, so a backtrack that re-queues
    /// this task must also restore what followed it — truncating lengths alone
    /// would lose siblings already popped after a failed later subtask (or
    /// leave stale ones from the abandoned choice). Inline for the common
    /// 2–5-subtask case (allocation-free commitments).
    stack: SmallVec<[(I, Option<u32>); 4]>,
    /// `rollback.ops.len()` before this decomposition's subtasks ran.
    /// Restoring the scratchpad rewinds the journal down to this length.
    rollback_len: usize,
}

/// A forward planner over a baked [`HtnDomain`].
///
/// Planning mutates no external state: it works on its own clone of the
/// extracted [`PlanState`] scratchpad, so it can be called repeatedly and
/// cheaply across turns.
pub struct HtnPlanner<'a> {
    domain: &'a HtnDomain,
    /// Whether the look-ahead sweep runs before each method commitment
    /// (default: enabled).
    lookahead: bool,
    /// Decomposition-step budget before the best partial plan is returned
    /// (default: 100).
    sanity_limit: usize,
}

impl<'a> HtnPlanner<'a> {
    /// Create a planner bound to a domain.
    pub fn new(domain: &'a HtnDomain) -> Self {
        Self {
            domain,
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
    /// `state` is only read: the planner works on its own clone of the
    /// scratchpad.
    ///
    /// Before committing each method choice, a look-ahead sweep
    /// ([`lookahead`]) proves the remaining sequence can possibly succeed;
    /// doomed methods are skipped at the frame and inevitable refinements
    /// (unique surviving methods) are pinned for when the planner reaches them.
    pub fn plan(&mut self, root: &str, state: &PlanState) -> Plan {
        // Dispatch once on the domain's task count: the whole search runs
        // monomorphized at the narrowest index width that can address every
        // task, so the hot loop's metadata is as small as the domain allows.
        let task_count = self.domain.tasks.len();
        if task_count <= u8::MAX_TASKS {
            self.plan_search::<u8>(root, state)
        } else if task_count <= u16::MAX_TASKS {
            self.plan_search::<u16>(root, state)
        } else {
            self.plan_search::<u32>(root, state)
        }
    }

    /// The search itself, monomorphized over the task-index width `I`.
    fn plan_search<I: NarrowIdx>(&mut self, root: &str, state: &PlanState) -> Plan {
        let sanity_limit = self.sanity_limit;
        let mut count = 0;
        let mut stack: VecDeque<(I, Option<u32>)> = VecDeque::with_capacity(16);
        let mut decomp_stack: Vec<DecompositionFrame<I>> = Vec::with_capacity(8);
        let mut mtr: Vec<usize> = Vec::with_capacity(8);
        let mut plan: Vec<I> = Vec::with_capacity(8);
        // Reusable look-ahead scratch: the sweep's "unknown components" overlay
        // and its inevitable-refinement output, cleared per sweep.
        let mut sweep_unknown = crate::summaries::FieldSet::new(self.domain.components.len());
        let mut sweep_pins: Vec<(usize, usize)> = Vec::with_capacity(4);
        // Reusable survivor buffer for the sweep's single-pass compound check.
        let mut sweep_surviving: Vec<usize> = Vec::with_capacity(8);
        // Reusable scratch for the commitment's resolved subtask list and the
        // sweep's sequence view of it.
        let mut resolved_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
        let mut seq_buf: Vec<usize> = Vec::with_capacity(8);
        // Rollback journal: snapshotted slot bytes + ops, restored on
        // backtrack down to the frame's length (allocation-free after warmup).
        let mut rollback = Rollback::new();
        // Reusable look-ahead state clone: the sweep's lazily-created private
        // copy, reused across sweeps (`copy_from`, no re-allocation).
        let mut sweep_owned: Option<PlanState> = None;
        let mut skip = 0;
        let mut state = state.clone();

        let tasks = &self.domain.tasks;

        let root = Ustr::from(root);
        let Some(&root_idx) = self.domain.index_of.get(&root) else {
            return Plan {
                steps: Vec::new(),
                names: Vec::new(),
                mtr: Mtr(Vec::new()),
            };
        };
        stack.push_back((I::new(root_idx), None));

        'search: while let Some((current, occurrence_pin)) = stack.pop_front() {
            count += 1;
            if count > sanity_limit {
                return materialize(tasks, &plan, mtr);
            }

            let task = &tasks[current.get()];

            match task {
                Task::Compound(compound) => {
                    // An ancestor look-ahead may have proven all methods but
                    // one infeasible for this occurrence under every reachable
                    // state: commit to (or exhaust) that method only.
                    let pin = occurrence_pin;
                    loop {
                        let eligible = match pin {
                            Some(pm) if pm as usize >= skip => compound
                                .methods
                                .get(pm as usize)
                                .filter(|m| m.preconditions.iter().all(|c| c.evaluate(&state)))
                                .map(|m| (m, pm as usize)),
                            // The pinned method was already tried and failed;
                            // every other method was proven infeasible at pin
                            // time, so this task is exhausted.
                            Some(_) => None,
                            None => compound.find_method(&state, skip),
                        };
                        let Some((method, idx)) = eligible else {
                            // No eligible method: unwind to the most recent
                            // decomposition and try its next choice.
                            if !backtrack(
                                &mut decomp_stack,
                                &mut plan,
                                &mut mtr,
                                &mut stack,
                                &mut rollback,
                                &mut skip,
                                &mut state,
                            ) {
                                break 'search;
                            }
                            continue 'search;
                        };

                        // Resolve this method's subtasks to (position, index)
                        // — needed to push the queue regardless of the sweep.
                        resolved_buf.clear();
                        for (pos, &sub) in method.subtasks.iter().enumerate() {
                            resolved_buf.push((pos, sub as usize));
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
                                &state,
                                &mut sweep_owned,
                                &seq_buf,
                                sanity_limit.saturating_sub(count),
                                &mut sweep_unknown,
                                &mut sweep_pins,
                                &mut sweep_surviving,
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
                                    rollback_len: rollback.len(),
                                };
                                decomp_stack.push(frame);
                                // Push subtask occurrences in reverse so the
                                // first pops first, attaching each one's
                                // inevitable-refinement pin (if the sweep
                                // proved its other methods infeasible).
                                for (pos, sub_idx) in resolved_buf.iter().rev() {
                                    let sub_pin = pins_found
                                        .iter()
                                        .find(|(p, _)| p == pos)
                                        .map(|&(_, m)| m as u32);
                                    stack.push_front((I::new(*sub_idx), sub_pin));
                                }
                                skip = 0;
                                continue 'search;
                            }
                        }
                    }
                }
                Task::Primitive(primitive) => {
                    if primitive.preconditions_met(&state) {
                        plan.push(current);
                        // Snapshot every slot the effects write before the
                        // first write, so backtracking can restore them.
                        for e in primitive
                            .effects
                            .iter()
                            .chain(primitive.expected_effects.iter())
                        {
                            for &w in e.writes() {
                                rollback.snapshot(&state, w);
                            }
                            e.apply(&mut state);
                        }
                        skip = 0;
                        continue;
                    }
                    if !backtrack(
                        &mut decomp_stack,
                        &mut plan,
                        &mut mtr,
                        &mut stack,
                        &mut rollback,
                        &mut skip,
                        &mut state,
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

        materialize(tasks, &plan, mtr)
    }
}

/// Unwind one decomposition frame (or, for a pinned task whose only viable
/// method failed, keep unwinding past it). Restores the append-only prefixes
/// by truncation, the scratchpad by rollback, and re-queues the frame's task
/// for its next method choice. Returns `false` when the search is exhausted.
#[allow(clippy::too_many_arguments)]
fn backtrack<I: NarrowIdx>(
    decomp_stack: &mut Vec<DecompositionFrame<I>>,
    plan: &mut Vec<I>,
    mtr: &mut Vec<usize>,
    stack: &mut VecDeque<(I, Option<u32>)>,
    rollback: &mut Rollback,
    skip: &mut usize,
    state: &mut PlanState,
) -> bool {
    loop {
        match decomp_stack.pop() {
            Some(frame) => {
                plan.truncate(frame.plan_len);
                mtr.truncate(frame.mtr_len);
                // Restore the scratchpad: undo every effect applied since the
                // frame committed (newest first).
                rollback.restore_to(frame.rollback_len, state);
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

/// Convert a plan of (narrow) task indices into the compiled step program:
/// contiguous `u32` task indices plus the parallel interned-name list.
fn materialize<I: NarrowIdx>(tasks: &[Task], plan: &[I], mtr: Vec<usize>) -> Plan {
    Plan {
        steps: plan.iter().map(|&i| i.get() as u32).collect(),
        names: plan.iter().map(|&i| tasks[i.get()].name().into()).collect(),
        mtr: Mtr(mtr),
    }
}
