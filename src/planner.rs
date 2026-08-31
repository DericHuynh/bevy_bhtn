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
use std::ptr::NonNull;

use smallvec::SmallVec;
use ustr::Ustr;

use crate::selection::{DecompositionTrace, SelectionPolicy, TraceOutcome};

use crate::domain::HtnDomain;
use crate::lookahead::{self, Lookahead};
use crate::order::{linearize, SubtaskOrder};
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

/// One entry of the search's task queue: a task occurrence (with its
/// optional look-ahead pin) or a pending linearization retry of a
/// partially-ordered method.
#[derive(Clone, Copy, Debug)]
enum Step {
    Task(usize, Option<u32>),
    /// Re-commit the partial-order method `method` of compound `task` with
    /// its `lin`-th topological order (0-indexed). Queued by `backtrack` when
    /// a partial method's subtree failed but linearizations remain; popped
    /// immediately (it never coexists with deeper commitments).
    Linearize {
        task: u32,
        method: u32,
        lin: u32,
    },
}

/// Growable, max-aligned byte arena: the rollback journal's value storage.
/// A `Vec<u8>` cannot guarantee its buffer's alignment, and journaled values
/// are cloned in place through their component's typed cloner — so the arena
/// allocates with the domain's maximum slot alignment and every entry is
/// offset-aligned to its slot's alignment. Growing moves the values bitwise
/// (a plain Rust move — the journal owns them; the old buffer is deallocated
/// without dropping values).
struct Journal {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
    align: usize,
}

impl Journal {
    fn new(align: usize) -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            align: align.max(1),
        }
    }

    fn reserve(&mut self, additional: usize) {
        if self.len + additional <= self.cap {
            return;
        }
        let new_cap = (self.cap.max(256) * 2).max(self.len + additional);
        let layout =
            std::alloc::Layout::from_size_align(new_cap, self.align).expect("valid journal layout");
        let new_ptr = if self.cap == 0 {
            unsafe { std::alloc::alloc(layout) }
        } else {
            let old = std::alloc::Layout::from_size_align(self.cap, self.align)
                .expect("valid journal layout");
            unsafe { std::alloc::realloc(self.ptr.as_ptr(), old, new_cap) }
        };
        self.ptr = NonNull::new(new_ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        self.cap = new_cap;
    }

    /// Reserve `size` bytes aligned to `align`, returning the entry's offset.
    fn push_aligned(&mut self, size: usize, align: usize) -> usize {
        let start = self.len.next_multiple_of(align);
        self.reserve(start + size - self.len);
        self.len = start + size;
        start
    }

    fn truncate(&mut self, len: usize) {
        self.len = len;
    }

    fn ptr_at(&self, offset: usize) -> *mut u8 {
        unsafe { self.ptr.as_ptr().add(offset) }
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        if self.cap != 0 {
            let layout = std::alloc::Layout::from_size_align(self.cap, self.align)
                .expect("valid journal layout");
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

// The arena uniquely owns its allocation and is only accessed through
// `&Journal`/`&mut Journal` — same reasoning as `Vec<u8>`.
unsafe impl Send for Journal {}
unsafe impl Sync for Journal {}

/// Rollback journal for the search: **deep-cloned** slot values plus the
/// `(slot, journal offset)` ops that produced them, both append-only. The
/// journal
/// owns its copies — a bitwise snapshot would dangle the moment an in-place
/// mutation freed a heap allocation (HashMap grow, Vec shrink, whole-value
/// replacement), and reanimating it on restore is a double free. Restoring
/// pops both stacks in lockstep: drop the slot's current value, clone the
/// journal's copy back in, drop the journal's copy — every allocation has
/// exactly one owner at every point. For plain-data slots the cloner is a
/// memcpy, so the common case keeps the old cost.
struct Rollback {
    values: Journal,
    /// `(slot, journal offset)` per snapshot — the offset is stored because
    /// alignment padding between entries makes offsets not derivable from
    /// the current length.
    ops: Vec<(usize, usize)>,
}

impl Rollback {
    fn new(max_align: usize) -> Self {
        Self {
            values: Journal::new(max_align),
            ops: Vec::with_capacity(16),
        }
    }

    /// The number of snapshotted ops (the value frames store).
    fn len(&self) -> usize {
        self.ops.len()
    }

    fn snapshot(&mut self, state: &PlanState, idx: usize) {
        let align = state.slot_align(idx);
        let padded = state.slot_size(idx).next_multiple_of(align);
        let start = self.values.push_aligned(padded, align);
        state.snapshot_slot(idx, self.values.ptr_at(start));
        self.ops.push((idx, start));
    }

    fn restore_to(&mut self, len: usize, state: &mut PlanState) {
        while self.ops.len() > len {
            let (idx, start) = self.ops.pop().expect("len invariant");
            unsafe {
                // Drop the slot's current value, clone the journal's copy
                // back in, then drop the journal's copy.
                state.restore_slot(idx, self.values.ptr_at(start));
                state.drop_journaled_slot(idx, self.values.ptr_at(start));
            }
            self.values.truncate(start);
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
    pinned: Option<u32>,
    /// The task-queue suffix at commitment time (everything queued behind this
    /// decomposition, each entry carrying its own occurrence pin). The queue
    /// is consumed front-to-back during search, so a backtrack that re-queues
    /// this task must also restore what followed it — truncating lengths alone
    /// would lose siblings already popped after a failed later subtask (or
    /// leave stale ones from the abandoned choice). Inline for the common
    /// 2–5-subtask case (allocation-free commitments).
    stack: SmallVec<[Step; 4]>,
    /// `rollback.ops.len()` before this decomposition's subtasks ran.
    /// Restoring the scratchpad rewinds the journal down to this length.
    rollback_len: usize,
    /// The position in the node's ranked branch order to resume from on
    /// backtrack. The order itself is *not* stored: every policy re-derives
    /// it deterministically on revisit (the sampler's nonce is the restored
    /// `plan.len()`), so backtracking resumes down the same ranked list
    /// without storing it.
    rank_resume: usize,
    /// The accumulated primitive cost when this decomposition committed
    /// (cost-bounded search only; 0 otherwise). Restoring it on backtrack
    /// mirrors the `plan` truncation: every primitive pushed after the
    /// commitment is removed, so `g` returns to its commitment value.
    g_commit: f32,
    /// The next linearization to try when this partial-order method's subtree
    /// fails (1-based; 0 for total-order methods, which have no retries).
    /// `lin_total` is the method's baked (capped) topological-order count; a
    /// retry is queued only while `lin < lin_total`.
    lin: u32,
    lin_total: u32,
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
    /// Fail-fast mode: abandon the branch on first downstream failure and
    /// return the partial plan immediately (no backtracking).
    fail_fast: bool,
    /// Cost-bounded branch-and-bound mode (see [`Self::set_cost_bounded`]).
    cost_bounded: bool,
}

impl<'a> HtnPlanner<'a> {
    /// Create a planner bound to a domain.
    pub fn new(domain: &'a HtnDomain) -> Self {
        Self {
            domain,
            lookahead: true,
            sanity_limit: 100,
            fail_fast: false,
            cost_bounded: false,
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

    /// Enable fail-fast mode: abandon the branch on first downstream failure
    /// and return the partial plan immediately (no backtracking). Used by the
    /// [`DepthFirstFailFast`](crate::selection::HtnSearchStrategy::DepthFirstFailFast)
    /// strategy.
    pub fn set_fail_fast(&mut self, enabled: bool) -> &mut Self {
        self.fail_fast = enabled;
        self
    }

    /// Enable cost-bounded branch-and-bound: keep the cheapest *complete*
    /// plan found within the sanity budget and prune any branch whose
    /// accumulated primitive cost plus the bake-time `min_cost` lower bound
    /// of its remaining sequence cannot strictly beat it. Used by the
    /// [`CostBounded`](crate::selection::HtnSearchStrategy::CostBounded)
    /// strategy. Primitives without a `cost`/`cost_fn` annotation count 0,
    /// so with no annotations at all this behaves exactly like plain
    /// [`DepthFirst`](crate::selection::HtnSearchStrategy::DepthFirst).
    pub fn set_cost_bounded(&mut self, enabled: bool) -> &mut Self {
        self.cost_bounded = enabled;
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
        self.plan_inner(root, state, None)
    }

    /// Decompose `root` into a [`Plan`], appending one
    /// [`DecompositionTrace`] per branch-selection decision to `trace`.
    ///
    /// Tracing is per *commitment* — one event per branch that was selected,
    /// failed its preconditions, or was backtracked past — never per
    /// precondition attempt inside the look-ahead sweep.
    pub fn plan_traced(
        &mut self,
        root: &str,
        state: &PlanState,
        trace: &mut Vec<DecompositionTrace>,
    ) -> Plan {
        self.plan_inner(root, state, Some(trace))
    }

    /// The search itself.
    fn plan_inner(
        &mut self,
        root: &str,
        state: &PlanState,
        mut trace: Option<&mut Vec<DecompositionTrace>>,
    ) -> Plan {
        let sanity_limit = self.sanity_limit;
        let mut count = 0;
        let mut stack: VecDeque<Step> = VecDeque::with_capacity(16);
        let mut decomp_stack: Vec<DecompositionFrame> = Vec::with_capacity(8);
        let mut mtr: Vec<usize> = Vec::with_capacity(8);
        let mut plan: Vec<usize> = Vec::with_capacity(8);
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
        // Rollback journal: snapshotted (deep-cloned) slot values + ops,
        // restored on backtrack down to the frame's length.
        let mut rollback = Rollback::new(self.domain.components.max_align());
        // Reusable look-ahead state clone: the sweep's lazily-created private
        // copy, reused across sweeps (`copy_from`, no re-allocation).
        let mut sweep_owned: Option<PlanState> = None;
        let mut skip = 0;
        // Cost-bounded branch-and-bound state: the accumulated cost of the
        // committed primitives (`g`), and the best complete plan found so far
        // with its cost. Both stay inert unless `cost_bounded` is set.
        let cost_bounded = self.cost_bounded;
        let mut g = 0.0f32;
        let mut best: Option<Plan> = None;
        let mut best_cost = f32::INFINITY;
        // Per-choice-point ranked branch order + resume position. The order
        // is computed once per node visit (precondition validity is constant
        // there) and snapshotted into the frame on commit, so backtracking
        // resumes down the ranked list without re-ranking — required for
        // WeightedRandom soundness.
        let mut rank_order: SmallVec<[u32; 4]> = SmallVec::new();
        let mut rank_pos: usize = 0;
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
        stack.push_back(Step::Task(root_idx, None));

        'search: loop {
            let Some(step) = stack.pop_front() else {
                // The task queue drained: the current partial plan is
                // *complete*. Under branch-and-bound, record it when it
                // strictly beats the best so far and keep searching; the
                // first complete plan is the answer otherwise.
                if cost_bounded && g < best_cost {
                    best = Some(materialize(tasks, &plan, mtr.clone()));
                    best_cost = g;
                    if backtrack(
                        &mut decomp_stack,
                        &mut plan,
                        &mut mtr,
                        &mut stack,
                        &mut rollback,
                        &mut skip,
                        &mut rank_order,
                        &mut rank_pos,
                        &mut g,
                        &mut state,
                        tasks,
                        &mut trace,
                    ) {
                        continue 'search;
                    }
                }
                break 'search;
            };
            count += 1;
            if count > sanity_limit {
                return match best {
                    Some(b) => b,
                    None => materialize(tasks, &plan, mtr.clone()),
                };
            }

            // A pending linearization retry: re-commit the same partial-order
            // method with its next topological order. The frame covering this
            // attempt was pushed by the backtrack that queued this entry, and
            // the queue/state were restored to commitment time — so this is
            // exactly the original commitment, with a different member order.
            let (current, occurrence_pin) = match step {
                Step::Task(current, pin) => (current, pin),
                Step::Linearize { task, method, lin } => {
                    let compound = match &tasks[task as usize] {
                        Task::Compound(c) => c,
                        // Defensive: only compound commitments queue retries.
                        _ => break 'search,
                    };
                    let m = &compound.methods[method as usize];
                    let SubtaskOrder::Partial { preds, .. } = &m.order else {
                        // Defensive: total methods never queue retries.
                        break 'search;
                    };
                    let Some(order) = linearize(preds, lin as usize) else {
                        // Unreachable when the baked order count is consistent
                        // with the enumeration; recover through the normal
                        // backtrack path (which exhausts cleanly).
                        if !backtrack(
                            &mut decomp_stack,
                            &mut plan,
                            &mut mtr,
                            &mut stack,
                            &mut rollback,
                            &mut skip,
                            &mut rank_order,
                            &mut rank_pos,
                            &mut g,
                            &mut state,
                            tasks,
                            &mut trace,
                        ) {
                            break 'search;
                        }
                        continue 'search;
                    };
                    // Re-commit: the method's MTR entry was removed by the
                    // backtrack, so re-record it.
                    mtr.push(method as usize);
                    if let Some(t) = trace.as_deref_mut() {
                        t.push(DecompositionTrace {
                            compound: task,
                            branch: method,
                            branch_name: m.name,
                            outcome: TraceOutcome::Selected,
                        });
                    }
                    // Push the linearized member sequence, without occurrence
                    // pins — the sweep's pins were derived for the first
                    // order; retries run unpinned (an optimization, never a
                    // soundness requirement).
                    for &pos in order.iter().rev() {
                        let sub_idx = m.subtasks[pos as usize] as usize;
                        stack.push_front(Step::Task(sub_idx, None));
                    }
                    skip = 0;
                    rank_order.clear();
                    rank_pos = 0;
                    continue 'search;
                }
            };

            let task = &tasks[current];

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
                            None => match &compound.policy {
                                // Fast path: the default declaration-order
                                // policy scans directly, no ranking setup.
                                SelectionPolicy::FirstMatch => compound.find_method(&state, skip),
                                // Rank once per node visit (precondition
                                // validity is constant there — the state
                                // only changes deeper down, and backtracking
                                // restores it), then walk the ranked list.
                                _ => {
                                    if rank_order.is_empty() {
                                        compound.rank_valid_methods(
                                            &state,
                                            plan.len() as u64,
                                            &mut rank_order,
                                        );
                                        if let Some(t) = trace.as_deref_mut() {
                                            // Every method NOT in the ranked order
                                            // failed its preconditions.
                                            let ranked: std::collections::HashSet<u32> =
                                                rank_order.iter().copied().collect();
                                            for (mi, m) in compound.methods.iter().enumerate() {
                                                if !ranked.contains(&(mi as u32)) {
                                                    t.push(DecompositionTrace {
                                                        compound: current as u32,
                                                        branch: mi as u32,
                                                        branch_name: m.name,
                                                        outcome: TraceOutcome::PrecondFailed,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    rank_order
                                        .get(rank_pos)
                                        .map(|&mi| (&compound.methods[mi as usize], mi as usize))
                                }
                            },
                        };
                        let Some((method, idx)) = eligible else {
                            // No eligible method: unwind to the most recent
                            // decomposition and try its next choice.
                            if self.fail_fast {
                                break 'search;
                            }
                            if !backtrack(
                                &mut decomp_stack,
                                &mut plan,
                                &mut mtr,
                                &mut stack,
                                &mut rollback,
                                &mut skip,
                                &mut rank_order,
                                &mut rank_pos,
                                &mut g,
                                &mut state,
                                tasks,
                                &mut trace,
                            ) {
                                break 'search;
                            }
                            continue 'search;
                        };

                        // Branch-and-bound: a commitment whose cost lower
                        // bound (accumulated cost + the method's bake-time
                        // sequence minimum) cannot strictly beat the best
                        // complete plan is pruned without recursing.
                        if cost_bounded && best.is_some() && g + method.min_cost >= best_cost {
                            if pin.is_some()
                                || matches!(compound.policy, SelectionPolicy::FirstMatch)
                            {
                                skip = idx + 1;
                            } else {
                                rank_pos += 1;
                            }
                            continue;
                        }

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
                        // subtask methods never clone the state.) Partially-
                        // ordered methods sweep their member SET: every
                        // member's writes are optimistic-unknown and no
                        // effects are applied, since the execution order is
                        // chosen by the search.
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
                                method.order.is_partial(),
                            )
                        } else {
                            Lookahead::Refine(Vec::new())
                        };
                        match verdict {
                            Lookahead::DeadEnd => {
                                // Proven doomed without recursing: try the
                                // next method at this site.
                                if pin.is_some()
                                    || matches!(compound.policy, SelectionPolicy::FirstMatch)
                                {
                                    skip = idx + 1;
                                } else {
                                    rank_pos += 1;
                                }
                                continue;
                            }
                            Lookahead::Refine(pins_found) => {
                                // Snapshot *before* the push: on backtrack the
                                // truncate must remove THIS method's entry, so
                                // the node's retry replaces it instead of
                                // appending (a stale entry would corrupt the
                                // MTR of every plan found after a backtrack).
                                let mtr_len = mtr.len();
                                mtr.push(idx);
                                if let Some(t) = trace.as_deref_mut() {
                                    t.push(DecompositionTrace {
                                        compound: current as u32,
                                        branch: idx as u32,
                                        branch_name: method.name,
                                        outcome: TraceOutcome::Selected,
                                    });
                                }
                                let frame = DecompositionFrame {
                                    task: current,
                                    plan_len: plan.len(),
                                    skip_next: idx + 1,
                                    mtr_len,
                                    pinned: pin,
                                    stack: stack.iter().copied().collect(),
                                    rollback_len: rollback.len(),
                                    // Resume position in the node's ranked
                                    // order (the order is re-derived
                                    // deterministically on revisit).
                                    rank_resume: rank_pos + 1,
                                    g_commit: g,
                                    // Partially-ordered methods retry their
                                    // next topological order (starting at 1;
                                    // order 0 is pushed below) before other
                                    // methods are offered.
                                    lin: if method.order.is_partial() { 1 } else { 0 },
                                    lin_total: match &method.order {
                                        SubtaskOrder::Partial { orders, .. } => *orders,
                                        SubtaskOrder::Total => 0,
                                    },
                                };
                                decomp_stack.push(frame);
                                // Push subtask occurrences in reverse so the
                                // first pops first, attaching each one's
                                // inevitable-refinement pin (if the sweep
                                // proved its other methods infeasible). Total-
                                // order methods run in declaration order;
                                // partially-ordered ones run in the baked
                                // first topological order (the declaration
                                // order whenever it is topological). Pins are
                                // keyed by member position, which both orders
                                // share.
                                let push_order: SmallVec<[usize; 4]> = match &method.order {
                                    SubtaskOrder::Total => (0..method.subtasks.len()).collect(),
                                    SubtaskOrder::Partial { first, .. } => {
                                        first.iter().map(|&p| p as usize).collect()
                                    }
                                };
                                for &pos in push_order.iter().rev() {
                                    let sub_idx = method.subtasks[pos] as usize;
                                    let sub_pin = pins_found
                                        .iter()
                                        .find(|(p, _)| *p == pos)
                                        .map(|&(_, m)| m as u32);
                                    stack.push_front(Step::Task(sub_idx, sub_pin));
                                }
                                skip = 0;
                                rank_order.clear();
                                rank_pos = 0;
                                continue 'search;
                            }
                        }
                    }
                }
                Task::Primitive(primitive) => {
                    if primitive.preconditions_met(&state) {
                        // Branch-and-bound: evaluate the step cost and prune
                        // the step when no completion through it can strictly
                        // beat the best complete plan (every remaining step
                        // costs ≥ 0).
                        let step_cost = if cost_bounded {
                            primitive
                                .cost
                                .as_ref()
                                .map(|f| f(&state))
                                .unwrap_or(0.0)
                                .max(0.0)
                        } else {
                            0.0
                        };
                        if cost_bounded && best.is_some() && g + step_cost >= best_cost {
                            if self.fail_fast {
                                break 'search;
                            }
                            if !backtrack(
                                &mut decomp_stack,
                                &mut plan,
                                &mut mtr,
                                &mut stack,
                                &mut rollback,
                                &mut skip,
                                &mut rank_order,
                                &mut rank_pos,
                                &mut g,
                                &mut state,
                                tasks,
                                &mut trace,
                            ) {
                                break 'search;
                            }
                            continue;
                        }
                        plan.push(current);
                        g += step_cost;
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
                    if self.fail_fast {
                        break 'search;
                    }
                    if !backtrack(
                        &mut decomp_stack,
                        &mut plan,
                        &mut mtr,
                        &mut stack,
                        &mut rollback,
                        &mut skip,
                        &mut rank_order,
                        &mut rank_pos,
                        &mut g,
                        &mut state,
                        tasks,
                        &mut trace,
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

        match best {
            Some(b) => b,
            None => materialize(tasks, &plan, mtr),
        }
    }
}

/// Unwind one decomposition frame (or, for a pinned task whose only viable
/// method failed, keep unwinding past it). Restores the append-only prefixes
/// by truncation, the scratchpad by rollback, and re-queues the frame's task
/// for its next method choice. Returns `false` when the search is exhausted.
#[allow(clippy::too_many_arguments)]
fn backtrack(
    decomp_stack: &mut Vec<DecompositionFrame>,
    plan: &mut Vec<usize>,
    mtr: &mut Vec<usize>,
    stack: &mut VecDeque<Step>,
    rollback: &mut Rollback,
    skip: &mut usize,
    rank_order: &mut SmallVec<[u32; 4]>,
    rank_pos: &mut usize,
    g: &mut f32,
    state: &mut PlanState,
    tasks: &[Task],
    trace: &mut Option<&mut Vec<DecompositionTrace>>,
) -> bool {
    loop {
        match decomp_stack.pop() {
            Some(frame) => {
                plan.truncate(frame.plan_len);
                mtr.truncate(frame.mtr_len);
                // Restore the scratchpad: undo every effect applied since the
                // frame committed (newest first).
                rollback.restore_to(frame.rollback_len, state);
                // Restore the accumulated cost to its commitment value (the
                // truncated primitives are exactly the ones added since).
                *g = frame.g_commit;
                // Restore the queue to its state at commitment time: the
                // failed subtree's remnants go, the suffix (with its
                // occurrence pins) comes back.
                stack.clear();
                stack.extend(frame.stack.iter().copied());
                // Partially-ordered method with pending linearizations: retry
                // the SAME method with its next topological order before other
                // methods are offered. The state/plan/mtr/queue above were
                // restored to commitment time, so the retry starts exactly
                // where the original commitment did. (A pinned task's other
                // methods are still infeasible — only its linearizations are
                // retried.)
                if frame.lin > 0 && frame.lin < frame.lin_total {
                    if let Some(t) = trace.as_deref_mut() {
                        let branch = frame.skip_next.saturating_sub(1);
                        let name = match &tasks[frame.task] {
                            Task::Compound(c) => c.methods.get(branch).and_then(|m| m.name),
                            _ => None,
                        };
                        t.push(DecompositionTrace {
                            compound: frame.task as u32,
                            branch: branch as u32,
                            branch_name: name,
                            outcome: TraceOutcome::Backtracked,
                        });
                    }
                    let lin_task = frame.task as u32;
                    let lin_method = frame.skip_next.saturating_sub(1) as u32;
                    let lin_try = frame.lin;
                    // The replacement frame covers the next attempt: identical
                    // restore points, one linearization further.
                    decomp_stack.push(DecompositionFrame {
                        lin: lin_try + 1,
                        ..frame
                    });
                    stack.push_front(Step::Linearize {
                        task: lin_task,
                        method: lin_method,
                        lin: lin_try,
                    });
                    return true;
                }
                if frame.pinned.is_some() {
                    // This task was pinned to a single method and it failed:
                    // all alternatives were proven infeasible at pin time, so
                    // keep unwinding instead of retrying them.
                    continue;
                }
                *skip = frame.skip_next;
                // Resume this node's ranked list past the committed choice:
                // the order is re-derived on the node's next visit (all
                // policies are deterministic per (state, nonce), and the
                // nonce — the restored plan length — matches the original
                // visit, so the resumed order is identical).
                rank_order.clear();
                *rank_pos = frame.rank_resume;
                if let Some(t) = trace.as_deref_mut() {
                    // The branch the unwound frame was committed to.
                    let branch = frame.skip_next.saturating_sub(1);
                    let name = match &tasks[frame.task] {
                        Task::Compound(c) => c.methods.get(branch).and_then(|m| m.name),
                        _ => None,
                    };
                    t.push(DecompositionTrace {
                        compound: frame.task as u32,
                        branch: branch as u32,
                        branch_name: name,
                        outcome: TraceOutcome::Backtracked,
                    });
                }
                stack.push_front(Step::Task(frame.task, frame.pinned));
                return true;
            }
            None => return false,
        }
    }
}

/// Convert a plan of (narrow) task indices into the compiled step program:
/// contiguous `u32` task indices plus the parallel interned-name list.
fn materialize(tasks: &[Task], plan: &[usize], mtr: Vec<usize>) -> Plan {
    Plan {
        steps: plan.iter().map(|&i| i as u32).collect(),
        names: plan.iter().map(|&i| tasks[i].name().into()).collect(),
        mtr: Mtr(mtr),
    }
}
