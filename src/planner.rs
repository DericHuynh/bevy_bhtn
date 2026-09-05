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
//! # Paused plans (PausePlan)
//!
//! A method body can declare a
//! [`pause marker`](crate::tasks::MethodBuilder::pause_plan) between its
//! members. When the search reaches the marker while compiling, the plan is
//! **truncated**: everything decomposed so far is the compiled prefix
//! ([`PlanStatus::Paused`]), and the work still queued behind the marker is
//! recorded in [`Plan::resume`] — the remaining task occurrences (chained
//! markers included) plus the MTR as it stood at the pause.
//! [`HtnPlanner::resume`] continues the decomposition from that point against
//! the state the world is in *then*: the executed prefix is committed history
//! (never re-decomposed, and the resumed search has no frames above the seed
//! to backtrack into — an unsatisfiable remainder is `NoPlan`). The look-ahead
//! sweep proves only the pre-pause prefix, so far-future steps are never
//! optimistically validated against state the executed prefix will have
//! replaced, and planner work per plan is bounded by the leg between markers
//! instead of the whole horizon.
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
use std::any::TypeId;

use ustr::Ustr;

use crate::domain::SelectionPolicy;
use crate::error::{HtnError, HtnResult};
use crate::selection::{DecompositionTrace, HtnSearchStrategy, LookaheadMode, TraceOutcome};
use crate::tasks::TaskFn;

use crate::domain::HtnDomain;
use crate::domain::Task;
use crate::lookahead::{self, Lookahead};
use crate::order::{linearize, SubtaskOrder};
use crate::state::{PlanState, Slot};

/// A completed forward plan: a **compiled step program** plus the MTR.
///
/// Whether a compiled [`Plan`] is the finished product of a completed
/// decomposition, or the best prefix cut out of a search that stopped early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanStatus {
    /// The search ran to completion: the plan is final and reaches a terminal
    /// state of the decomposition (possibly empty — a root whose only methods
    /// are empty terminal branches). A root with *no* valid decomposition is
    /// reported as [`HtnError::NoPlan`] by the planner, never as an empty
    /// `Complete` plan.
    #[default]
    Complete,
    /// The search stopped early (sanity budget exhausted or fail-fast): this
    /// is the best partial plan found so far and may not reach the goal.
    /// Raise the budget (`HtnPlanner::set_sanity_limit`) or check the domain
    /// for unbounded recursion (the `terminating` summary flags it).
    Partial,
    /// The search stopped at a [`pause marker`](crate::tasks::MethodBuilder::pause_plan)
    /// — a deliberate authoring boundary, not a failure. The compiled steps
    /// are the leg the author committed to planning; the work still queued
    /// behind the marker is the [`Plan::resume`] point, and decomposition
    /// **resumes from it** ([`HtnPlanner::resume`]) once the prefix has run.
    /// Unlike `Partial`, nothing was cut short: the plan ends exactly where
    /// the author drew the line.
    Paused,
}

/// One entry of a paused plan's resume queue: a task still to decompose, or
/// a pause marker that re-truncates the resumed plan (chained legs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeStep {
    /// A task occurrence queued behind the pause, to be decomposed when
    /// decomposition resumes. Look-ahead occurrence pins are deliberately not
    /// carried: they were derived against the pre-pause state, which the
    /// executed prefix has replaced — the resumed search re-sweeps.
    Task(u32),
    /// A pause marker queued behind the first one (chained legs): when the
    /// resumed search reaches it, the resumed plan truncates again and a new
    /// resume point is recorded.
    Pause,
}

/// The resume point of a plan truncated by a
/// [`pause marker`](crate::tasks::MethodBuilder::pause_plan): the work still
/// queued behind the marker, plus the MTR (method traversal record) as it
/// stood at the pause. Passed back to [`HtnPlanner::resume`] once the prefix
/// has executed — decomposition continues from the pause against the state
/// the world is in *then*, without re-decomposing (or being able to
/// backtrack past) the already-executed prefix.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResumePoint {
    /// The remaining decomposition work in execution order: task occurrences
    /// still queued behind the pause, interleaved with any chained pause
    /// markers. Task indices address [`HtnDomain::tasks`](crate::domain::HtnDomain).
    pub tasks: Vec<ResumeStep>,
    /// The MTR as it stood at the pause: the method choices from the root
    /// down to (and including) the method holding the marker. Seeded into the
    /// resumed plan's MTR, so the resumed plan records the full decomposition
    /// path; backtracking within the resumed search never truncates below it.
    pub mtr: Vec<usize>,
}

/// How many consecutive non-refuting sweeps a method tolerates under
/// [`LookaheadMode::Adaptive`] before its sweep is disabled for the rest of
/// the plan (a refutation resets the streak immediately).
const ADAPTIVE_SWEEP_TRIALS: u32 = 2;

/// What a planning call starts from: a task function (its `TypeId` resolves
/// through the baked type index) or a task index into
/// [`HtnDomain::tasks`](crate::domain::HtnDomain). Constructed implicitly —
/// pass either the fn item or the `usize` wherever a root is expected
/// ([`HtnPlanner::plan`]/[`HtnPlanner::plan_traced`]). The index form exists
/// for callers holding a baked domain without its function types (the ECS
/// driver, test beds) and is the only way to address GTN-synthesized tasks,
/// which have no function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanRoot {
    /// Start from this task function (resolved by `TypeId`). The function's
    /// `type_name` travels along for error diagnostics only.
    Fn(TypeId, &'static str),
    /// Start from this task index directly.
    Index(usize),
}

impl<F: TaskFn> From<F> for PlanRoot {
    fn from(f: F) -> Self {
        Self::Fn(f.task_type_id(), std::any::type_name::<F>())
    }
}

impl From<usize> for PlanRoot {
    fn from(idx: usize) -> Self {
        Self::Index(idx)
    }
}

/// Where one built-in search starts: from a registered root task (the
/// [`HtnPlanner::plan`] entry) or from a paused plan's resume point (the
/// [`HtnPlanner::resume`] entry — the work a pause marker queued, seeded
/// with the committed MTR prefix).
enum Start<'a> {
    Root(usize),
    Resume(&'a ResumePoint),
}

/// The compiled plan a planner returns: a flat step program over the domain's
/// task indices.
///
/// `steps` holds task indices into [`HtnDomain::tasks`](crate::domain::HtnDomain)
/// in execution order, so executing a plan is a flat array walk — the driver
/// indexes the baked task array directly, with no name lookups and no string
/// comparisons on the hot path. The interned names are kept in parallel for
/// display and introspection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    /// Baked step program: task indices, in execution order.
    pub steps: Vec<u32>,
    /// Interned task names, parallel to [`Self::steps`] (display/introspection;
    /// execution reads [`Self::steps`] only).
    pub names: Vec<Ustr>,
    /// The MTR (Method Traversal Record): the index of the chosen method at
    /// each decomposition level. Used to compare plans by priority (lower
    /// index = higher priority) and for debugging; exposed as a slice by
    /// [`Self::mtr`].
    pub mtr: Vec<usize>,
    /// Whether the search finished (`Complete`) or was cut short (`Partial`).
    pub status: PlanStatus,
    /// The remaining decomposition work when the plan was truncated by a
    /// pause marker ([`PlanStatus::Paused`]): pass it to
    /// [`HtnPlanner::resume`] once the compiled steps have executed.
    /// `None` on `Complete` and `Partial` plans.
    pub resume: Option<ResumePoint>,
}

impl Plan {
    /// The ordered primitive task names (interned handles; deref to `&str`).
    pub fn task_names(&self) -> &[Ustr] {
        &self.names
    }

    /// Whether the search ran to completion — the plan is final. `false`
    /// means the sanity budget or fail-fast cut the search short and this is
    /// the best partial plan found (it may not reach the goal).
    pub fn is_complete(&self) -> bool {
        self.status == PlanStatus::Complete
    }

    /// Whether the search was cut short (see [`Self::is_complete`]).
    pub fn is_partial(&self) -> bool {
        !self.is_complete()
    }

    /// Whether the plan was truncated by a pause marker (see
    /// [`PlanStatus::Paused`]): the compiled steps are the leg the author
    /// committed to planning, and [`Self::resume`] holds the work still
    /// queued behind the marker.
    pub fn is_paused(&self) -> bool {
        self.status == PlanStatus::Paused
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

    /// The MTR: the index of the chosen method at each decomposition level
    /// (lower = higher priority). Forward-only — backward plans and
    /// custom-searcher plans have an empty MTR.
    pub fn mtr(&self) -> &[usize] {
        &self.mtr
    }

    /// Order two plans by MTR priority (lower first).
    pub fn is_preferred_over(&self, other: &Self) -> bool {
        for (a, b) in self.mtr.iter().zip(other.mtr.iter()) {
            match a.cmp(b) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord == std::cmp::Ordering::Less,
            }
        }
        // Prefer the shorter MTR when they share a prefix.
        self.mtr.len() < other.mtr.len()
    }
}

/// One entry of the search's task queue: a task occurrence (with its
/// optional look-ahead pin), a pending linearization retry of a
/// partially-ordered method, or a [`pause marker`](crate::tasks::MethodBuilder::pause_plan).
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
    /// A pause marker between a method's members: popping it truncates the
    /// compiled plan — everything decomposed so far is the prefix, everything
    /// still queued becomes the plan's resume point. Only ever queued by
    /// commitments of methods that carry pause markers (total-order only;
    /// rejected on partial branches at bake).
    Pause,
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
struct Rollback<'a> {
    values: Journal,
    /// `(slot, journal offset)` per snapshot — the offset is stored because
    /// alignment padding between entries makes offsets not derivable from
    /// the current length.
    ops: Vec<(usize, usize)>,
    /// The baked slot table: releases journal copies the search never
    /// restored (a successful plan or a sanity-limit exit leaves snapshots on
    /// the journal — the arena's `Drop` only deallocates bytes).
    slots: &'a [Slot],
}

impl<'a> Rollback<'a> {
    fn new(max_align: usize, slots: &'a [Slot]) -> Self {
        Self {
            values: Journal::new(max_align),
            ops: Vec::with_capacity(16),
            slots,
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

impl Drop for Rollback<'_> {
    fn drop(&mut self) {
        // Snapshots the search never restored (successful-plan and
        // sanity-limit exits) are still owned by the journal: drop each
        // through its slot's baked dropper before the arena deallocates.
        // Fully-restored searches leave `ops` empty, so this is a no-op there.
        for &(idx, start) in &self.ops {
            unsafe { (self.slots[idx].drop_fn)(self.values.ptr_at(start)) };
        }
    }
}

/// Planner state used during one decomposition site for backtracking.
///
/// Stores **task indices** (not names) so frames stay tiny and copy-free.
/// `plan` and `mtr` are append-only within one `plan()` call — recursive
/// descent (primitives and method indices are only ever `push`ed). So instead
/// of deep-cloning both `Vec`s into every frame (which costs ~2n allocations
/// and O(n) copies per recursion level — catastrophic when a domain recursively
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
    /// 2–5-subtask case (allocation-free commitments). (A persistent/
    /// structurally-shared queue would make this O(1) — a known future
    /// optimization; a bare length is NOT enough, because a completed
    /// subtree's frame can still be backtracked into after the search has
    /// consumed part of the suffix.)
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
    /// The look-ahead gating mode (default: [`Always`](LookaheadMode::Always)).
    lookahead: LookaheadMode,
    /// Decomposition-step budget before the best partial plan is returned
    /// (default: 100).
    sanity_limit: usize,
    /// The active search strategy (default: [`DepthFirst`](HtnSearchStrategy::DepthFirst)).
    /// Encodes the mode as one value — the old independent `fail_fast`/
    /// `cost_bounded` bools could represent combinations (both on) whose
    /// behavior was undefined.
    strategy: HtnSearchStrategy,
    /// Flat method-index base per task (`method_base[task] + method_idx`
    /// addresses a method's sweep-streak slot). Built lazily on the first
    /// [`LookaheadMode::Adaptive`] plan — the other modes never touch it, so
    /// planner construction stays allocation-free.
    method_base: Vec<usize>,
    /// Per-method consecutive non-refuting sweep count under
    /// [`LookaheadMode::Adaptive`] — reset at every `plan_inner` entry, so
    /// gating is deterministic per plan and re-learns on each replan.
    sweep_streaks: Vec<u32>,
}

impl<'a> HtnPlanner<'a> {
    /// Create a planner bound to a domain.
    pub fn new(domain: &'a HtnDomain) -> Self {
        Self {
            domain,
            lookahead: LookaheadMode::default(),
            sanity_limit: 100,
            strategy: HtnSearchStrategy::default(),
            method_base: Vec::new(),
            sweep_streaks: Vec::new(),
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
        self.lookahead = if enabled {
            LookaheadMode::Always
        } else {
            LookaheadMode::Off
        };
        self
    }

    /// Set the look-ahead gating mode (see [`LookaheadMode`]).
    pub fn set_lookahead_mode(&mut self, mode: LookaheadMode) -> &mut Self {
        self.lookahead = mode;
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

    /// Set the search strategy — the single knob for the planner's mode:
    /// [`DepthFirst`](HtnSearchStrategy::DepthFirst) (default, full MTR
    /// backtracking), [`DepthFirstFailFast`](HtnSearchStrategy::DepthFirstFailFast)
    /// (abandon on first downstream failure),
    /// [`CostBounded`](HtnSearchStrategy::CostBounded) (branch-and-bound over
    /// accumulated primitive cost), or [`Custom`](HtnSearchStrategy::Custom)
    /// (a caller-supplied [`Searcher`](crate::selection::Searcher) that bypasses
    /// the built-in machinery —
    /// its statistics live in the strategy object, and lookahead/sanity do
    /// not apply to it). Replacing a strategy replaces it entirely: the old
    /// independent bools could describe combinations (fail-fast **and**
    /// cost-bounded) whose behavior was undefined.
    pub fn set_strategy(&mut self, strategy: HtnSearchStrategy) -> &mut Self {
        self.strategy = strategy;
        self
    }

    /// Decompose `root` into a [`Plan`]. Never errors: on failure it returns
    /// the best partial plan found (with an empty task list if nothing was
    /// decomposable — including an unregistered root function or an
    /// out-of-bounds index). Check [`Plan::status`] to tell a finished
    /// decomposition ([`PlanStatus::Complete`]) from one the sanity budget or
    /// fail-fast cut short ([`PlanStatus::Partial`]) — a partial plan may not
    /// reach the goal — and from one truncated by a
    /// [`pause marker`](crate::tasks::MethodBuilder::pause_plan)
    /// ([`PlanStatus::Paused`], with the remaining work in [`Plan::resume`]
    /// for [`Self::resume`]).
    ///
    /// `root` is a task function (passed by value — fn items are zero-sized,
    /// so turbofish is impossible; its `TypeId` resolves through the baked
    /// type index; names are display-only) or a task index into
    /// [`HtnDomain::tasks`](crate::domain::HtnDomain) — the form callers
    /// holding a baked domain without its function types use (the driver,
    /// test beds, and the only way to address GTN-synthesized tasks).
    ///
    /// `state` is only read: the planner works on its own clone of the
    /// scratchpad.
    ///
    /// Before committing each method choice, a look-ahead sweep
    /// ([`lookahead`]) proves the remaining sequence can possibly succeed;
    /// doomed methods are skipped at the frame and inevitable refinements
    /// (unique surviving methods) are pinned for when the planner reaches them.
    ///
    /// # Errors
    ///
    /// - [`HtnError::UnregisteredTask`] — the root function was never recorded
    ///   in this domain (or the index is out of bounds).
    /// - [`HtnError::NoPlan`] — the search space was exhausted with no
    ///   complete decomposition: the domain genuinely cannot solve this root
    ///   in this state. Symmetric with [`BackPlanner`](crate::back_planner)'s
    ///   error. (An empty *successful* plan — a root whose decomposition is
    ///   legitimately empty — is `Ok` with an empty `Complete` plan; the two
    ///   are never conflated.) Budget truncation is *not* an error: a
    ///   sanity-limited search returns `Ok` with a `Partial` plan.
    pub fn plan(&mut self, root: impl Into<PlanRoot>, state: &PlanState) -> HtnResult<Plan> {
        let idx = self.resolve_root(root)?;
        self.plan_inner(Start::Root(idx), state, None)
    }

    /// Decompose `root` into a [`Plan`], appending one [`DecompositionTrace`]
    /// per branch-selection decision to `trace` (see [`Self::plan`]).
    ///
    /// Tracing is per *commitment* — one event per branch that was selected,
    /// failed its preconditions, or was backtracked past — never per
    /// precondition attempt inside the look-ahead sweep. `Custom` strategies
    /// emit nothing: they own their search.
    pub fn plan_traced(
        &mut self,
        root: impl Into<PlanRoot>,
        state: &PlanState,
        trace: &mut Vec<DecompositionTrace>,
    ) -> HtnResult<Plan> {
        let idx = self.resolve_root(root)?;
        self.plan_inner(Start::Root(idx), state, Some(trace))
    }

    /// Resume a decomposition that a
    /// [`pause marker`](crate::tasks::MethodBuilder::pause_plan) truncated:
    /// decompose the work `point` still queues (the plan's
    /// [`Plan::resume`]) against `state` — the state the world is in *now*,
    /// after the paused plan's compiled steps have executed.
    ///
    /// The already-executed prefix is never re-decomposed and can never be
    /// backtracked into: its methods are committed (they are `point.mtr`,
    /// seeded into the resumed plan's MTR), so the resumed search explores
    /// only the work behind the pause. If that work has no valid
    /// decomposition in the current state, this is
    /// [`HtnError::NoPlan`] — the caller's recovery is the same as any other
    /// failed plan: replan from the root against reality.
    ///
    /// # Errors
    ///
    /// - [`HtnError::UnregisteredTask`] — the resume point carries an
    ///   out-of-bounds task index (a stale point against a rebuilt domain).
    /// - [`HtnError::NoPlan`] — the remaining work has no valid
    ///   decomposition in `state`.
    pub fn resume(&mut self, point: &ResumePoint, state: &PlanState) -> HtnResult<Plan> {
        for step in &point.tasks {
            if let ResumeStep::Task(idx) = *step {
                if idx as usize >= self.domain.tasks.len() {
                    return Err(HtnError::UnregisteredTask {
                        type_name: format!("<task index {idx}>"),
                    });
                }
            }
        }
        self.plan_inner(Start::Resume(point), state, None)
    }

    /// Resolve a plan root to a task index. Errors carry the fn's `type_name`
    /// (captured at conversion) or the offending index for diagnosis.
    fn resolve_root(&self, root: impl Into<PlanRoot>) -> HtnResult<usize> {
        match root.into() {
            PlanRoot::Fn(tid, name) => {
                self.domain
                    .task_index_by_type(tid)
                    .ok_or(HtnError::UnregisteredTask {
                        type_name: name.to_string(),
                    })
            }
            PlanRoot::Index(idx) if idx < self.domain.tasks.len() => Ok(idx),
            PlanRoot::Index(idx) => Err(HtnError::UnregisteredTask {
                type_name: format!("<task index {idx}>"),
            }),
        }
    }

    /// The search itself. `start` is either a registered root task or a
    /// paused plan's resume point; a resume seeds the queue with the paused
    /// work (occurrence pins dropped — they were derived against the
    /// pre-pause state) and the MTR with the committed method chain, and the
    /// search proceeds identically from there. With no committed frames above
    /// the seed, backtracking past the pause point (whose prefix already
    /// executed) is structurally impossible: exhaustion is `NoPlan`.
    fn plan_inner(
        &mut self,
        start: Start<'_>,
        state: &PlanState,
        mut trace: Option<&mut Vec<DecompositionTrace>>,
    ) -> HtnResult<Plan> {
        // Custom searchers bypass the built-in machinery entirely: they own
        // their search (and their statistics); lookahead/sanity do not apply.
        if let HtnSearchStrategy::Custom(searcher) = &self.strategy {
            return searcher.search(self.domain, state).ok_or(HtnError::NoPlan);
        }
        // Adaptive sweep streaks are per-plan: gating decisions are
        // deterministic for a given (domain, state) and re-learn on replan.
        // The stats are built lazily — only Adaptive pays for them.
        let adaptive = matches!(self.lookahead, LookaheadMode::Adaptive);
        if adaptive && self.sweep_streaks.is_empty() {
            let mut base = Vec::with_capacity(self.domain.tasks.len() + 1);
            let mut running = 0usize;
            for task in &self.domain.tasks {
                base.push(running);
                if let crate::domain::Task::Compound(c) = task {
                    running += c.methods.len();
                }
            }
            base.push(running);
            self.method_base = base;
            self.sweep_streaks = vec![0u32; running];
        }
        if adaptive {
            self.sweep_streaks.fill(0);
        }
        // The strategy enum encodes the valid combinations the old independent
        // bools left undefined (fail-fast + cost-bounded together).
        let (fail_fast, cost_bounded) = match self.strategy {
            HtnSearchStrategy::DepthFirstFailFast => (true, false),
            HtnSearchStrategy::CostBounded => (false, true),
            _ => (false, false),
        };
        let sanity_limit = self.sanity_limit;
        let mut count = 0;
        // `Complete` unless the search stops early (fail-fast or a defensive
        // exit); the sanity-limit return sets its own status.
        let mut status = PlanStatus::Complete;
        // Set when the search backtracks past the root: the space is provably
        // exhausted with no complete decomposition (an empty `Complete` plan
        // from a legitimate empty decomposition is the drain-exit below, and
        // the two must never be conflated — see `plan`'s error docs).
        let mut exhausted = false;
        // The resume point recorded when a pause marker pops (the pause exit
        // below materializes it). `None` everywhere else.
        let mut paused_resume: Option<ResumePoint> = None;
        let mut stack: VecDeque<Step> = VecDeque::with_capacity(16);
        let mut decomp_stack: Vec<DecompositionFrame> = Vec::with_capacity(8);
        let mut mtr: Vec<usize> = Vec::with_capacity(8);
        let mut plan: Vec<usize> = Vec::with_capacity(8);
        // Reusable look-ahead scratch: the sweep's "unknown components" overlay
        // and its inevitable-refinement output, cleared per sweep.
        let mut sweep_unknown = crate::state::FieldSet::new(self.domain.components.len());
        let mut sweep_pins: Vec<(usize, usize)> = Vec::with_capacity(4);
        // Reusable survivor buffer for the sweep's single-pass compound check.
        let mut sweep_surviving: Vec<usize> = Vec::with_capacity(8);
        // Reusable scratch for the commitment's resolved subtask list and the
        // sweep's sequence view of it.
        let mut resolved_buf: Vec<(usize, usize)> = Vec::with_capacity(8);
        let mut seq_buf: Vec<usize> = Vec::with_capacity(8);
        // Rollback journal: snapshotted (deep-cloned) slot values + ops,
        // restored on backtrack down to the frame's length.
        let mut rollback = Rollback::new(
            self.domain.components.max_align(),
            self.domain.components.slots(),
        );
        // Reusable look-ahead state clone: the sweep's lazily-created private
        // copy, reused across sweeps (`copy_from`, no re-allocation).
        let mut sweep_owned: Option<PlanState> = None;
        let mut skip = 0;
        // Cost-bounded branch-and-bound state: the accumulated cost of the
        // committed primitives (`g`), and the best complete plan found so far
        // with its cost. Both stay inert unless the strategy is CostBounded.
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

        // Seed the queue: a root search starts from the root task; a resume
        // starts from the paused work in execution order (chained pause
        // markers re-enter the queue with it), with the committed MTR prefix
        // already recorded.
        match start {
            Start::Root(root) => stack.push_front(Step::Task(root, None)),
            Start::Resume(point) => {
                for step in point.tasks.iter().rev() {
                    stack.push_front(match step {
                        ResumeStep::Task(idx) => Step::Task(*idx as usize, None),
                        ResumeStep::Pause => Step::Pause,
                    });
                }
                mtr.extend_from_slice(&point.mtr);
            }
        }

        'search: loop {
            let Some(step) = stack.pop_front() else {
                // The task queue drained: the current partial plan is
                // *complete*. Under branch-and-bound, record it when it
                // strictly beats the best so far and keep searching; the
                // first complete plan is the answer otherwise.
                if cost_bounded && g < best_cost {
                    best = Some(materialize(
                        tasks,
                        &plan,
                        mtr.clone(),
                        PlanStatus::Complete,
                        None,
                    ));
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
                // Budget exhausted: the recorded best (a complete plan, if any)
                // or the current prefix — the search was cut short, not
                // exhausted, so this stays a `Partial` `Ok` (not `NoPlan`).
                return Ok(match best {
                    Some(b) => b,
                    None => materialize(tasks, &plan, mtr.clone(), PlanStatus::Partial, None),
                });
            }

            // A pending linearization retry: re-commit the same partial-order
            // method with its next topological order. The frame covering this
            // attempt was pushed by the backtrack that queued this entry, and
            // the queue/state were restored to commitment time — so this is
            // exactly the original commitment, with a different member order.
            let (current, occurrence_pin) = match step {
                Step::Task(current, pin) => (current, pin),
                Step::Pause => {
                    // A pause marker popped: the compiled plan ends here.
                    // Everything still queued is the resume point's work —
                    // front-to-back, chained pause markers included (the
                    // resumed search re-truncates at them). A `Linearize`
                    // entry cannot sit behind a pause (each is pushed by
                    // `backtrack` and popped back-to-back, before anything
                    // else can leave the queue), so the queue holds only
                    // Task/Pause entries here.
                    let mut left: Vec<ResumeStep> = Vec::with_capacity(stack.len());
                    let mut live = true;
                    for entry in &stack {
                        match entry {
                            Step::Task(idx, _) => left.push(ResumeStep::Task(*idx as u32)),
                            Step::Pause => left.push(ResumeStep::Pause),
                            // Unreachable by construction (see above):
                            // degrade to a budget-style partial — the driver
                            // replans from the root — rather than trusting
                            // an impossible queue shape.
                            Step::Linearize { .. } => {
                                live = false;
                                status = PlanStatus::Partial;
                                break;
                            }
                        }
                    }
                    let has_work = live && left.iter().any(|s| matches!(s, ResumeStep::Task(_)));
                    if has_work {
                        if cost_bounded && best.is_some() {
                            // Branch-and-bound already holds a complete
                            // plan; it stays the answer (a pause only
                            // truncates the branch it fires on).
                            break 'search;
                        }
                        status = PlanStatus::Paused;
                        paused_resume = Some(ResumePoint {
                            tasks: left,
                            mtr: mtr.clone(),
                        });
                    }
                    // No work queued behind the pause (or a complete plan is
                    // already held): the pause was vacuous — the prefix (or
                    // `best`) is the whole answer.
                    break 'search;
                }
                Step::Linearize { task, method, lin } => {
                    let compound = match &tasks[task as usize] {
                        Task::Compound(c) => c,
                        // Defensive: only compound commitments queue retries.
                        // The current prefix is not a finished decomposition.
                        _ => {
                            status = PlanStatus::Partial;
                            break 'search;
                        }
                    };
                    let m = &compound.methods[method as usize];
                    let SubtaskOrder::Partial { preds, .. } = &m.order else {
                        // Defensive: total methods never queue retries.
                        status = PlanStatus::Partial;
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
                            exhausted = true;
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
                            if fail_fast {
                                status = PlanStatus::Partial;
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
                                exhausted = true;
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
                        // A pause marker bounds the leg: the sweep proves
                        // only the PRE-pause prefix. The far side of the
                        // marker is not planned in this plan — optimistically
                        // validating it is exactly what the marker forbids
                        // (its preconditions depend on state that the
                        // executed prefix will have replaced). Pins only
                        // exist for swept positions, so post-pause members
                        // queue unpinned.
                        let sweep_end = method
                            .pause_positions
                            .first()
                            .map_or(resolved_buf.len(), |&p| p as usize)
                            .min(resolved_buf.len());
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
                        // Adaptive only: flat slot of this method's sweep
                        // streak (`None` in the other modes — no stats, no
                        // per-commitment bookkeeping).
                        let flat_method = if adaptive {
                            Some(self.method_base[current] + idx)
                        } else {
                            None
                        };
                        // Adaptive tiers: Full sweeps (with pins) until a
                        // method proves sweep-useless, cheap refutation-only
                        // sweeps while it is on probation, no sweep where the
                        // sweep provably duplicates the next queue pop.
                        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
                        enum SweepTier {
                            Full,
                            RefutationOnly,
                            Skip,
                        }
                        let sweep_tier = if sweep_end == 0 {
                            // The pause sits before every member: this
                            // method contributes nothing to the compiled
                            // plan — nothing to sweep.
                            SweepTier::Skip
                        } else {
                            match self.lookahead {
                                LookaheadMode::Always => SweepTier::Full,
                                LookaheadMode::Off => SweepTier::Skip,
                                LookaheadMode::Adaptive => {
                                    // Tier 1 — shape: a single-subtask,
                                    // totally-ordered method with a
                                    // terminating step duplicates the next queue pop's
                                    // precondition check exactly.
                                    let shape_skip = method.subtasks.len() == 1
                                        && !method.order.is_partial()
                                        && self.domain.summaries[method.subtasks[0] as usize]
                                            .min_yield
                                            != usize::MAX;
                                    if shape_skip {
                                        SweepTier::Skip
                                    } else if self.sweep_streaks
                                        [flat_method.expect("adaptive stats built")]
                                        >= ADAPTIVE_SWEEP_TRIALS
                                    {
                                        // Tier 3 — track record: swept
                                        // `ADAPTIVE_SWEEP_TRIALS` times in a row
                                        // without one refutation → refutation-only
                                        // for the rest of THIS plan (a refutation
                                        // resets the streak).
                                        SweepTier::RefutationOnly
                                    } else {
                                        // Tier 2 — on probation: full sweep.
                                        SweepTier::Full
                                    }
                                }
                            }
                        };
                        let verdict = match sweep_tier {
                            SweepTier::Skip => Lookahead::Refine,
                            SweepTier::RefutationOnly => {
                                seq_buf.clear();
                                seq_buf.extend(resolved_buf[..sweep_end].iter().map(|&(_, idx)| idx));
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
                                    lookahead::SweepDepth::RefutationOnly,
                                )
                            }
                            SweepTier::Full => {
                                seq_buf.clear();
                                seq_buf.extend(resolved_buf[..sweep_end].iter().map(|&(_, idx)| idx));
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
                                    lookahead::SweepDepth::Full,
                                )
                            }
                        };
                        match verdict {
                            Lookahead::DeadEnd => {
                                // Proven doomed without recursing: try the
                                // next method at this site. The streak
                                // resets — this method's sweep just paid
                                // for itself.
                                if let Some(flat) = flat_method {
                                    self.sweep_streaks[flat] = 0;
                                }
                                if pin.is_some()
                                    || matches!(compound.policy, SelectionPolicy::FirstMatch)
                                {
                                    skip = idx + 1;
                                } else {
                                    rank_pos += 1;
                                }
                                continue;
                            }
                            Lookahead::Refine => {
                                // The sweep left the inevitable refinements in
                                // `sweep_pins` (scratch discipline — no
                                // ownership handoff).
                                if let Some(flat) = flat_method {
                                    self.sweep_streaks[flat] += 1;
                                }
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
                                // share. (The stack pops last-pushed-first,
                                // so reverse iteration + push = forward
                                // execution order; no intermediate buffer.)
                                match &method.order {
                                    SubtaskOrder::Total => {
                                        // Pause markers queue as pseudo-entries:
                                        // a pause declared before member `pos`
                                        // pops right after every member before
                                        // it; a pause at `subtasks.len()` —
                                        // after the last member — pops after the
                                        // whole sequence (pushed first, since the
                                        // stack pops last-pushed-first).
                                        let pauses = &method.pause_positions;
                                        if pauses.contains(&(method.subtasks.len() as u32)) {
                                            stack.push_front(Step::Pause);
                                        }
                                        for pos in (0..method.subtasks.len()).rev() {
                                            let sub_idx = method.subtasks[pos] as usize;
                                            // Skipped sweeps leave `sweep_pins`
                                            // holding the LAST swept method's
                                            // verdicts — attaching them here
                                            // would pin this method's
                                            // occurrences to another method's
                                            // refinements (exhausting nodes
                                            // wrongly). Only full sweeps
                                            // produce pins.
                                            let sub_pin = if sweep_tier != SweepTier::Full {
                                                None
                                            } else {
                                                sweep_pins
                                                    .iter()
                                                    .find(|(p, _)| *p == pos)
                                                    .map(|&(_, m)| m as u32)
                                            };
                                            stack.push_front(Step::Task(sub_idx, sub_pin));
                                            // Pushed after the member, so it pops
                                            // *before* it: the pause before
                                            // member `pos` pops right after
                                            // member `pos - 1`.
                                            if pauses.contains(&(pos as u32)) {
                                                stack.push_front(Step::Pause);
                                            }
                                        }
                                    }
                                    SubtaskOrder::Partial { first, .. } => {
                                        for &pos in first.iter().rev() {
                                            let sub_idx = method.subtasks[pos as usize] as usize;
                                            let sub_pin = if sweep_tier != SweepTier::Full {
                                                None
                                            } else {
                                                sweep_pins
                                                    .iter()
                                                    .find(|(p, _)| *p == pos as usize)
                                                    .map(|&(_, m)| m as u32)
                                            };
                                            stack.push_front(Step::Task(sub_idx, sub_pin));
                                        }
                                    }
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
                            if fail_fast {
                                status = PlanStatus::Partial;
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
                                exhausted = true;
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
                    if fail_fast {
                        status = PlanStatus::Partial;
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
                        exhausted = true;
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
            Some(b) => Ok(b),
            // Backtracked past the root: no complete decomposition exists.
            // Anything else breaking out early already set `Partial` and
            // carries the best prefix as a non-error result.
            None if exhausted => Err(HtnError::NoPlan),
            None => Ok(materialize(tasks, &plan, mtr, status, paused_resume.take())),
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
                // occurrence pins) comes back. The snapshot copy is load-
                // bearing: a completed subtree's frame can be backtracked
                // into after the search consumed part of the suffix, and a
                // bare length cannot restore what was already popped.
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
fn materialize(
    tasks: &[Task],
    plan: &[usize],
    mtr: Vec<usize>,
    status: PlanStatus,
    resume: Option<ResumePoint>,
) -> Plan {
    Plan {
        steps: plan.iter().map(|&i| i as u32).collect(),
        names: plan.iter().map(|&i| tasks[i].name().into()).collect(),
        mtr,
        status,
        resume,
    }
}
