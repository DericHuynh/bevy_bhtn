//! [`HtnDomain`] — the baked, validated network of tasks.
//!
//! Domains are built from **task functions**: [`HtnDomain::from_root`] records
//! the root function (and, transitively, every function it `.then`-references)
//! into a graph, then **bakes** it into flat `Vec`s with contiguous task
//! indices. The runtime planner never calls task functions again — decomposing
//! a task is a direct flat-array lookup, and backtracking rewinds a
//! pre-allocated rollback stack over the [`PlanState`] scratchpad.

use std::any::TypeId;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use smallvec::SmallVec;
use ustr::Ustr;

use crate::error::{HtnError, HtnResult};
use crate::order::{topo_order_count, SubtaskOrder, LINEARIZATION_CAP};
use crate::state::{ComponentRegistry, FieldSet, PlanState, RegistryBuilder};
use crate::summaries::{compute_summaries, TaskSummary};
use crate::tasks::{
    Action, Effect, GoalBuilder, GoalFn, Precondition, Recorder, ScoreFn, SubtaskRef, TaskFn,
    TaskProto,
};

/// A baked and validated HTN domain. Immutable after bake; both planners read
/// it.
pub struct HtnDomain {
    /// All tasks in bake (discovery) order; index `0` is the root.
    pub tasks: Vec<Task>,
    /// The root task index (forward planning starts here).
    pub root: usize,
    /// `name -> index` lookup, built at bake time so the planners resolve
    /// task names in O(1). Keyed by interned [`Ustr`].
    pub(crate) index_of: HashMap<Ustr, usize>,
    /// `TypeId -> index` lookup (graph identity of the task functions).
    pub(crate) type_index: HashMap<TypeId, usize>,
    /// The component registry: every component type the domain's
    /// preconditions/effects touch, with dense slot indices.
    pub components: ComponentRegistry,
    /// Inferred per-task summaries (see [`summaries`]), one per task index.
    pub(crate) summaries: Vec<TaskSummary>,
}

/// Expand queued task functions until the queue drains: each function is
/// recorded exactly once — under the display name captured at its FIRST
/// reference site — as a compound placeholder that the recorded body fills
/// in; repeated/recursive references were already turned into plain edges at
/// their `then` sites. Shared by `from_root`, `DomainBuilder::root`, and
/// `DomainBuilder::insertable` (the discipline must not drift between them).
fn expand_queue(rec: &mut crate::tasks::Recorder) {
    while let Some((f, name)) = rec.queue.pop_front() {
        let tid = SubtaskRef::Fn(f.task_type_id());
        if rec.index_of.contains_key(&tid) {
            continue; // already recorded — the edge was recorded at `then`
        }
        rec.index_of.insert(tid, rec.tasks.len());
        rec.tasks.push((
            tid,
            name,
            TaskProto::Compound {
                methods: Vec::new(),
                policy: SelectionPolicy::default(),
            },
        ));
        let mut builder = crate::tasks::TaskBuilder::new(rec, rec.tasks.len() - 1);
        f.record(&mut builder);
        builder.finish();
    }
}

impl HtnDomain {
    /// Begin building a domain from a root task function. The function (and
    /// everything it references via `.then`) is recorded immediately; add
    /// goals with [`DomainBuilder::goal`], then [`DomainBuilder::build`].
    #[track_caller]
    pub fn from_root<F: TaskFn>(root: F) -> DomainBuilder {
        let mut rec = Recorder {
            registry: RegistryBuilder::default(),
            tasks: Vec::new(),
            index_of: HashMap::new(),
            queue: VecDeque::new(),
            errors: Vec::new(),
            next_synthetic: 0,
            shares: Vec::new(),
            insertables: Vec::new(),
            insertion: false,
            extra_roots: Vec::new(),
        };

        // Register the root placeholder, then record it. Recursive and
        // repeated `.then` references resolve against `index_of`, so each
        // task function is expanded exactly once and cycles become edges.
        let root_tid = SubtaskRef::Fn(TypeId::of::<F>());
        let root_name =
            crate::tasks::reference_display_name(F::task_name(), std::panic::Location::caller());
        rec.index_of.insert(root_tid, 0);
        rec.tasks.push((
            root_tid,
            root_name,
            TaskProto::Compound {
                methods: Vec::new(),
                policy: SelectionPolicy::default(),
            },
        ));
        {
            let task_index = rec.tasks.len() - 1;
            let mut builder = crate::tasks::TaskBuilder::new(&mut rec, task_index);
            root.record(&mut builder);
            builder.finish();
        }

        // Expand every referenced task function: each is recorded exactly once
        // (cycles become plain graph edges).
        expand_queue(&mut rec);

        DomainBuilder {
            rec,
            goals: Vec::new(),
        }
    }

    /// The root task (forward planning starts here).
    pub fn root_task(&self) -> &Task {
        &self.tasks[self.root]
    }

    /// Look up a task by its display name (O(1) via the precomputed index
    /// map). Introspection only — planning addresses tasks by their task
    /// function's `TypeId` (see [`Self::task_index`]).
    pub fn get_task(&self, name: &str) -> Option<&Task> {
        let idx = *self.index_of.get(&Ustr::from(name))?;
        self.tasks.get(idx)
    }

    /// Resolve a task function to its index into [`Self::tasks`] (O(1)) — the
    /// typed replacement for name lookup. The graph identity of a task is its
    /// function's `TypeId`; the fn item is passed by value (fn items are
    /// zero-sized and their types are unnameable).
    pub fn task_index<F: TaskFn>(&self, task: F) -> Option<usize> {
        self.type_index.get(&task.task_type_id()).copied()
    }

    /// Resolve a raw task `TypeId` to its index (O(1)) — the type-erased form
    /// of [`Self::task_index`] (GTN-synthesized tasks have no function type).
    pub fn task_index_by_type(&self, tid: TypeId) -> Option<usize> {
        self.type_index.get(&tid).copied()
    }

    /// The inferred [`TaskSummary`] of a task function (O(1)). The fn item is
    /// passed by value (see [`Self::task_index`]).
    pub fn task_summary<F: TaskFn>(&self, task: F) -> Option<&TaskSummary> {
        let idx = self.task_index(task)?;
        self.summaries.get(idx)
    }

    /// Look up a goal task by its goal function (for back-planning). The fn
    /// item is passed by value (see [`Self::task_index`]).
    pub fn goal<F: GoalFn>(&self, goal: F) -> Option<&GoalTask> {
        let idx = self.type_index.get(&goal.goal_type_id()).copied()?;
        match self.tasks.get(idx) {
            Some(Task::Goal(g)) => Some(g),
            _ => None,
        }
    }

    /// The name of every primitive task in the domain (for back-planning).
    pub fn primitive_names(&self) -> Vec<Ustr> {
        self.tasks
            .iter()
            .filter(|t| matches!(t, Task::Primitive(_)))
            .map(|t| t.name().into())
            .collect()
    }
}

impl std::fmt::Debug for HtnDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HtnDomain")
            .field("tasks", &self.tasks.len())
            .field("root", &self.root)
            .field("components", &self.components.len())
            .field("summaries", &self.summaries.len())
            .finish()
    }
}

/// Intermediate builder returned by [`HtnDomain::from_root`]: add goals, then
/// [`build`](Self::build).
pub struct DomainBuilder {
    rec: Recorder,
    goals: Vec<(&'static str, TypeId, Vec<crate::tasks::Effect>)>,
}

impl DomainBuilder {
    /// Mark a recorded primitive task as **shared** (GTN Theorem 4.8's
    /// `fin_o` construction, compiled at bake time): every method body that
    /// references it is rewritten to a wrapper compound whose plan contains
    /// the primitive at most once. See [`gtn`](crate::gtn) for the compiled
    /// shape and its execution contract.
    #[must_use]
    pub fn share_task<F: TaskFn>(mut self, _f: F) -> Self {
        self.rec.shares.push(SubtaskRef::Fn(TypeId::of::<F>()));
        self
    }

    /// Compile **task insertion** (GTN plan repair): a synthetic
    /// `gtn/insert` compound is spliced between the members of every
    /// total-order method body, letting any applicable primitive run in the
    /// gaps — but only on backtrack, so plain plans are found first. See
    /// [`gtn`](crate::gtn).
    #[must_use]
    pub fn with_insertion(mut self) -> Self {
        self.rec.insertion = true;
        self
    }

    /// Register an **additional root task** — a second compound entry point
    /// into the same baked network, alongside the forward planner's root.
    /// The motivating use is adversarial planning
    /// ([`Ahtn`](crate::ahtn)): player max plans from the main root, player
    /// min from an extra root, both against the same world state. Extra
    /// roots must be compound tasks (validated at bake) and are unreachable
    /// from the forward root unless referenced — forward planning is
    /// unaffected.
    #[must_use]
    #[track_caller]
    pub fn root<F: TaskFn>(mut self, f: F) -> Self {
        let tid = SubtaskRef::Fn(TypeId::of::<F>());
        if self.rec.index_of.contains_key(&tid) {
            // Already recorded (e.g. also referenced as a subtask): just mark
            // it as an extra root — bake validates its kind.
            self.rec.extra_roots.push(tid);
            return self;
        }
        self.rec.index_of.insert(tid, self.rec.tasks.len());
        self.rec.tasks.push((
            tid,
            crate::tasks::reference_display_name(F::task_name(), std::panic::Location::caller()),
            TaskProto::Compound {
                methods: Vec::new(),
                policy: SelectionPolicy::default(),
            },
        ));
        let task_index = self.rec.tasks.len() - 1;
        let mut builder = crate::tasks::TaskBuilder::new(&mut self.rec, task_index);
        f.record(&mut builder);
        builder.finish();
        // Drain queued (`.then`-referenced) tasks, same discipline as the
        // root expansion loop.
        expand_queue(&mut self.rec);
        self.rec.extra_roots.push(tid);
        self
    }

    /// Record a task that no method body references and mark it as an
    /// **insertion candidate**: with [`Self::with_insertion`] compiled in,
    /// the search may weave it into plan gaps (plan repair). Unreferenced
    /// task functions are otherwise dropped by the recording, and only
    /// explicitly registered tasks become candidates — an ungated primitive
    /// in the candidate set is an unbounded insertion well (the search can
    /// re-insert it forever), so curation is deliberate. Candidates are
    /// routed through shared wrappers when the task is also shared.
    #[must_use]
    #[track_caller]
    pub fn insertable<F: TaskFn>(mut self, f: F) -> Self {
        let tid = SubtaskRef::Fn(TypeId::of::<F>());
        self.rec.insertables.push(tid);
        if self.rec.index_of.contains_key(&tid) {
            return self; // already recorded (also referenced somewhere)
        }
        self.rec.index_of.insert(tid, self.rec.tasks.len());
        self.rec.tasks.push((
            tid,
            crate::tasks::reference_display_name(F::task_name(), std::panic::Location::caller()),
            TaskProto::Compound {
                methods: Vec::new(),
                policy: SelectionPolicy::default(),
            },
        ));
        let task_index = self.rec.tasks.len() - 1;
        let mut builder = crate::tasks::TaskBuilder::new(&mut self.rec, task_index);
        f.record(&mut builder);
        builder.finish();
        // Drain queued (`.then`-referenced) tasks, same discipline as the
        // root expansion loop.
        expand_queue(&mut self.rec);
        self
    }

    /// Register a goal task (a named set of desired effects) for
    /// back-planning. The goal function's `TypeId` is the goal's identity; its
    /// name is display/debug only.
    #[must_use]
    pub fn goal<F: GoalFn>(mut self, f: F) -> Self {
        let name = F::goal_name();
        let mut gb = GoalBuilder::new(&mut self.rec);
        f.record(&mut gb);
        let effects = gb.finish();
        self.goals.push((name, TypeId::of::<F>(), effects));
        self
    }

    /// Validate and freeze the domain: duplicate names, mixed compound/
    /// primitive declarations, methodless compounds, and a non-compound root
    /// are errors; the index maps and the inferred task summaries are
    /// computed here.
    pub fn build(mut self) -> HtnResult<HtnDomain> {
        // GTN compilations run before validation so the synthesized tasks are
        // validated, baked, and summarized like hand-written ones. Sharing
        // first: insertion routes its candidates through the wrappers.
        let wrapped = crate::gtn::apply_sharing(&mut self.rec)?;
        crate::gtn::apply_insertion(&mut self.rec, &wrapped)?;

        // Soft-collected recording errors (e.g. an effect closure taking the
        // same component type twice) join the builder's validation errors, so
        // one `build()` call reports every authoring bug at once instead of
        // panicking on the first.
        self.rec.errors.extend(self.rec.registry.take_errors());

        if !self.rec.errors.is_empty() {
            return Err(HtnError::builder(self.rec.errors.join("; ")));
        }

        // Extra roots (adversarial planning) must be compound tasks.
        for rref in &self.rec.extra_roots {
            let Some(&idx) = self.rec.index_of.get(rref) else {
                continue;
            };
            let (name, proto) = (&self.rec.tasks[idx].1, &self.rec.tasks[idx].2);
            if !matches!(proto, TaskProto::Compound { .. }) {
                return Err(HtnError::builder(format!(
                    "extra root `{name}` must be a compound task"
                )));
            }
        }

        let mut tasks: Vec<Task> = Vec::with_capacity(self.rec.tasks.len() + self.goals.len());
        let mut index_of: HashMap<Ustr, usize> = HashMap::with_capacity(self.rec.tasks.len());
        let mut type_index: HashMap<TypeId, usize> = HashMap::with_capacity(self.rec.tasks.len());

        for (i, (tid, name, proto)) in self.rec.tasks.into_iter().enumerate() {
            let key = Ustr::from(name);
            // Display names are not identity: two distinct task functions may
            // share a last-path-segment name (same-named fns in different
            // modules, closures' `{{closure}}`), so this map is first-wins and
            // introspection only. Identity is the `TypeId` below.
            index_of.entry(key).or_insert(i);
            // Only task functions participate in the TypeId graph index —
            // transform-synthesized tasks (GTN compilation) have no identity.
            if let SubtaskRef::Fn(tid) = tid {
                type_index.insert(tid, i);
            }
            tasks.push(match proto {
                TaskProto::Compound { methods, policy } => {
                    if methods.is_empty() {
                        return Err(HtnError::builder(format!(
                            "compound task `{name}` has no branches — it can never decompose"
                        )));
                    }
                    // Bake each method's subtask order: a pure `then` chain
                    // stays total; branches that used `subtask`/`before`
                    // build their constraint DAG (validated acyclic) and
                    // precompute the linearization count + first order.
                    let baked_methods: HtnResult<Vec<Method>> = methods
                        .into_iter()
                        .map(|m| {
                            let order = bake_subtask_order(&m, name)?;
                            Ok(Method {
                                name: m.name,
                                utility: m.utility,
                                preconditions: m.preconditions,
                                subtasks: m
                                    .subtasks
                                    .iter()
                                    .map(|(tid, _, _)| {
                                        *self.rec.index_of.get(tid).expect("queued task recorded")
                                            as u32
                                    })
                                    .collect(),
                                order,
                                possible_writes: Default::default(),
                                guaranteed_writes: Default::default(),
                                min_cost: 0.0,
                            })
                        })
                        .collect();
                    Task::Compound(CompoundTask {
                        name,
                        type_id: match tid {
                            SubtaskRef::Fn(t) => Some(t),
                            SubtaskRef::Synthetic(_) => None,
                        },
                        policy,
                        methods: baked_methods?,
                    })
                }
                TaskProto::Primitive {
                    preconditions,
                    effects,
                    expected_effects,
                    action,
                    cost,
                    static_cost,
                } => {
                    // Bake the write-slot lists once: the executor's commit
                    // list (effects + expected) and the back-planner's
                    // guaranteed list (effects only).
                    let mut write_slot_list = smallvec::SmallVec::<[usize; 4]>::new();
                    let mut guaranteed_slot_list = smallvec::SmallVec::<[usize; 4]>::new();
                    for e in effects.iter() {
                        for &w in e.writes() {
                            guaranteed_slot_list.push(w);
                            write_slot_list.push(w);
                        }
                    }
                    for e in expected_effects.iter() {
                        for &w in e.writes() {
                            write_slot_list.push(w);
                        }
                    }
                    Task::Primitive(PrimitiveTask {
                        name,
                        type_id: match tid {
                            SubtaskRef::Fn(t) => Some(t),
                            SubtaskRef::Synthetic(_) => None,
                        },
                        preconditions,
                        effects,
                        expected_effects,
                        action,
                        cost,
                        static_cost,
                        write_slot_list,
                        guaranteed_slot_list,
                    })
                }
            });
        }

        for (name, tid, effects) in self.goals {
            let key = Ustr::from(name);
            index_of.entry(key).or_insert(tasks.len());
            if type_index.insert(tid, tasks.len()).is_some() {
                return Err(HtnError::builder(format!(
                    "duplicate goal function `{name}` (a goal function may be registered once)"
                )));
            }
            tasks.push(Task::Goal(GoalTask { name, effects }));
        }

        // The root must be a compound task (it is the first recorded one).
        if !tasks[0].is_compound() {
            return Err(HtnError::builder(
                "the root task function declared no branches — forward planning needs a compound root",
            ));
        }

        let mut domain = HtnDomain {
            tasks,
            root: 0,
            index_of,
            type_index,
            components: self.rec.registry.freeze(),
            summaries: Vec::new(),
        };
        compute_summaries(&mut domain);
        Ok(domain)
    }
}

/// Bake one method's [`SubtaskOrder`] from its recorded members: a pure
/// `then` chain stays total; a branch that used `subtask`/`before` builds
/// the per-member predecessor bitmask, validates the DAG (acyclic, ≤ 64
/// members), and precomputes the (capped) linearization count and the first
/// topological order. Single-order sets normalize back to [`SubtaskOrder::Total`].
fn bake_subtask_order(m: &crate::tasks::MethodProto, task_name: &str) -> HtnResult<SubtaskOrder> {
    if !m.unordered && m.edges.is_empty() {
        return Ok(SubtaskOrder::Total);
    }
    let n = m.subtasks.len();
    if n > 64 {
        return Err(HtnError::builder(format!(
            "compound task `{task_name}` has a partially-ordered branch with {n} members — the limit is 64"
        )));
    }
    let mut preds = smallvec::SmallVec::<[u64; 4]>::from_elem(0, n);
    // `then` members run after every prior member; `subtask` members after
    // every prior `then` member (unordered relative to other subtask
    // members).
    for (p, &(_, _, is_then)) in m.subtasks.iter().enumerate() {
        if is_then {
            for q in 0..p {
                preds[p] |= 1 << q;
            }
        }
    }
    for (p, &(_, _, is_then)) in m.subtasks.iter().enumerate() {
        if !is_then {
            for (q, &(_, _, then_q)) in m.subtasks.iter().enumerate().take(p) {
                if then_q {
                    preds[p] |= 1 << q;
                }
            }
        }
    }
    for &(a, b) in &m.edges {
        debug_assert!(a < n as u32 && b < n as u32, "handle out of range");
        preds[b as usize] |= 1 << a;
    }
    let orders = topo_order_count(&preds, LINEARIZATION_CAP);
    if orders == 0 {
        return Err(HtnError::builder(format!(
            "compound task `{task_name}` has a branch whose `before` constraints form a cycle"
        )));
    }
    let first: smallvec::SmallVec<[u8; 4]> = crate::order::linearize(&preds, 0)
        .expect("acyclic, so order 0 exists")
        .into_iter()
        .collect();
    let is_declaration = first.iter().enumerate().all(|(p, &q)| p == q as usize);
    if orders == 1 && is_declaration {
        // Fully constrained AND the single order is the declaration order:
        // schedule it on the total-order fast path. (A single order that
        // differs from the declaration order stays partial — `Total` would
        // push the wrong sequence.)
        return Ok(SubtaskOrder::Total);
    }
    Ok(SubtaskOrder::Partial {
        preds,
        orders: orders as u32,
        first,
    })
}
// ---------------------------------------------------------------------------
// Selection policies
// ---------------------------------------------------------------------------

/// How a compound task's *valid* branches are ranked before the planner
/// descends. Applied after precondition evaluation — only valid branches are
/// ranked, and a look-ahead pin (unique surviving method) overrides ranking
/// entirely.
#[derive(Clone, Default)]
pub enum SelectionPolicy {
    /// Branches are tried in declaration order (the default; what the
    /// planner has always done).
    #[default]
    FirstMatch,

    /// All valid branches are scored by their `utility` closure (branches
    /// without one score 0); the highest wins. Ties break by declaration
    /// order. Deterministic: backtracking re-derives the same order.
    HighestUtility,

    /// Valid branches are sampled without replacement, proportional to their
    /// utility scores (branches without one weigh 1.0). The sampled order is
    /// snapshotted into the decomposition frame, so backtracking exhausts the
    /// sampled order instead of re-sampling — completeness is preserved.
    ///
    /// Sampling is stateless and deterministic: the permutation is derived
    /// from `seed` and the choice point's position in the plan, so the same
    /// state always yields the same order (stable replans).
    WeightedRandom {
        /// Seed for the deterministic weighted sampler.
        seed: u64,
    },

    /// The caller supplies the comparator at domain-build time. The ranker
    /// must be deterministic for a given `(candidates, state)` pair, and its
    /// output must be a permutation of the candidate indices (missing
    /// candidates are appended in declaration order so no branch is lost).
    Custom(Arc<dyn BranchRanker>),
}

/// One valid branch offered to a [`BranchRanker`].
pub struct BranchCandidate<'a> {
    /// The branch's declaration index (its MTR identity).
    pub index: u32,
    /// The branch's declared name, if any.
    pub name: Option<&'static str>,
    /// The branch's declared static utility, if any.
    pub utility: Option<f32>,
    /// The branch's subtask list (task indices, for structural heuristics).
    pub subtasks: &'a [u32],
}

/// Ranks *valid* branch candidates. Implementations must be deterministic
/// for a given `(candidates, state)` pair.
pub trait BranchRanker: Send + Sync {
    /// Appends the candidate indices to `out` in preferred order. Scratch-
    /// buffer signature: no per-node allocation inside the planner.
    fn rank(&self, candidates: &[BranchCandidate<'_>], state: &PlanState, out: &mut Vec<u32>);
}

// ---------------------------------------------------------------------------
// Baked nodes — what the planner searches
// ---------------------------------------------------------------------------

/// One baked decomposition alternative of a compound task. Its preconditions
/// must all hold for its subtask list to be selected.
pub struct Method {
    /// The branch's declared name, if any.
    pub name: Option<&'static str>,
    /// The branch's declared dynamic utility, if any.
    pub utility: Option<ScoreFn>,
    /// Conditions that must all hold for this method to be chosen.
    pub preconditions: Vec<Precondition>,
    /// Subtask indices into [`HtnDomain::tasks`](crate::domain::HtnDomain),
    /// in declaration order (the member set; see [`Self::order`]).
    pub subtasks: Vec<u32>,
    /// How the members are ordered: a total `then` chain (the default) or a
    /// partially-ordered set scheduled by the search.
    pub order: SubtaskOrder,
    /// Fields that *some* refinement of this method's subtasks may write
    /// (bake-time over-approximation; computed with the summaries).
    pub(crate) possible_writes: FieldSet,
    /// Fields that *every* refinement of this method's subtask sequence
    /// writes: the union of the subtasks' guaranteed-write summaries (bake
    /// time). The under-approximation the backward planner uses to let
    /// compound tasks participate in reverse chaining.
    pub(crate) guaranteed_writes: FieldSet,
    /// Lower bound on the total primitive cost of executing this method's
    /// subtask sequence (sum of the subtasks' `min_cost` summaries; bake
    /// time). The [`CostBounded`](crate::selection::HtnSearchStrategy::CostBounded)
    /// strategy prunes commitments whose bound cannot beat the best complete
    /// plan.
    pub(crate) min_cost: f32,
}

/// A baked compound task: on decomposition, pick the first method whose
/// preconditions evaluate true.
pub struct CompoundTask {
    /// The task's clean function name.
    pub name: &'static str,
    /// The task function's `TypeId` (graph identity for introspection);
    /// `None` for transform-synthesized tasks (the GTN compilation).
    pub type_id: Option<TypeId>,
    /// How this task's valid branches are ranked.
    pub policy: SelectionPolicy,
    /// Ordered decomposition alternatives.
    pub methods: Vec<Method>,
}

impl CompoundTask {
    /// First method (at or after `skip`) whose preconditions hold.
    pub fn find_method(&self, state: &PlanState, skip: usize) -> Option<(&Method, usize)> {
        self.methods
            .iter()
            .enumerate()
            .skip(skip)
            .find(|(_i, m)| m.preconditions.iter().all(|c| c.evaluate(state)))
            .map(|(i, m)| (m, i))
    }

    /// Rank this task's **valid** branches per the task's
    /// [`SelectionPolicy`], appending declaration indices to `out` in the
    /// order the search should try them.
    ///
    /// Called once per node visit: at a given choice point the scratchpad is
    /// fixed, so precondition validity is constant across the node's attempts
    /// and the ranked order can be snapshotted into the decomposition frame
    /// (backtracking resumes from the snapshot instead of re-ranking —
    /// required for [`WeightedRandom`](SelectionPolicy::WeightedRandom)
    /// soundness).
    ///
    /// `nonce` disambiguates choice points for the weighted sampler (the
    /// planner passes the current partial-plan length), keeping the sampling
    /// stateless and deterministic.
    pub(crate) fn rank_valid_methods(
        &self,
        state: &PlanState,
        nonce: u64,
        out: &mut SmallVec<[u32; 4]>,
    ) {
        out.clear();

        // Validity pass: preconditions are constant at this node.
        let valid: SmallVec<[usize; 8]> = self
            .methods
            .iter()
            .enumerate()
            .filter(|(_i, m)| m.preconditions.iter().all(|c| c.evaluate(state)))
            .map(|(i, _)| i)
            .collect();

        match &self.policy {
            SelectionPolicy::FirstMatch => {
                out.extend(valid.iter().map(|&i| i as u32));
            }
            SelectionPolicy::HighestUtility => {
                // Stable sort by score descending; ties keep declaration
                // order. NaN scores sort last.
                let mut scored: Vec<(usize, f32)> = valid
                    .iter()
                    .map(|&i| {
                        let score = self.methods[i]
                            .utility
                            .as_ref()
                            .map(|f| f(state))
                            .unwrap_or(0.0);
                        (
                            i,
                            if score.is_nan() {
                                f32::NEG_INFINITY
                            } else {
                                score
                            },
                        )
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                out.extend(scored.iter().map(|&(i, _)| i as u32));
            }
            SelectionPolicy::WeightedRandom { seed } => {
                let weights: SmallVec<[f32; 8]> = valid
                    .iter()
                    .map(|&i| {
                        self.methods[i]
                            .utility
                            .as_ref()
                            .map(|f| f(state))
                            .unwrap_or(1.0)
                            .max(0.0)
                    })
                    .collect();
                // Stateless RNG: splitmix64 over (seed, task identity, nonce)
                // — the same choice point always yields the same order.
                // DefaultHasher::new() is deterministic within a process.
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                self.type_id.hash(&mut hasher);
                let mut rng = seed
                    ^ hasher.finish().wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ nonce.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                sample_weighted_order(&weights, &mut rng, out);
            }
            SelectionPolicy::Custom(ranker) => {
                let candidates: Vec<BranchCandidate<'_>> = valid
                    .iter()
                    .map(|&i| BranchCandidate {
                        index: i as u32,
                        name: self.methods[i].name,
                        utility: None,
                        subtasks: &self.methods[i].subtasks,
                    })
                    .collect();
                // The ranker writes into a plain Vec (its trait signature);
                // the ranked order is copied into the inline scratch after.
                let mut ranked = Vec::with_capacity(candidates.len());
                ranker.rank(&candidates, state, &mut ranked);
                // Sanitize: keep unique in-range valid entries in the
                // ranker's order, then append any valid entries the ranker
                // omitted (declaration order) — a bad ranker must not be able
                // to make branches unreachable.
                let mut seen = SmallVec::<[bool; 8]>::from_elem(false, self.methods.len());
                for mi in ranked {
                    let i = mi as usize;
                    if i < self.methods.len() && valid.contains(&i) && !seen[i] {
                        seen[i] = true;
                        out.push(mi);
                    }
                }
                for &i in &valid {
                    if !seen[i] {
                        out.push(i as u32);
                    }
                }
            }
        }
    }
}

/// Sample `0..weights.len()` without replacement, proportional to weight.
/// Zero total weight falls back to declaration order. Deterministic given
/// the RNG state (splitmix64 stream).
fn sample_weighted_order(weights: &[f32], rng: &mut u64, out: &mut SmallVec<[u32; 4]>) {
    let mut remaining: SmallVec<[usize; 8]> = (0..weights.len()).collect();
    while !remaining.is_empty() {
        let total: f32 = remaining.iter().map(|&i| weights[i]).sum();
        if !(total > 0.0) {
            // All-zero (or degenerate) weights: declaration order for the rest.
            out.extend(remaining.drain(..).map(|i| i as u32));
            break;
        }
        // splitmix64 step.
        *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let r = (z >> 11) as f32 / (1u64 << 53) as f32 * total;
        let mut acc = 0.0;
        let mut chosen = remaining.len() - 1;
        for (k, &i) in remaining.iter().enumerate() {
            acc += weights[i];
            if r < acc {
                chosen = k;
                break;
            }
        }
        out.push(remaining.remove(chosen) as u32);
    }
}

/// A baked primitive (leaf) task: preconditions, effects, expected effects,
/// and the real-world action. `expected_effects` are simulated during
/// planning but are only *hoped* effects (never applied to the real entity).
pub struct PrimitiveTask {
    /// The task's clean function name.
    pub name: &'static str,
    /// The task function's `TypeId`; `None` for transform-synthesized tasks.
    pub type_id: Option<TypeId>,
    /// Conditions that must all hold for this task to be pickable.
    pub preconditions: Vec<Precondition>,
    /// Effects applied to the scratchpad during search and to the real
    /// entity at execution.
    pub effects: Vec<Effect>,
    /// Anticipated (non-guaranteed) effects applied during planning only.
    pub expected_effects: Vec<Effect>,
    /// The real-world action dispatched at execution (if any).
    pub action: Option<Action>,
    /// The declared cost estimate, if any (used by cost-aware strategies).
    pub cost: Option<ScoreFn>,
    /// The constant declared via `.cost(c)`, if the cost signal is a constant
    /// — the lower bound the `min_cost` summary infers from (`None` for
    /// dynamic `cost_fn` costs, which conservatively bound at 0).
    pub(crate) static_cost: Option<f32>,
    /// Baked union of all effect + expected-effect write slots (what
    /// [`Self::write_slots`] iterates). Precomputed so the executor and
    /// back-planner never re-collect per step.
    pub(crate) write_slot_list: SmallVec<[usize; 4]>,
    /// Baked write slots of the guaranteed (real) effects only (what
    /// [`Self::guaranteed_slots`] iterates).
    pub(crate) guaranteed_slot_list: SmallVec<[usize; 4]>,
}

impl PrimitiveTask {
    /// True when every precondition holds for the scratchpad state.
    pub fn preconditions_met(&self, state: &PlanState) -> bool {
        self.preconditions.iter().all(|c| c.evaluate(state))
    }

    /// Apply effects (and expected effects) to the scratchpad during search.
    pub fn apply_effects(&self, state: &mut PlanState) {
        for e in self.effects.iter().chain(self.expected_effects.iter()) {
            e.apply(state);
        }
    }

    /// The union of all effect + expected-effect write slots.
    pub fn write_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.write_slot_list.iter().copied()
    }

    /// The write slots of the guaranteed (real) effects only.
    pub fn guaranteed_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.guaranteed_slot_list.iter().copied()
    }

    /// The baked union of all effect + expected-effect write slots (the
    /// executor's commit list — no per-step collection).
    pub(crate) fn write_slot_slice(&self) -> &[usize] {
        &self.write_slot_list
    }
}

/// A baked goal task — a named set of desired effects for back-planning.
pub struct GoalTask {
    /// The goal function's clean name.
    pub name: &'static str,
    /// The desired effects that form the goal state.
    pub effects: Vec<Effect>,
}

impl GoalTask {
    /// The slot indices the goal's effects write.
    pub fn write_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.effects.iter().flat_map(|e| e.writes.iter().copied())
    }
}

/// One task in the baked domain.
pub enum Task {
    /// A leaf task.
    Primitive(PrimitiveTask),
    /// A decomposition task.
    Compound(CompoundTask),
    /// A back-planning target.
    Goal(GoalTask),
}

impl Task {
    /// This task's name.
    pub fn name(&self) -> &str {
        match self {
            Task::Primitive(t) => t.name,
            Task::Compound(t) => t.name,
            Task::Goal(t) => t.name,
        }
    }

    /// Whether this is a compound task (the root is the first one recorded).
    pub fn is_compound(&self) -> bool {
        matches!(self, Task::Compound(_))
    }
}

/// Interned task-name handle used by [`Plan`](crate::planner::Plan).
pub type TaskName = Ustr;
