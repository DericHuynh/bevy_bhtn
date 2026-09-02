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

use ustr::Ustr;

use crate::error::{HtnError, HtnResult};
use crate::order::{topo_order_count, SubtaskOrder, LINEARIZATION_CAP};
use crate::selection::SelectionPolicy;
use crate::state::{ComponentRegistry, RegistryBuilder};
use crate::summaries::{compute_summaries, TaskSummary};
use crate::tasks::{
    CompoundTask, GoalBuilder, GoalFn, GoalTask, Method, PrimitiveTask, Recorder, SubtaskRef, Task,
    TaskFn, TaskProto,
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

impl HtnDomain {
    /// Begin building a domain from a root task function. The function (and
    /// everything it references via `.then`) is recorded immediately; add
    /// goals with [`DomainBuilder::goal`], then [`DomainBuilder::build`].
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
        };

        // Register the root placeholder, then record it. Recursive and
        // repeated `.then` references resolve against `index_of`, so each
        // task function is expanded exactly once and cycles become edges.
        let root_tid = SubtaskRef::Fn(TypeId::of::<F>());
        let root_name = F::task_name();
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
            let mut builder = crate::tasks::TaskBuilder::new(&mut rec);
            root.record(&mut builder);
            builder.finish();
        }

        // Expand every referenced task function (LIFO order; each function
        // is recorded exactly once — cycles become plain graph edges).
        while let Some(f) = rec.queue.pop_front() {
            let tid = SubtaskRef::Fn(f.task_type_id());
            if rec.index_of.contains_key(&tid) {
                continue; // already recorded — the edge was recorded at `then`
            }
            rec.index_of.insert(tid, rec.tasks.len());
            rec.tasks.push((
                tid,
                f.task_name_erased(),
                TaskProto::Compound {
                    methods: Vec::new(),
                    policy: SelectionPolicy::default(),
                },
            ));
            let mut builder = crate::tasks::TaskBuilder::new(&mut rec);
            f.record(&mut builder);
            builder.finish();
        }

        DomainBuilder {
            rec,
            goals: Vec::new(),
        }
    }

    /// The root task (forward planning starts here).
    pub fn root_task(&self) -> &Task {
        &self.tasks[self.root]
    }

    /// Look up a task by name (O(1) via the precomputed index map).
    pub fn get_task(&self, name: &str) -> Option<&Task> {
        let idx = *self.index_of.get(&Ustr::from(name))?;
        self.tasks.get(idx)
    }

    /// Resolve a task name to its index into [`Self::tasks`] (O(1)).
    pub fn task_index(&self, name: Ustr) -> Option<usize> {
        self.index_of.get(&name).copied()
    }

    /// Resolve a task function's `TypeId` to its index (O(1)).
    pub fn task_index_by_type(&self, tid: TypeId) -> Option<usize> {
        self.type_index.get(&tid).copied()
    }

    /// The inferred [`TaskSummary`] of a task, by name (O(1)).
    pub fn task_summary(&self, name: &str) -> Option<&TaskSummary> {
        let idx = *self.index_of.get(&Ustr::from(name))?;
        self.summaries.get(idx)
    }

    /// The inferred possible-write set of one method of a compound task, by
    /// task name and method index (O(1)). This is what the forward planner's
    /// look-ahead sweep uses for optimistic state propagation.
    pub fn method_possible_writes(
        &self,
        task: &str,
        method: usize,
    ) -> Option<&crate::summaries::FieldSet> {
        match self.get_task(task)? {
            Task::Compound(c) => c.methods.get(method).map(|m| &m.possible_writes),
            _ => None,
        }
    }

    /// Look up a goal task by name (for back-planning).
    pub fn goal(&self, name: &str) -> Option<&GoalTask> {
        match self.get_task(name) {
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
    goals: Vec<(&'static str, Vec<crate::tasks::Effect>)>,
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

    /// Record a task that no method body references and mark it as an
    /// **insertion candidate**: with [`Self::with_insertion`] compiled in,
    /// the search may weave it into plan gaps (plan repair). Unreferenced
    /// task functions are otherwise dropped by the recording, and only
    /// explicitly registered tasks become candidates — an ungated primitive
    /// in the candidate set is an unbounded insertion well (the search can
    /// re-insert it forever), so curation is deliberate. Candidates are
    /// routed through shared wrappers when the task is also shared.
    #[must_use]
    pub fn insertable<F: TaskFn>(mut self, f: F) -> Self {
        let tid = SubtaskRef::Fn(TypeId::of::<F>());
        self.rec.insertables.push(tid);
        if self.rec.index_of.contains_key(&tid) {
            return self; // already recorded (also referenced somewhere)
        }
        self.rec.index_of.insert(tid, self.rec.tasks.len());
        self.rec.tasks.push((
            tid,
            F::task_name(),
            TaskProto::Compound {
                methods: Vec::new(),
                policy: SelectionPolicy::default(),
            },
        ));
        let mut builder = crate::tasks::TaskBuilder::new(&mut self.rec);
        f.record(&mut builder);
        builder.finish();
        // Drain queued (`.then`-referenced) tasks, same discipline as the
        // root expansion loop.
        while let Some(g) = self.rec.queue.pop_front() {
            let gid = SubtaskRef::Fn(g.task_type_id());
            if self.rec.index_of.contains_key(&gid) {
                continue;
            }
            self.rec.index_of.insert(gid, self.rec.tasks.len());
            self.rec.tasks.push((
                gid,
                g.task_name_erased(),
                TaskProto::Compound {
                    methods: Vec::new(),
                    policy: SelectionPolicy::default(),
                },
            ));
            let mut b2 = crate::tasks::TaskBuilder::new(&mut self.rec);
            g.record(&mut b2);
            b2.finish();
        }
        self
    }

    /// Register a goal task (a named set of desired effects) for
    /// back-planning. The goal function's name becomes the goal's lookup key.
    #[must_use]
    pub fn goal<F: GoalFn>(mut self, f: F) -> Self {
        let name = F::goal_name();
        let mut gb = GoalBuilder::new(&mut self.rec);
        f.record(&mut gb);
        let effects = gb.finish();
        self.goals.push((name, effects));
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

        if let Some(err) = self.rec.errors.first() {
            return Err(HtnError::builder(err.clone()));
        }

        let mut tasks: Vec<Task> = Vec::with_capacity(self.rec.tasks.len() + self.goals.len());
        let mut index_of: HashMap<Ustr, usize> = HashMap::with_capacity(self.rec.tasks.len());
        let mut type_index: HashMap<TypeId, usize> = HashMap::with_capacity(self.rec.tasks.len());

        for (i, (tid, name, proto)) in self.rec.tasks.into_iter().enumerate() {
            let key = Ustr::from(name);
            if index_of.insert(key, i).is_some() {
                return Err(HtnError::builder(format!("duplicate task name `{name}`")));
            }
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

        for (name, effects) in self.goals {
            let key = Ustr::from(name);
            if index_of.insert(key, tasks.len()).is_some() {
                return Err(HtnError::builder(format!("duplicate task name `{name}`")));
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
