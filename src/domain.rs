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
use crate::state::ComponentRegistry;
use crate::summaries::{compute_summaries, TaskSummary};
use crate::tasks::{
    CompoundTask, GoalBuilder, GoalFn, GoalTask, Method, PrimitiveTask, Recorder, Task, TaskFn,
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

impl HtnDomain {
    /// Begin building a domain from a root task function. The function (and
    /// everything it references via `.then`) is recorded immediately; add
    /// goals with [`DomainBuilder::goal`], then [`DomainBuilder::build`].
    pub fn from_root<F: TaskFn>(root: F) -> DomainBuilder {
        let mut rec = Recorder {
            registry: ComponentRegistry::default(),
            tasks: Vec::new(),
            index_of: HashMap::new(),
            queue: VecDeque::new(),
            errors: Vec::new(),
        };

        // Register the root placeholder, then record it. Recursive and
        // repeated `.then` references resolve against `index_of`, so each
        // task function is expanded exactly once and cycles become edges.
        let root_tid = TypeId::of::<F>();
        let root_name = F::task_name();
        rec.index_of.insert(root_tid, 0);
        rec.tasks.push((
            root_tid,
            root_name,
            TaskProto::Compound {
                methods: Vec::new(),
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
            let tid = f.task_type_id();
            if rec.index_of.contains_key(&tid) {
                continue; // already recorded — the edge was recorded at `then`
            }
            rec.index_of.insert(tid, rec.tasks.len());
            rec.tasks.push((
                tid,
                f.task_name_erased(),
                TaskProto::Compound {
                    methods: Vec::new(),
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
    pub fn build(self) -> HtnResult<HtnDomain> {
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
            type_index.insert(tid, i);
            tasks.push(match proto {
                TaskProto::Compound { methods } => {
                    if methods.is_empty() {
                        return Err(HtnError::builder(format!(
                            "compound task `{name}` has no branches — it can never decompose"
                        )));
                    }
                    Task::Compound(CompoundTask {
                        name,
                        type_id: tid,
                        methods: methods
                            .into_iter()
                            .map(|m| Method {
                                preconditions: m.preconditions,
                                subtasks: m
                                    .subtasks
                                    .iter()
                                    .map(|(tid, _)| {
                                        *self.rec.index_of.get(tid).expect("queued task recorded")
                                            as u32
                                    })
                                    .collect(),
                                possible_writes: Default::default(),
                            })
                            .collect(),
                    })
                }
                TaskProto::Primitive {
                    preconditions,
                    effects,
                    expected_effects,
                    action,
                } => Task::Primitive(PrimitiveTask {
                    name,
                    type_id: tid,
                    preconditions,
                    effects,
                    expected_effects,
                    action,
                }),
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
            components: self.rec.registry,
            summaries: Vec::new(),
        };
        compute_summaries(&mut domain);
        Ok(domain)
    }
}
