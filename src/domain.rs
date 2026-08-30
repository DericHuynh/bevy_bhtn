//! [`HtnDomain`] — the parsed, validated network of tasks.

use std::collections::HashMap;

use bevy_reflect::{Reflect, TypeRegistry};
use ustr::Ustr;

use crate::error::HtnResult;
use crate::operators::verify_operator;
use crate::summaries::{FieldSet, TaskSummary};
use crate::tasks::{GoalTask, Task};

/// A parsed and validated HTN domain. Immutable after parse; both planners read
/// it.
#[derive(Debug, Clone)]
pub struct HtnDomain {
    /// The schema version string from the `.htn` file.
    pub schema: String,
    /// All tasks in declaration order.
    pub tasks: Vec<Task>,
    /// `name -> index` lookup into [`Self::tasks`], built once at parse time so
    /// the planners resolve subtask names in O(1) without rebuilding a map on
    /// every `plan` call. Keyed by interned [`Ustr`] for cheap hashing.
    pub(crate) index_of: HashMap<Ustr, usize>,
    /// Every state-field name referenced by any condition or effect in the
    /// domain, in index order (the universe [`FieldSet`]s range over).
    pub fields: Vec<Ustr>,
    /// `field name -> index` into [`Self::fields`], built at parse time.
    pub(crate) field_index: HashMap<Ustr, usize>,
    /// Inferred per-task summaries (see [`summaries`]), one per task index.
    /// Empty only for domains built by hand without a parse-time index rebuild.
    pub(crate) summaries: Vec<TaskSummary>,
}

impl HtnDomain {
    /// The schema version string.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// The default forward-planning root: the first compound task. Falls back to
    /// the first task if no compound exists.
    pub fn root_task(&self) -> Option<&Task> {
        self.tasks
            .iter()
            .find(|t| t.is_root_compound())
            .or_else(|| self.tasks.first())
    }

    /// Look up a task by name (O(1) via the precomputed index map).
    pub fn get_task(&self, name: &str) -> Option<&Task> {
        let key = Ustr::from(name);
        let idx = *self.index_of.get(&key)?;
        self.tasks.get(idx)
    }

    /// Resolve a task name to its index into [`Self::tasks`] (O(1)).
    pub(crate) fn task_index(&self, name: Ustr) -> Option<usize> {
        self.index_of.get(&name).copied()
    }

    /// Resolve a state-field name to its index into [`Self::fields`] (O(1)).
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.field_index.get(&Ustr::from(name)).copied()
    }

    /// The inferred [`TaskSummary`] of a task, by name (O(1)).
    pub fn task_summary(&self, name: &str) -> Option<&TaskSummary> {
        let idx = *self.index_of.get(&Ustr::from(name))?;
        self.summaries.get(idx)
    }

    /// The inferred possible-write set of one method of a compound task, by
    /// task name and method index (O(1)). This is what the forward planner's
    /// look-ahead sweep uses for optimistic state propagation.
    pub fn method_possible_writes(&self, task: &str, method: usize) -> Option<&FieldSet> {
        match self.get_task(task)? {
            Task::Compound(c) => c.methods.get(method).map(|m| &m.possible_writes),
            _ => None,
        }
    }

    /// Build the `name -> index` lookup map and recompute the inferred
    /// task/method summaries. Called at parse time.
    pub(crate) fn rebuild_index(&mut self) {
        self.index_of.clear();
        self.index_of.reserve(self.tasks.len());
        for (i, t) in self.tasks.iter().enumerate() {
            self.index_of.insert(t.name().into(), i);
        }
        crate::summaries::compute_summaries(self);
    }

    /// Look up a goal task by name (for back-planning).
    pub fn goal(&self, name: &str) -> Option<&GoalTask> {
        match self.get_task(name) {
            Some(Task::Goal(g)) => Some(g),
            _ => None,
        }
    }

    /// Validate every task's conditions/effects reference existing state fields,
    /// enums are registered, and every operator is registered.
    pub fn verify<S: Reflect>(&self, state: &S, registry: &TypeRegistry) -> HtnResult<()> {
        let erased = state.as_reflect();
        for task in &self.tasks {
            task.verify(erased, registry)?;
            if let Task::Primitive(p) = task {
                verify_operator(registry, &p.operator.name, &p.operator.params)?;
            }
        }
        Ok(())
    }

    /// Validate tasks that don't reference operators (conditions + effects only).
    /// Useful for tests that check planner output without registered operators.
    pub fn verify_without_operators<S: Reflect>(
        &self,
        state: &S,
        registry: &TypeRegistry,
    ) -> HtnResult<()> {
        let erased = state.as_reflect();
        for task in &self.tasks {
            task.verify(erased, registry)?;
        }
        Ok(())
    }

    /// The name of every primitive task in the domain (for back-planning).
    pub fn primitive_names(&self) -> Vec<Ustr> {
        self.tasks
            .iter()
            .filter_map(|t| match t {
                Task::Primitive(p) => Some(p.name),
                _ => None,
            })
            .collect()
    }
}
