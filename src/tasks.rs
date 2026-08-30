//! HTN task types: compound tasks (methods), primitive tasks (operators), and
//! goal tasks (back-planning targets).
//!
//! Every task name, method name, and subtask reference is stored as a
//! [`Ustr`] — an interned, `Copy` string handle. Within a domain there are few
//! unique names but many references (a subtask can be listed dozens of times),
//! so interning makes comparing/substituting names a single pointer compare and
//! keeps the domain compact. The `&str` API (`name()`, `get_task`) is preserved
//! via deref coercion.

use bevy_reflect::{Reflect, TypeRegistry};
use ustr::Ustr;

use crate::conditions::HtnCondition;
use crate::effects::Effect;
use crate::error::HtnResult;
use crate::operators::Operator;
use crate::summaries::FieldSet;

// ---------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------

/// One decomposition alternative of a compound task. Its preconditions must all
/// hold for its subtask list to be selected.
#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    /// Optional human-readable name (from `method "foo" { ... }`).
    pub name: Option<Ustr>,
    /// Conditions that must all hold for this method to be chosen.
    pub preconditions: Vec<HtnCondition>,
    /// Names of subtasks (compound or primitive) chosen by this method.
    pub subtasks: Vec<Ustr>,
    /// Fields that *some* refinement of this method's subtasks may write
    /// (parse-time over-approximation; empty until summaries are computed).
    pub(crate) possible_writes: FieldSet,
    /// Per-precondition read-field indices (parallel to [`Self::preconditions`],
    /// `[primary, comparison]`), precomputed at parse time so the look-ahead
    /// sweep's known/unknown checks need no per-call hash lookups. Empty until
    /// summaries are computed.
    pub(crate) prec_reads: Vec<[Option<usize>; 2]>,
}

impl Method {
    /// Verify every precondition's referenced fields exist on `state`.
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        for c in &self.preconditions {
            c.verify(state, registry)?;
        }
        Ok(())
    }
}
// ---------------------------------------------------------------------------
// CompoundTask
// ---------------------------------------------------------------------------

/// A compound task: on decomposition, pick the first method (at or after `skip`)
/// whose preconditions evaluate true.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundTask {
    /// Task name.
    pub name: Ustr,
    /// Ordered decomposition alternatives.
    pub methods: Vec<Method>,
}

impl CompoundTask {
    /// First method (at or after `skip`) whose preconditions hold.
    pub fn find_method<'a>(
        &'a self,
        state: &dyn Reflect,
        skip: usize,
    ) -> Option<(&'a Method, usize)> {
        self.methods
            .iter()
            .enumerate()
            .skip(skip)
            .find(|(_i, m)| m.preconditions.iter().all(|c| c.evaluate(state)))
            .map(|(i, m)| (m, i))
    }

    /// Verify preconditions of every method.
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        for m in &self.methods {
            m.verify(state, registry)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PrimitiveTask
// ---------------------------------------------------------------------------

/// A primitive (leaf) task: an operator plus preconditions, effects, and
/// expected effects. `expected_effects` are simulated during planning but are
/// only *hoped* effects (not guaranteed) — used to let later tasks' preconditions
/// rely on a chain of effects.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveTask {
    /// Task name.
    pub name: Ustr,
    /// The operator that executes this task at run time.
    pub operator: Operator,
    /// Conditions that must all hold for this task to be pickable.
    pub preconditions: Vec<HtnCondition>,
    /// Effects applied when the task completes.
    pub effects: Vec<Effect>,
    /// Anticipated (non-guaranteed) effects applied during planning only.
    pub expected_effects: Vec<Effect>,
    /// Per-precondition read-field indices (parallel to
    /// [`Self::preconditions`], `[primary, comparison]`), precomputed at parse
    /// time so the look-ahead sweep's known/unknown checks need no per-call
    /// hash lookups. Empty until summaries are computed.
    pub(crate) prec_reads: Vec<[Option<usize>; 2]>,
}

impl PrimitiveTask {
    /// True when every precondition holds for `state` (reflection-based).
    pub fn preconditions_met(&self, state: &dyn Reflect) -> bool {
        self.preconditions.iter().all(|c| c.evaluate(state))
    }

    /// Apply effects (and expected effects) to a work state during search.
    pub fn apply_effects(&self, state: &mut dyn Reflect, registry: &TypeRegistry) {
        for e in &self.effects {
            e.apply_dyn(state, registry);
        }
        for e in &self.expected_effects {
            e.apply_dyn(state, registry);
        }
    }

    /// Verify preconditions, effects, and the operator registration.
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        for c in &self.preconditions {
            c.verify(state, registry)?;
        }
        for e in self.effects.iter().chain(self.expected_effects.iter()) {
            e.verify(state, registry)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GoalTask
// ---------------------------------------------------------------------------

/// `goal_task` — a named set of [`Effect`]s describing a desired end state. The
/// back-planner runs with a goal task as its target.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalTask {
    /// Task name.
    pub name: Ustr,
    /// The desired effects that form the goal state.
    pub effects: Vec<Effect>,
}

impl GoalTask {
    /// Verify every effect's referenced field exists on `state`.
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        for e in &self.effects {
            e.verify(state, registry)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/// One task in the domain.
#[derive(Debug, Clone, PartialEq)]
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
            Task::Primitive(t) => &t.name,
            Task::Compound(t) => &t.name,
            Task::Goal(t) => &t.name,
        }
    }

    /// Whether this is the root (first compound task) for forward planning.
    pub fn is_root_compound(&self) -> bool {
        matches!(self, Task::Compound(_))
    }

    /// Verify this task against the state type and registry.
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        match self {
            Task::Primitive(p) => p.verify(state, registry),
            Task::Compound(c) => c.verify(state, registry),
            Task::Goal(g) => g.verify(state, registry),
        }
    }
}
