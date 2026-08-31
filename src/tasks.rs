//! HTN task types: the recording builders (what developers write) and the
//! baked nodes (what the planner runs on).
//!
//! # Writing tasks
//!
//! Tasks are **plain Rust functions**. Every named function has a unique,
//! unnameable zero-sized type, so the function itself is the task's identity —
//! no marker structs, no string ids, no registration:
//!
//! ```
//! use bevy_bhtn::tasks::TaskBuilder;
//! use bevy_bhtn::state::PlanComponent;
//! use bevy_ecs::prelude::*;
//!
//! #[derive(Component, Clone, Default, Debug)]
//! struct Ammo(pub u32);
//!
//! #[derive(Component, Clone, Default, Debug)]
//! struct TargetVision { visible: bool }
//!
//! #[derive(Component, Default)]
//! struct ReloadAction;
//!
//! // A primitive: preconditions read components, effects mutate them (on the
//! // planning scratchpad), the action dispatches real ECS commands.
//! fn reload(task: &mut TaskBuilder) {
//!     task.effect(|ammo: &mut Ammo| ammo.0 = 30)
//!         .action(|cmds: &mut EntityCommands| {
//!             cmds.insert(ReloadAction);
//!         });
//! }
//!
//! // A compound: ordered branches, each with optional preconditions and a
//! // subtask list referencing other task functions directly.
//! fn engage_target(task: &mut TaskBuilder) {
//!     task.branch()
//!         .precondition(|ammo: &Ammo, vision: &TargetVision| ammo.0 > 0 && vision.visible)
//!         .then(reload);
//! }
//! ```
//!
//! # Recording & baking
//!
//! [`HtnDomain::from_root`](crate::domain::HtnDomain::from_root) calls the root
//! function with a **recording** [`TaskBuilder`]; every `.then(shoot)` captures
//! the callee's `TypeId` and queues the function for expansion. The result is
//! baked into flat `Vec`s of [`CompoundTask`]/[`PrimitiveTask`] nodes with
//! contiguous `usize` indices — the runtime planner never calls task functions
//! again; it searches the arrays directly.

use std::any::{type_name, TypeId};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use bevy_ecs::system::EntityCommands;
use smallvec::{smallvec, SmallVec};
use ustr::Ustr;

use crate::state::{ComponentRegistry, PlanComponent, PlanState};
use crate::summaries::FieldSet;

// ---------------------------------------------------------------------------
// Precondition / Effect / Action — the type-erased state accessors
// ---------------------------------------------------------------------------

/// A compiled precondition: a type-erased checker over the [`PlanState`]
/// scratchpad plus the slot indices it reads (for summary inference and
/// look-ahead "unknown component" tracking).
pub struct Precondition {
    check: Box<dyn Fn(&PlanState) -> bool + Send + Sync>,
    pub(crate) reads: SmallVec<[usize; 4]>,
}

impl Precondition {
    /// Evaluate against the scratchpad.
    pub fn evaluate(&self, state: &PlanState) -> bool {
        (self.check)(state)
    }

    /// The slot indices this precondition reads.
    pub fn reads(&self) -> &[usize] {
        &self.reads
    }
}

/// A compiled effect: a type-erased mutator over the [`PlanState`] scratchpad
/// plus the slot indices it writes.
pub struct Effect {
    apply: Box<dyn Fn(&mut PlanState) + Send + Sync>,
    pub(crate) writes: SmallVec<[usize; 4]>,
}

impl Effect {
    /// Apply to the scratchpad.
    pub fn apply(&self, state: &mut PlanState) {
        (self.apply)(state)
    }

    /// The slot indices this effect writes.
    pub fn writes(&self) -> &[usize] {
        &self.writes
    }
}

/// A compiled real-world action: dispatched against the agent entity's
/// [`EntityCommands`] when the planner executes the primitive. `Arc`-wrapped
/// so the driver can hold a handle while running it against the world.
pub type Action = Arc<dyn Fn(&mut EntityCommands) + Send + Sync>;

/// Conversion into a compiled [`Precondition`], implemented (up to 8
/// parameters) for closures over shared component references. The `Args`
/// parameter is inferred from the closure's `Fn` signature (the axum-handler
/// pattern), so call sites need no disambiguation — but closure parameters
/// must be annotated so the concrete component types (and their slot indices)
/// can be captured at build time:
///
/// ```
/// # use bevy_bhtn::tasks::TaskBuilder;
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Ammo(pub u32);
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Vision { visible: bool }
/// # fn demo(task: &mut TaskBuilder) {
/// task.precondition(|ammo: &Ammo, vision: &Vision| ammo.0 > 0 && vision.visible);
/// # }
/// ```
pub trait IntoPrecondition<Args> {
    /// Compile into a [`Precondition`], registering read components in
    /// `registry`.
    fn build(self, registry: &mut ComponentRegistry) -> Precondition;
}

macro_rules! impl_precondition {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        impl<F, $($name,)*> IntoPrecondition<fn($(& $name,)*) -> bool> for F
        where
            F: Fn($(& $name,)*) -> bool + Send + Sync + 'static,
            $($name: PlanComponent,)*
        {
            #[allow(unused_variables)]
            fn build(self, registry: &mut ComponentRegistry) -> Precondition {
                $(let $name = registry.index::<$name>();)*
                #[allow(unused_mut)]
                let reads: SmallVec<[usize; 4]> = smallvec![$($name,)*];
                Precondition {
                    reads,
                    check: Box::new(move |state| self($(state.get::<$name>($name),)*)),
                }
            }
        }
    };
}

/// Conversion into a compiled [`Effect`], implemented (up to 8 parameters)
/// for closures over exclusive component references. The `Args` parameter is
/// inferred from the closure's `Fn` signature; closure parameters must be
/// annotated:
///
/// ```
/// # use bevy_bhtn::tasks::TaskBuilder;
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Cover { in_cover: bool }
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Vision { visible: bool }
/// # fn demo(task: &mut TaskBuilder) {
/// task.effect(|cover: &mut Cover, vision: &mut Vision| {
///     cover.in_cover = true;
///     vision.visible = false;
/// });
/// # }
/// ```
pub trait IntoEffect<Args> {
    /// Compile into an [`Effect`], registering written components in
    /// `registry`.
    fn build(self, registry: &mut ComponentRegistry) -> Effect;
}

/// Reject an effect closure that takes `&mut` to the same component type
/// twice: both parameters would resolve to the same registered slot, and the
/// compiled closure would create two aliasing `&mut` — a soundness violation.
/// Shared-reference preconditions may repeat freely.
fn assert_distinct_slots(indices: &[usize]) {
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            assert!(
                indices[i] != indices[j],
                "effect closure takes `&mut` to the same component type twice — \
                 the slots would alias; merge the parameters into one `&mut`"
            );
        }
    }
}

macro_rules! impl_effect {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        impl<F, $($name,)*> IntoEffect<fn($(&mut $name,)*)> for F
        where
            F: Fn($(&mut $name,)*) + Send + Sync + 'static,
            $($name: PlanComponent,)*
        {
            fn build(self, registry: &mut ComponentRegistry) -> Effect {
                $(let $name = registry.index::<$name>();)*
                #[allow(unused_mut)]
                let writes: SmallVec<[usize; 4]> = smallvec![$($name,)*];
                assert_distinct_slots(&writes);
                Effect {
                    writes,
                    apply: Box::new(move |state| {
                        // The closure's arguments are distinct registered
                        // slots, so their byte regions are disjoint by
                        // construction — the raw pointers never alias.
                        let [$($name),*] = state.disjoint_slots([$($name,)*]);
                        self($(
                            unsafe { &mut *($name as *mut $name) },
                        )*)
                    }),
                }
            }
        }
    };
}

impl_precondition!();
impl_precondition!(A);
impl_precondition!(A, B);
impl_precondition!(A, B, C);
impl_precondition!(A, B, C, D);
impl_precondition!(A, B, C, D, E);
impl_precondition!(A, B, C, D, E, F2);
impl_precondition!(A, B, C, D, E, F2, G);
impl_precondition!(A, B, C, D, E, F2, G, H);

impl_effect!(A);
impl_effect!(A, B);
impl_effect!(A, B, C);
impl_effect!(A, B, C, D);
impl_effect!(A, B, C, D, E);
impl_effect!(A, B, C, D, E, F2);
impl_effect!(A, B, C, D, E, F2, G);
impl_effect!(A, B, C, D, E, F2, G, H);

// ---------------------------------------------------------------------------
// TaskFn — function items as task identity
// ---------------------------------------------------------------------------

/// A task function: any `Fn(&mut TaskBuilder)` with a `'static` type.
///
/// Every named function has a unique zero-sized type, so the function item
/// itself is the task's identity: `TypeId::of::<F>()` is the graph node id and
/// `std::any::type_name::<F>()` supplies the debug name. No marker structs,
/// no strings, no registration.
pub trait TaskFn: 'static {
    /// Run this function against a recording builder.
    fn record(&self, builder: &mut TaskBuilder);

    /// This function's `TypeId` (type-erased access for the expansion loop).
    fn task_type_id(&self) -> TypeId;

    /// The task's clean debug name, available through the erased trait object.
    fn task_name_erased(&self) -> &'static str;

    /// The task's clean debug name (the last path segment of
    /// `std::any::type_name::<Self>()`).
    fn task_name() -> &'static str
    where
        Self: Sized,
    {
        let full = type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full)
    }
}

impl<F: Fn(&mut TaskBuilder) + 'static> TaskFn for F {
    fn record(&self, builder: &mut TaskBuilder) {
        self(builder)
    }

    fn task_type_id(&self) -> TypeId {
        TypeId::of::<F>()
    }

    fn task_name_erased(&self) -> &'static str {
        F::task_name()
    }
}

/// A goal function: any `Fn(&mut GoalBuilder)` with a `'static` type. Same
/// identity scheme as [`TaskFn`] — the function item is the goal's name and
/// identity.
pub trait GoalFn: 'static {
    /// Run this function against a recording goal builder.
    fn record(&self, builder: &mut GoalBuilder);

    /// The goal's clean debug name.
    fn goal_name() -> &'static str
    where
        Self: Sized,
    {
        let full = type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full)
    }
}

impl<F: Fn(&mut GoalBuilder) + 'static> GoalFn for F {
    fn record(&self, builder: &mut GoalBuilder) {
        self(builder)
    }
}

// ---------------------------------------------------------------------------
// Recording builders
// ---------------------------------------------------------------------------

/// One recorded branch of a compound task (a method), prior to baking.
#[derive(Default)]
pub(crate) struct MethodProto {
    pub(crate) preconditions: Vec<Precondition>,
    pub(crate) subtasks: Vec<(TypeId, &'static str)>,
}

/// What a task function recorded, prior to baking.
pub(crate) enum TaskProto {
    /// Declared only `branch`es — a compound task.
    Compound { methods: Vec<MethodProto> },
    /// Declared only preconditions/effects/actions — a primitive task.
    Primitive {
        preconditions: Vec<Precondition>,
        effects: Vec<Effect>,
        expected_effects: Vec<Effect>,
        action: Option<Action>,
    },
}

/// The recording context threaded through task functions during baking.
pub(crate) struct Recorder {
    pub(crate) registry: ComponentRegistry,
    pub(crate) tasks: Vec<(TypeId, &'static str, TaskProto)>,
    pub(crate) index_of: HashMap<TypeId, usize>,
    pub(crate) queue: VecDeque<Box<dyn TaskFn>>,
    pub(crate) errors: Vec<String>,
}

/// The builder handed to task functions. Its API surface determines the
/// task's kind: calling [`TaskBuilder::branch`] makes it a compound task;
/// calling [`TaskBuilder::effect`] / [`TaskBuilder::precondition`] /
/// [`TaskBuilder::action`] makes it a primitive. Mixing the two is a
/// build-time error.
pub struct TaskBuilder<'a> {
    rec: &'a mut Recorder,
    preconditions: Vec<Precondition>,
    effects: Vec<Effect>,
    expected_effects: Vec<Effect>,
    action: Option<Action>,
    methods: Vec<MethodProto>,
}

impl<'a> TaskBuilder<'a> {
    pub(crate) fn new(rec: &'a mut Recorder) -> Self {
        Self {
            rec,
            preconditions: Vec::new(),
            effects: Vec::new(),
            expected_effects: Vec::new(),
            action: None,
            methods: Vec::new(),
        }
    }

    /// Add a precondition (all must hold for a primitive to be pickable, or
    /// for a method's subtask list to be chosen).
    pub fn precondition<P, Args>(&mut self, p: P) -> &mut Self
    where
        P: IntoPrecondition<Args>,
    {
        self.preconditions.push(p.build(&mut self.rec.registry));
        self
    }

    /// Add an effect — applied to the planning scratchpad during search **and**
    /// to the real entity when the task executes.
    pub fn effect<E, Args>(&mut self, e: E) -> &mut Self
    where
        E: IntoEffect<Args>,
    {
        self.effects.push(e.build(&mut self.rec.registry));
        self
    }

    /// Add an anticipated (planning-only, non-guaranteed) effect — applied to
    /// the scratchpad during search but never to the real entity.
    pub fn expected<E, Args>(&mut self, e: E) -> &mut Self
    where
        E: IntoEffect<Args>,
    {
        self.expected_effects.push(e.build(&mut self.rec.registry));
        self
    }

    /// Set the real-world action dispatched when the task executes. The
    /// closure receives the agent entity's [`EntityCommands`] (standard Bevy
    /// command dispatch; the driver flushes after each step).
    pub fn action<F: Fn(&mut EntityCommands) + Send + Sync + 'static>(
        &mut self,
        f: F,
    ) -> &mut Self {
        self.action = Some(Arc::new(f));
        self
    }

    /// Begin a new decomposition branch (method). Branches are tried in
    /// declaration order; the first whose preconditions hold is chosen.
    /// Calling this marks the task as compound.
    pub fn branch(&mut self) -> MethodBuilder<'_> {
        self.methods.push(MethodProto::default());
        MethodBuilder {
            rec: &mut *self.rec,
            proto: self.methods.last_mut().expect("just pushed"),
        }
    }

    pub(crate) fn finish(self) {
        let kind = if !self.methods.is_empty() {
            if !self.preconditions.is_empty()
                || !self.effects.is_empty()
                || !self.expected_effects.is_empty()
                || self.action.is_some()
            {
                self.rec.errors.push(
                    "task mixes compound (`branch`) and primitive (`precondition`/`effect`/`action`) declarations".into(),
                );
                TaskProto::Compound {
                    methods: self.methods,
                }
            } else {
                TaskProto::Compound {
                    methods: self.methods,
                }
            }
        } else {
            TaskProto::Primitive {
                preconditions: self.preconditions,
                effects: self.effects,
                expected_effects: self.expected_effects,
                action: self.action,
            }
        };
        // The task's own TypeId was registered by the expansion loop before
        // this builder was created; replace its placeholder proto.
        let last = self.rec.tasks.last_mut().expect("placeholder registered");
        last.2 = kind;
    }
}

/// Configure one branch inside [`TaskBuilder::branch`]. Committed on drop.
pub struct MethodBuilder<'a> {
    rec: &'a mut Recorder,
    proto: &'a mut MethodProto,
}

impl<'a> MethodBuilder<'a> {
    /// Add a precondition for this branch to be chosen.
    pub fn precondition<P, Args>(&mut self, p: P) -> &mut Self
    where
        P: IntoPrecondition<Args>,
    {
        self.proto
            .preconditions
            .push(p.build(&mut self.rec.registry));
        self
    }

    /// Append a subtask, referenced by its function. The function is queued
    /// for expansion (each task function is recorded exactly once; recursive
    /// and repeated references become plain graph edges).
    pub fn then<F: TaskFn>(&mut self, f: F) -> &mut Self {
        let tid = TypeId::of::<F>();
        self.proto.subtasks.push((tid, F::task_name()));
        self.rec.queue.push_back(Box::new(f));
        self
    }
}

/// Configure a goal task (a set of desired effects for back-planning).
pub struct GoalBuilder<'a> {
    rec: &'a mut Recorder,
    effects: Vec<Effect>,
}

impl<'a> GoalBuilder<'a> {
    pub(crate) fn new(rec: &'a mut Recorder) -> Self {
        Self {
            rec,
            effects: Vec::new(),
        }
    }

    /// Add one desired effect.
    pub fn effect<E, Args>(&mut self, e: E) -> &mut Self
    where
        E: IntoEffect<Args>,
    {
        self.effects.push(e.build(&mut self.rec.registry));
        self
    }

    pub(crate) fn finish(self) -> Vec<Effect> {
        let _ = self.rec;
        self.effects
    }
}

// ---------------------------------------------------------------------------
// Baked nodes — what the planner searches
// ---------------------------------------------------------------------------

/// One baked decomposition alternative of a compound task. Its preconditions
/// must all hold for its subtask list to be selected.
pub struct Method {
    /// Conditions that must all hold for this method to be chosen.
    pub preconditions: Vec<Precondition>,
    /// Subtask indices into [`HtnDomain::tasks`](crate::domain::HtnDomain),
    /// in execution order.
    pub subtasks: Vec<u32>,
    /// Fields that *some* refinement of this method's subtasks may write
    /// (bake-time over-approximation; computed with the summaries).
    pub(crate) possible_writes: FieldSet,
}

/// A baked compound task: on decomposition, pick the first method whose
/// preconditions evaluate true.
pub struct CompoundTask {
    /// The task's clean function name.
    pub name: &'static str,
    /// The task function's `TypeId` (graph identity for introspection).
    pub type_id: TypeId,
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
}

/// A baked primitive (leaf) task: preconditions, effects, expected effects,
/// and the real-world action. `expected_effects` are simulated during
/// planning but are only *hoped* effects (never applied to the real entity).
pub struct PrimitiveTask {
    /// The task's clean function name.
    pub name: &'static str,
    /// The task function's `TypeId`.
    pub type_id: TypeId,
    /// Conditions that must all hold for this task to be pickable.
    pub preconditions: Vec<Precondition>,
    /// Effects applied to the scratchpad during search and to the real
    /// entity at execution.
    pub effects: Vec<Effect>,
    /// Anticipated (non-guaranteed) effects applied during planning only.
    pub expected_effects: Vec<Effect>,
    /// The real-world action dispatched at execution (if any).
    pub action: Option<Action>,
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
        self.effects
            .iter()
            .chain(self.expected_effects.iter())
            .flat_map(|e| e.writes.iter().copied())
    }

    /// The write slots of the guaranteed (real) effects only.
    pub fn guaranteed_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.effects.iter().flat_map(|e| e.writes.iter().copied())
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
