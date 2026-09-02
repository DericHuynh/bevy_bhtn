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
//! // subtask list referencing other task functions directly. A branch with
//! // no `.then` is an **empty terminal branch** — the idiom for a "done"
//! // method: when its precondition holds, the task decomposes to nothing.
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
use std::hash::Hash;
use std::sync::Arc;

use bevy_ecs::system::EntityCommands;
use smallvec::{smallvec, SmallVec};

use crate::domain::SelectionPolicy;
use crate::state::{PlanComponent, PlanState, RegistryBuilder};

// ---------------------------------------------------------------------------
// Precondition / Effect / Action — the type-erased state accessors
// ---------------------------------------------------------------------------

/// A compiled precondition: a type-erased checker over the [`PlanState`]
/// scratchpad plus the slot indices it reads (for summary inference and
/// look-ahead "unknown component" tracking). `Clone` (Arc'd checker) so the
/// GTN compilation can re-capture a task's preconditions inside a synthesized
/// wrapper method.
#[derive(Clone)]
pub struct Precondition {
    check: Arc<dyn Fn(&PlanState) -> bool + Send + Sync>,
    pub(crate) reads: SmallVec<[usize; 4]>,
}

impl Precondition {
    /// Build a precondition directly from a slot-read list and a checker.
    /// `pub(crate)` because the GTN compilation synthesizes preconditions for
    /// its marker components; user code goes through [`IntoPrecondition`].
    pub(crate) fn new(
        reads: SmallVec<[usize; 4]>,
        check: Arc<dyn Fn(&PlanState) -> bool + Send + Sync>,
    ) -> Self {
        Self { check, reads }
    }
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

    /// Build an effect directly from a write list and an apply closure.
    /// `pub(crate)` — the GTN compilation synthesizes marker effects; user
    /// code goes through [`IntoEffect`].
    pub(crate) fn new(
        writes: SmallVec<[usize; 4]>,
        apply: Box<dyn Fn(&mut PlanState) + Send + Sync>,
    ) -> Self {
        Self { apply, writes }
    }
}

/// A compiled real-world action: dispatched against the agent entity's
/// [`EntityCommands`] when the planner executes the primitive. `Arc`-wrapped
/// so the driver can hold a handle while running it against the world.
pub type Action = Arc<dyn Fn(&mut EntityCommands) + Send + Sync>;

/// A compiled dynamic utility/cost scorer over the scratchpad.
pub type ScoreFn = Box<dyn Fn(&PlanState) -> f32 + Send + Sync>;

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
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid precondition closure",
    label = "this closure's parameters don't match any supported precondition signature",
    note = "precondition closures take 0-8 shared component references: `|ammo: &Ammo, vision: &Vision| ...`",
    note = "closure parameters must be annotated with their component types so the planner can find their slots"
)]
pub trait IntoPrecondition<Args> {
    /// Compile into a [`Precondition`], registering read components in
    /// `registry`.
    fn build(self, registry: &mut RegistryBuilder) -> Precondition;
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
            fn build(self, registry: &mut RegistryBuilder) -> Precondition {
                $(let $name = registry.index::<$name>();)*
                #[allow(unused_mut)]
                let reads: SmallVec<[usize; 4]> = smallvec![$($name,)*];
                Precondition {
                    reads,
                    check: Arc::new(move |state| self($(state.get::<$name>($name),)*)),
                }
            }
        }
    };
}

/// Conversion into a compiled [`Effect`], implemented for closures over
/// component references where **`&mut T` marks a component the effect
/// writes** (journaled for rollback and committed to the real entity at
/// execution) and **`&T` marks a read-only component** (costs nothing — it is
/// never journaled or committed). The `Args` parameter is inferred from the
/// closure's `Fn` signature (the axum-handler pattern); closure parameters
/// must be annotated so the concrete component types (and their slot indices)
/// can be captured at build time:
///
/// ```
/// # use bevy_bhtn::tasks::TaskBuilder;
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Cover { in_cover: bool }
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Vision { visible: bool }
/// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
/// # struct Ammo(pub u32);
/// # fn demo(task: &mut TaskBuilder) {
/// // Writes Cover and Vision; reads Ammo.
/// task.effect(|cover: &mut Cover, vision: &mut Vision, ammo: &Ammo| {
///     if ammo.0 > 0 {
///         cover.in_cover = true;
///         vision.visible = false;
///     }
/// });
/// # }
/// ```
///
/// Up to 8 parameters may mix `&` and `&mut` freely at any arity. Read-only
/// parameters are never journaled for rollback and never committed to the real
/// entity — only what the closure actually mutates costs anything. No
/// parameter may name the same component type twice (a `&mut`/`&` pair on one
/// slot would alias).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid effect closure",
    label = "this closure's parameters don't match any supported effect signature",
    note = "effect closures take 0-8 annotated component parameters: `&mut T` marks a written component, `&T` a read-only one",
    note = "any mix of `&` and `&mut` is allowed at any arity up to 8",
    note = "no parameter may name the same component type twice"
)]
pub trait IntoEffect<Args> {
    /// Compile into an [`Effect`], registering written components in
    /// `registry`.
    fn build(self, registry: &mut RegistryBuilder) -> Effect;
}

/// Reject an effect closure that takes the same component type twice: both
/// parameters would resolve to the same registered slot, and a `&mut` pair
/// would alias — a soundness violation. (Repeated *shared* reads would be
/// harmless, but they are rejected too for a single, simple rule.)
/// Shared-reference preconditions may repeat freely.
fn assert_distinct_slots(indices: &[usize]) {
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            assert!(
                indices[i] != indices[j],
                "effect closure takes the same component type twice — a `&mut` \
                 pair would alias; merge the parameters into one `&mut`"
            );
        }
    }
}

// -- Effect-signature machinery ---------------------------------------------
//
// Effect closures take component parameters where `&mut T` marks a component
// the effect *writes* (journaled for rollback and committed to the real
// entity) and `&T` marks a read-only component (never journaled — a read
// costs nothing). The axum-style `Args` marker is the closure's fn-pointer
// type; the [`PlanParam`] trait lets one linear tuple impl per arity accept
// any `&`/`&mut` mix at that arity — no combinatorial pattern enumeration.

/// One parameter of an effect closure: a shared (`&T`) or exclusive (`&mut T`)
/// reference to a registered component slot.
///
/// The reference kind is the type itself, so the [`IntoEffect`] tuple impls
/// unify each closure parameter against exactly one impl — `&mut A` picks the
/// write impl, `&A` the read one — with no combinatorial enumeration of
/// patterns. `IS_WRITE` is a `const`, so the tuple impl's journal/commit-list
/// filter folds at compile time (no runtime branch per parameter).
///
/// Effectively sealed: the only self types are `&T` and `&mut T` for
/// [`PlanComponent`]s, and users never name this trait — any annotated
/// component reference works through the impls automatically.
pub trait PlanParam {
    /// Whether this parameter writes its slot (`&mut T`) or only reads it
    /// (`&T`). Write slots are journaled for rollback and committed to the
    /// real entity at execution; read slots cost nothing.
    const IS_WRITE: bool;

    /// The component's slot index, registering it on first use.
    fn register(registry: &mut RegistryBuilder) -> usize;

    /// Build the closure argument from the slot's raw pointer.
    ///
    /// # Safety
    /// `ptr` must point at the initialized slot for `Self`'s component, and
    /// must be disjoint from every other parameter's slot (guaranteed by the
    /// all-distinct slot assert at effect-build time). The returned reference
    /// must not outlive the scratchpad the pointer points into — the compiled
    /// effect only ever calls this while its `&mut PlanState` borrow is live.
    unsafe fn fetch(ptr: *mut u8) -> Self;
}

impl<T: PlanComponent> PlanParam for &T {
    const IS_WRITE: bool = false;
    fn register(registry: &mut RegistryBuilder) -> usize {
        registry.index::<T>()
    }
    unsafe fn fetch(ptr: *mut u8) -> Self {
        &*(ptr as *const T)
    }
}

impl<T: PlanComponent> PlanParam for &mut T {
    const IS_WRITE: bool = true;
    fn register(registry: &mut RegistryBuilder) -> usize {
        registry.index::<T>()
    }
    unsafe fn fetch(ptr: *mut u8) -> Self {
        &mut *(ptr as *mut T)
    }
}

/// One linear `IntoEffect` impl per arity (0–8): each parameter unifies
/// against [`PlanParam`] independently, so any `&`/`&mut` mix compiles at any
/// arity. `IS_WRITE` is const-folded, so read-only parameters vanish from the
/// write list (never journaled, never committed) at zero runtime cost.
macro_rules! impl_effect_tuple {
    ($($p:ident),*) => {
        #[allow(non_snake_case, unused_variables)]
        impl<F, $($p: PlanParam,)*> IntoEffect<fn($($p,)*)> for F
        where
            F: Fn($($p,)*) + Send + Sync + 'static,
        {
            fn build(self, registry: &mut RegistryBuilder) -> Effect {
                $(let $p = <$p as PlanParam>::register(registry);)*
                // Every parameter's slot must be distinct: a `&mut` pair
                // would alias, and so would a `&mut`/`&` pair.
                let all = &[$($p),*];
                assert_distinct_slots(all);
                #[allow(unused_mut)]
                let mut writes: SmallVec<[usize; 4]> = SmallVec::new();
                $(
                    if <$p as PlanParam>::IS_WRITE {
                        writes.push($p);
                    }
                )*
                Effect {
                    writes,
                    apply: Box::new(move |state| {
                        // The closure's arguments are distinct registered
                        // slots, so their byte regions are disjoint by
                        // construction — the raw pointers never alias.
                        let [$($p),*] = state.disjoint_slots([$($p),*]);
                        self($(
                            unsafe { <$p as PlanParam>::fetch($p) },
                        )*)
                    }),
                }
            }
        }
    };
}
impl_effect_tuple!();
impl_effect_tuple!(A);
impl_effect_tuple!(A, B);
impl_effect_tuple!(A, B, C);
impl_effect_tuple!(A, B, C, D);
impl_effect_tuple!(A, B, C, D, E);
impl_effect_tuple!(A, B, C, D, E, F2);
impl_effect_tuple!(A, B, C, D, E, F2, G);
impl_effect_tuple!(A, B, C, D, E, F2, G, H);

/// Conversion into a compiled `f32` scorer over shared component references
/// (used by `utility_fn`). Same axum-style `Args` inference as
/// [`IntoPrecondition`]; closure parameters must be annotated.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid utility/cost closure",
    label = "this closure's parameters don't match any supported scorer signature",
    note = "utility and cost closures take 0-8 shared component references and return f32: `|d: &Distance| d.0 as f32`",
    note = "closure parameters must be annotated with their component types so the planner can find their slots"
)]
pub trait IntoUtility<Args> {
    /// Compile into a [`ScoreFn`], registering read components in `registry`.
    fn build(self, registry: &mut RegistryBuilder) -> ScoreFn;
}

macro_rules! impl_utility {
    ($($name:ident),*) => {
        #[allow(non_snake_case)]
        impl<F, $($name,)*> IntoUtility<fn($(& $name,)*) -> f32> for F
        where
            F: Fn($(& $name,)*) -> f32 + Send + Sync + 'static,
            $($name: PlanComponent,)*
        {
            #[allow(unused_variables)]
            fn build(self, registry: &mut RegistryBuilder) -> ScoreFn {
                $(let $name = registry.index::<$name>();)*
                Box::new(move |state| self($(state.get::<$name>($name),)*))
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
impl_utility!();
impl_utility!(A);
impl_utility!(A, B);
impl_utility!(A, B, C);
impl_utility!(A, B, C, D);
impl_utility!(A, B, C, D, E);
impl_utility!(A, B, C, D, E, F2);
impl_utility!(A, B, C, D, E, F2, G);
impl_utility!(A, B, C, D, E, F2, G, H);

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
    pub(crate) name: Option<&'static str>,
    pub(crate) utility: Option<ScoreFn>,
    pub(crate) preconditions: Vec<Precondition>,
    /// Subtask references in declaration order, each tagged with whether it
    /// was appended via [`MethodBuilder::then`] (totally ordered after every
    /// prior member) or [`MethodBuilder::subtask`] (unordered relative to
    /// other unordered members).
    pub(crate) subtasks: Vec<(SubtaskRef, &'static str, bool)>,
    /// Whether any [`MethodBuilder::subtask`] was used (the branch is not a
    /// pure `then` chain).
    pub(crate) unordered: bool,
    /// Explicit [`MethodBuilder::before`] constraints as
    /// `(predecessor position, successor position)` pairs.
    pub(crate) edges: Vec<(u32, u32)>,
}

/// The identity of a referenced task in a recorded method body: a task
/// function's zero-sized type (the normal case) or a transform-synthesized
/// task with no function identity (the GTN compilation's wrapper/marker/insert
/// tasks — keyed by a bake-time-unique synthetic id).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum SubtaskRef {
    Fn(TypeId),
    Synthetic(u32),
}

/// What a task function recorded, prior to baking.
pub(crate) enum TaskProto {
    /// Declared only `branch`es — a compound task.
    Compound {
        methods: Vec<MethodProto>,
        policy: SelectionPolicy,
    },
    /// Declared only preconditions/effects/actions — a primitive task.
    Primitive {
        preconditions: Vec<Precondition>,
        effects: Vec<Effect>,
        expected_effects: Vec<Effect>,
        action: Option<Action>,
        cost: Option<ScoreFn>,
        /// The constant declared via [`TaskBuilder::cost`] (if the final cost
        /// signal is a constant) — the bake-time lower bound the `min_cost`
        /// summary infers from. A later [`TaskBuilder::cost_fn`] clears it:
        /// dynamic costs are opaque at bake time and conservatively count 0.
        static_cost: Option<f32>,
    },
}

/// The recording context threaded through task functions during baking.
pub(crate) struct Recorder {
    pub(crate) registry: RegistryBuilder,
    pub(crate) tasks: Vec<(SubtaskRef, &'static str, TaskProto)>,
    pub(crate) index_of: HashMap<SubtaskRef, usize>,
    pub(crate) queue: VecDeque<Box<dyn TaskFn>>,
    pub(crate) errors: Vec<String>,
    /// Next id for transform-synthesized tasks (see [`SubtaskRef::Synthetic`]).
    pub(crate) next_synthetic: u32,
    /// Deferred GTN compilations: task functions marked for sharing (applied
    /// at bake time, before insertion).
    pub(crate) shares: Vec<SubtaskRef>,
    /// Tasks registered as insertion candidates (see `DomainBuilder::insertable`).
    pub(crate) insertables: Vec<SubtaskRef>,
    /// Whether task insertion is compiled in at bake time.
    pub(crate) insertion: bool,
    /// Additional root tasks (see `DomainBuilder::root`) — e.g. an opponent's
    /// plan for adversarial planning. Validated compound at bake.
    pub(crate) extra_roots: Vec<SubtaskRef>,
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
    cost: Option<ScoreFn>,
    static_cost: Option<f32>,
    methods: Vec<MethodProto>,
    selection: Option<SelectionPolicy>,
}

impl<'a> TaskBuilder<'a> {
    pub(crate) fn new(rec: &'a mut Recorder) -> Self {
        Self {
            rec,
            preconditions: Vec::new(),
            effects: Vec::new(),
            expected_effects: Vec::new(),
            action: None,
            cost: None,
            static_cost: None,
            methods: Vec::new(),
            selection: None,
        }
    }

    /// Set this compound task's branch-selection policy (Axis 1). Only
    /// meaningful on compound tasks; setting it on a primitive is a
    /// build-time error.
    pub fn select(&mut self, policy: SelectionPolicy) -> &mut Self {
        self.selection = Some(policy);
        self
    }

    /// Constant action cost, fed to cost-aware search strategies. Inert
    /// under [`DepthFirst`](crate::selection::HtnSearchStrategy::DepthFirst);
    /// the [`CostBounded`](crate::selection::HtnSearchStrategy::CostBounded)
    /// strategy uses it both as the step cost and as the bake-time lower
    /// bound the `min_cost` summary infers. Negative costs are clamped to 0
    /// (branch-and-bound requires non-negative step costs).
    pub fn cost(&mut self, c: f32) -> &mut Self {
        self.static_cost = Some(c.max(0.0));
        self.cost = Some(Box::new(move |_| c));
        self
    }

    /// Dynamic cost sampled from the scratchpad at plan time. Clears any
    /// constant declared by a previous [`TaskBuilder::cost`] (last call wins):
    /// dynamic costs are opaque at bake time, so the `min_cost` summary
    /// conservatively lower-bounds this primitive at 0.
    pub fn cost_fn<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&PlanState) -> f32 + Send + Sync + 'static,
    {
        self.static_cost = None;
        self.cost = Some(Box::new(f));
        self
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
                    policy: self.selection.unwrap_or_default(),
                }
            } else {
                TaskProto::Compound {
                    methods: self.methods,
                    policy: self.selection.unwrap_or_default(),
                }
            }
        } else {
            TaskProto::Primitive {
                preconditions: self.preconditions,
                effects: self.effects,
                expected_effects: self.expected_effects,
                action: self.action,
                cost: self.cost,
                static_cost: self.static_cost,
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
    /// Name this branch (debugging, tracing, rankers).
    pub fn named(&mut self, name: &'static str) -> &mut Self {
        self.proto.name = Some(name);
        self
    }

    /// Static utility score for
    /// [`HighestUtility`](crate::selection::SelectionPolicy::HighestUtility) /
    /// [`WeightedRandom`](crate::selection::SelectionPolicy::WeightedRandom)
    /// selection. Branches without one score 0 under HighestUtility and
    /// weight 1.0 under WeightedRandom.
    pub fn utility(&mut self, u: f32) -> &mut Self {
        self.proto.utility = Some(Box::new(move |_| u));
        self
    }

    /// Dynamic utility scored from components at branch-evaluation time.
    /// Closure parameters must be annotated (same as preconditions).
    pub fn utility_fn<F, Args>(&mut self, f: F) -> &mut Self
    where
        F: IntoUtility<Args>,
    {
        self.proto.utility = Some(f.build(&mut self.rec.registry));
        self
    }

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

    /// Append a subtask at the end of the current total order: it runs after
    /// every member declared before it.
    pub fn then<F: TaskFn>(&mut self, f: F) -> &mut Self {
        let tid = SubtaskRef::Fn(TypeId::of::<F>());
        self.proto.subtasks.push((tid, F::task_name(), true));
        self.rec.queue.push_back(Box::new(f));
        self
    }

    /// Add a subtask with no ordering commitment relative to other unordered
    /// subtasks (it still runs after every [`MethodBuilder::then`] member
    /// declared before it). Returns a handle for [`MethodBuilder::before`]
    /// constraints. The search schedules the branch's unordered members in
    /// any topological order of the constraints, backtracking over
    /// alternatives.
    pub fn subtask<F: TaskFn>(&mut self, f: F) -> SubtaskHandle {
        let pos = self.proto.subtasks.len() as u32;
        let tid = SubtaskRef::Fn(TypeId::of::<F>());
        self.proto.subtasks.push((tid, F::task_name(), false));
        self.proto.unordered = true;
        self.rec.queue.push_back(Box::new(f));
        SubtaskHandle { pos }
    }

    /// Require that `before` completes before `after` starts. Called
    /// multiple times to build a constraint DAG over this branch's members;
    /// a cycle is a build-time error.
    pub fn before(&mut self, before: SubtaskHandle, after: SubtaskHandle) -> &mut Self {
        self.proto.edges.push((before.pos, after.pos));
        self
    }

    /// All subtasks in the set may execute in any order (each exactly once).
    /// Sugar for repeated [`MethodBuilder::subtask`] with no
    /// [`MethodBuilder::before`] constraints.
    ///
    /// Takes a **tuple** of task functions (up to 8), matching Bevy's tuple
    /// convention — an array would coerce every distinct function item to
    /// the same `fn(&mut TaskBuilder)` pointer type and collapse their
    /// identities (see [`AnyOrder`]).
    ///
    /// ```
    /// # use bevy_bhtn::tasks::TaskBuilder;
    /// # #[derive(bevy_ecs::prelude::Component, Clone, Default)]
    /// # struct Gold(i32);
    /// fn root(task: &mut TaskBuilder) {
    ///     // The three gathers may run in any order; the search backtracks
    ///     // over linearizations if one order fails.
    ///     task.branch().any_order((gather_wood, gather_food, gather_stone));
    /// }
    /// # fn gather_wood(task: &mut TaskBuilder) { task.effect(|g: &mut Gold| g.0 += 1); }
    /// # fn gather_food(task: &mut TaskBuilder) { task.effect(|g: &mut Gold| g.0 += 1); }
    /// # fn gather_stone(task: &mut TaskBuilder) { task.effect(|g: &mut Gold| g.0 += 1); }
    /// ```
    pub fn any_order<T: AnyOrder>(&mut self, tasks: T) -> &mut Self {
        tasks.any_order(self);
        self
    }
}

/// A handle to one member of a branch's subtask set, returned by
/// [`MethodBuilder::subtask`] and consumed by [`MethodBuilder::before`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubtaskHandle {
    pub(crate) pos: u32,
}

/// Task-function groups accepted by [`MethodBuilder::any_order`].
///
/// Implemented for tuples of up to 8 task functions. Tuples (rather than
/// arrays or iterators) are required for identity: an array `[a, b, c]` of
/// distinct task functions coerces every element to the single fn-pointer
/// type `fn(&mut TaskBuilder)`, so all members would share one `TypeId` and
/// collapse into the same task. A tuple keeps each function item's own
/// zero-sized type — the crate's core identity mechanism.
pub trait AnyOrder {
    /// Record every member as an unordered subtask of the branch.
    fn any_order(self, b: &mut MethodBuilder<'_>);
}

macro_rules! impl_any_order {
    ($($name:ident),*) => {
        #[allow(non_snake_case, unused_variables)]
        impl<$($name: TaskFn),*> AnyOrder for ($($name,)*) {
            fn any_order(self, b: &mut MethodBuilder<'_>) {
                let ($($name,)*) = self;
                $(
                    b.subtask($name);
                )*
            }
        }
    };
}

impl_any_order!();
impl_any_order!(A);
impl_any_order!(A, B);
impl_any_order!(A, B, C);
impl_any_order!(A, B, C, D);
impl_any_order!(A, B, C, D, E);
impl_any_order!(A, B, C, D, E, F2);
impl_any_order!(A, B, C, D, E, F2, G);
impl_any_order!(A, B, C, D, E, F2, G, H);

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
