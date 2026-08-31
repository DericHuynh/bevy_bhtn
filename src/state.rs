//! The planning scratchpad: a dense snapshot of the components an agent's
//! domain touches, plus the registry that maps component types to slot
//! indices.
//!
//! An entity's state in Bevy is distributed across independent components, so
//! the planner never sees a monolithic struct. Instead, at plan time it
//! **extracts** every registered component from the real entity into a
//! [`PlanState`] (missing components materialize as `Default`), runs the whole
//! forward search on that isolated scratchpad — preconditions read it, effects
//! mutate it, backtracking rolls it back — and never touches the `World`.
//!
//! Component access is fully monomorphized: a precondition closure
//! `|ammo: &Ammo, vision: &TargetVision| ...` is compiled (via the
//! [`crate::tasks::IntoPrecondition`] impls) into a type-erased checker that
//! captured its components' slot indices at build time, so per-evaluation cost
//! is a downcast plus the closure body — no reflection, no hash lookups.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Component;
use bevy_ecs::world::World;

/// A component that can participate in planning: it must be an ordinary
/// [`Component`] that is [`Clone`] (scratchpad snapshots + rollback),
/// [`Default`] (missing components materialize as their default), and
/// `Send + Sync` (planning may run on any thread).
///
/// Any ordinary `#[derive(Component, Clone, Default)]` struct satisfies this
/// via the blanket impl.
pub trait PlanComponent: Component + Clone + Default + Send + Sync + 'static {}
impl<T: Component + Clone + Default + Send + Sync + 'static> PlanComponent for T {}

/// Type-erased, cloneable component value stored in a [`PlanState`] slot.
pub trait ErasedValue: Any + Send + Sync {
    /// Deep-clone the value as a new erased box.
    fn clone_value(&self) -> Box<dyn ErasedValue>;
    /// Read the concrete value as `dyn Any` (for downcasting).
    fn as_any(&self) -> &dyn Any;
    /// Mutably access the concrete value as `dyn Any` (for downcasting).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: PlanComponent> ErasedValue for T {
    fn clone_value(&self) -> Box<dyn ErasedValue> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A type-erased per-component default constructor. Captured at registration
/// time via monomorphization — no reflection.
pub type DefaultFn = fn() -> Box<dyn ErasedValue>;

/// A type-erased per-component fetcher: clones the component off `entity` if
/// present. Captured at registration time via monomorphization.
pub type FetchFn = fn(&World, Entity) -> Option<Box<dyn ErasedValue>>;

/// A type-erased per-component writer: copies an erased value back onto
/// `entity`'s component storage.
pub type WriteFn = fn(&mut World, Entity, &dyn Any);

/// Monomorphized default constructor for one component type.
pub fn default_fn<T: PlanComponent>() -> DefaultFn {
    || Box::new(T::default())
}

/// Monomorphized fetcher for one component type.
pub fn fetch_fn<T: PlanComponent>() -> FetchFn {
    |world, entity| {
        world
            .get::<T>(entity)
            .map(|v| Box::new(v.clone()) as Box<dyn ErasedValue>)
    }
}

/// Monomorphized writer for one component type.
pub fn write_fn<T: PlanComponent>() -> WriteFn {
    |world, entity, value| {
        if let Some(v) = value.downcast_ref::<T>() {
            if let Ok(mut e) = world.get_entity_mut(entity) {
                e.insert(v.clone());
            }
        }
    }
}

/// The registry of component types a domain touches, assigning each a dense
/// slot index plus its monomorphized default/fetch/write fn pointers. Built
/// once at domain-bake time; the index space is the universe the summary
/// [`FieldSet`](crate::summaries::FieldSet)s range over.
#[derive(Debug, Clone, Default)]
pub struct ComponentRegistry {
    types: Vec<(TypeId, &'static str)>,
    defaults: Vec<DefaultFn>,
    fetchers: Vec<FetchFn>,
    writers: Vec<WriteFn>,
    index: HashMap<TypeId, usize>,
}

impl ComponentRegistry {
    /// The slot index of component `T`, registering it (with its monomorphized
    /// default/fetch/write fns) on first use.
    pub fn index<T: PlanComponent>(&mut self) -> usize {
        *self.index.entry(TypeId::of::<T>()).or_insert_with(|| {
            self.types
                .push((TypeId::of::<T>(), std::any::type_name::<T>()));
            self.defaults.push(default_fn::<T>());
            self.fetchers.push(fetch_fn::<T>());
            self.writers.push(write_fn::<T>());
            self.types.len() - 1
        })
    }

    /// Whether component `T` is registered.
    pub fn contains<T: 'static>(&self) -> bool {
        self.index.contains_key(&TypeId::of::<T>())
    }

    /// The slot index of component `T`, if registered.
    pub fn get<T: 'static>(&self) -> Option<usize> {
        self.index.get(&TypeId::of::<T>()).copied()
    }

    /// The number of registered components (the
    /// [`FieldSet`](crate::summaries::FieldSet) universe size).
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Whether no components are registered.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// The registered component's short type name, by slot index.
    pub fn name_of(&self, idx: usize) -> &'static str {
        let full = self.types[idx].1;
        full.rsplit("::").next().unwrap_or(full)
    }
}

/// The planning scratchpad: a dense array of cloned components, one slot per
/// [`ComponentRegistry`] entry.
///
/// The forward search runs entirely on this snapshot — effects mutate slots,
/// backtracking restores them — so planning never mutates the real `World`.
#[derive(Default)]
pub struct PlanState {
    slots: Vec<Option<Box<dyn ErasedValue>>>,
}

impl PlanState {
    /// Extract a scratchpad from `entity`: every registered component is
    /// cloned out of the world, or materialized as `Default` when absent.
    pub fn extract(world: &World, entity: Entity, registry: &ComponentRegistry) -> Self {
        let mut slots: Vec<Option<Box<dyn ErasedValue>>> = Vec::with_capacity(registry.len());
        slots.resize_with(registry.len(), || None);
        for (i, fetch) in registry.fetchers.iter().enumerate() {
            slots[i] = Some(fetch(world, entity).unwrap_or_else(registry.defaults[i]));
        }
        Self { slots }
    }

    /// Begin building a scratchpad directly from component values (no `World`
    /// needed). Unset slots materialize as `Default` on
    /// [`finish`](PlanStateBuilder::finish).
    pub fn build(registry: &ComponentRegistry) -> PlanStateBuilder<'_> {
        PlanStateBuilder {
            registry,
            slots: (0..registry.len()).map(|_| None).collect(),
        }
    }

    /// Write the given slots back onto `entity`'s real components, using
    /// `registry`'s monomorphized writers. Only the listed slots are
    /// committed; call this after mutating the scratchpad to apply simulated
    /// effects to the world.
    pub fn write_back_with(
        &self,
        world: &mut World,
        entity: Entity,
        registry: &ComponentRegistry,
        indices: &[usize],
    ) {
        for &i in indices {
            if let Some(slot) = &self.slots[i] {
                (registry.writers[i])(world, entity, slot.as_any());
            }
        }
    }

    /// Read component `T` at slot `idx`.
    ///
    /// Panics if `idx` is not `T`'s registered slot — closures capture their
    /// indices from the same registry that sized the scratchpad, so this can
    /// only fail on a caller bug.
    pub fn get<T: PlanComponent>(&self, idx: usize) -> &T {
        self.slots[idx]
            .as_ref()
            .expect("scratchpad slot is materialized")
            .as_any()
            .downcast_ref::<T>()
            .expect("slot holds the registered component type")
    }

    /// Mutably read component `T` at slot `idx` (same contract as [`Self::get`]).
    pub fn get_mut<T: PlanComponent>(&mut self, idx: usize) -> &mut T {
        self.slots[idx]
            .as_mut()
            .expect("scratchpad slot is materialized")
            .as_any_mut()
            .downcast_mut::<T>()
            .expect("slot holds the registered component type")
    }

    /// The raw slot array (for disjoint multi-slot mutable access via
    /// `[T]::get_disjoint_mut` in the compiled effect closures).
    pub(crate) fn slots_mut(&mut self) -> &mut [Option<Box<dyn ErasedValue>>] {
        &mut self.slots
    }

    /// Snapshot one slot for rollback.
    pub(crate) fn snapshot(&self, idx: usize) -> Option<Box<dyn ErasedValue>> {
        self.slots[idx].as_ref().map(|s| s.clone_value())
    }

    /// Restore one slot from a rollback snapshot.
    pub(crate) fn restore(&mut self, idx: usize, value: Option<Box<dyn ErasedValue>>) {
        self.slots[idx] = value;
    }

    /// The number of slots (registry size at extraction time).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the scratchpad has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl std::fmt::Debug for PlanState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let filled = self.slots.iter().filter(|s| s.is_some()).count();
        f.debug_struct("PlanState")
            .field("slots", &self.slots.len())
            .field("filled", &filled)
            .finish()
    }
}

impl Clone for PlanState {
    fn clone(&self) -> Self {
        Self {
            slots: self
                .slots
                .iter()
                .map(|s| s.as_ref().map(|v| v.clone_value()))
                .collect(),
        }
    }
}

/// Builder for hand-constructed [`PlanState`]s (see [`PlanState::build`]).
pub struct PlanStateBuilder<'a> {
    registry: &'a ComponentRegistry,
    slots: Vec<Option<Box<dyn ErasedValue>>>,
}

impl PlanStateBuilder<'_> {
    /// Set component `T`'s slot to `value` (a no-op if `T` is not registered).
    #[must_use]
    pub fn set<T: PlanComponent>(mut self, value: T) -> Self {
        if let Some(i) = self.registry.get::<T>() {
            self.slots[i] = Some(Box::new(value));
        }
        self
    }

    /// Materialize every unset slot as `Default` and freeze the scratchpad.
    #[must_use]
    pub fn finish(self) -> PlanState {
        let mut slots = self.slots;
        for (i, slot) in slots.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some((self.registry.defaults[i])());
            }
        }
        PlanState { slots }
    }
}
