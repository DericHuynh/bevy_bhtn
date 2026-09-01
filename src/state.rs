//! The planning scratchpad: a dense byte-pool snapshot of the components an
//! agent's domain touches, plus the registry that maps component types to
//! slot offsets.
//!
//! An entity's state in Bevy is distributed across independent components, so
//! the planner never sees a monolithic struct. Instead, at plan time it
//! **extracts** every registered component from the real entity into a
//! [`PlanState`] (missing components materialize as `Default`), runs the whole
//! forward search on that isolated scratchpad — preconditions read it, effects
//! mutate it, backtracking rolls it back — and never touches the `World`.
//!
//! # Layout
//!
//! The scratchpad is **one contiguous allocation**: each registered component
//! type owns an aligned slot region (offset/size fixed at registration), and
//! all slots are always initialized. The pool itself is allocated with the
//! maximum slot alignment, so every slot's absolute address is aligned. This
//! makes the operations the planner performs thousands of times per search
//! allocation-free:
//!
//! - `clone` — one pool allocation + per-slot monomorphized clones (no
//!   per-component heap boxes, so the look-ahead sweep's lazy clone is cheap),
//! - rollback snapshots/restores — deep clones of single slot values (the
//!   journal owns its copies; plain-data slots clone as a `memcpy`),
//! - `get`/`get_mut` — aligned pointer casts (no `dyn Any` downcasts).
//!
//! Component access is fully monomorphized: a precondition closure
//! `|ammo: &Ammo, vision: &TargetVision| ...` is compiled (via the
//! [`crate::tasks::IntoPrecondition`] impls) into a type-erased checker that
//! captured its components' slot offsets at build time, so per-evaluation cost
//! is a pointer cast plus the closure body — no reflection, no hash lookups.
//!
//! # Safety
//!
//! The pool is raw bytes; all typed access goes through monomorphized fn
//! pointers captured at registration (`fetch`/`write`/`clone`/`drop` for the
//! concrete component type). The invariants that make this sound:
//!
//! 1. a slot's offset/size/alignment are fixed before any [`PlanState`] exists
//!    (the registry is frozen once the domain is baked — `Arc::get_mut`
//!    enforces this),
//! 2. every slot is initialized for the lifetime of the scratchpad (extract
//!    and the builder write every slot; effects mutate in place through
//!    `&mut T`, never reinitialize),
//! 3. whole-value overwrites (fetch/set/restore) drop the old value first,
//! 4. multi-slot mutable access uses distinct registered offsets, so the
//!    `&mut` pointers never alias,
//! 5. the pool allocation carries the maximum slot alignment, so every
//!    typed slot pointer is aligned.

use std::alloc::Layout;
use std::any::TypeId;
use std::ptr::NonNull;
use std::sync::Arc;

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

/// A type-erased per-component fetcher: writes the component's value (or its
/// `Default`) into the scratchpad slot at `slot`. `slot` is uninitialized
/// memory when called from [`PlanState::extract`] — implementations must
/// `write` (not drop-first).
pub type FetchFn = fn(&World, Entity, *mut u8);

/// A type-erased per-component writer: reads the component's value from
/// `slot` and inserts it onto `entity`'s component storage.
pub type WriteFn = fn(&mut World, Entity, *const u8);

/// A type-erased per-component deep clone: `clone(src, dst)` writes a fresh
/// clone of the value at `src` into the **uninitialized** slot at `dst`.
///
/// # Safety
/// `src` must hold an initialized value of the registered type; `dst` must be
/// uninitialized (or dropped-before) memory of at least the slot's size and
/// alignment.
pub type CloneFn = unsafe fn(*const u8, *mut u8);

/// A type-erased per-component drop.
///
/// # Safety
/// `slot` must hold an initialized value of the registered type; the value
/// must not be used afterwards.
pub type DropFn = unsafe fn(*mut u8);

/// Monomorphized fetcher for one component type (value or `Default`).
fn fetch_fn<T: PlanComponent>() -> FetchFn {
    |world, entity, slot| {
        let value = world.get::<T>(entity).cloned().unwrap_or_default();
        unsafe { std::ptr::write(slot as *mut T, value) };
    }
}

/// Monomorphized writer for one component type.
fn write_fn<T: PlanComponent>() -> WriteFn {
    |world, entity, slot| {
        let value = unsafe { &*(slot as *const T) };
        if let Ok(mut e) = world.get_entity_mut(entity) {
            e.insert(value.clone());
        }
    }
}

/// Monomorphized deep clone for one component type.
fn clone_fn<T: PlanComponent>() -> CloneFn {
    |src, dst| unsafe {
        let value = &*(src as *const T);
        std::ptr::write(dst as *mut T, value.clone());
    }
}

/// Monomorphized drop for one component type.
fn drop_fn<T: PlanComponent>() -> DropFn {
    |slot| unsafe { std::ptr::drop_in_place::<T>(slot as *mut T) }
}

/// The frozen slot layout of a baked domain: one entry per registered
/// component type, with its byte offset in the [`PlanState`] pool and its
/// monomorphized access fns.
#[derive(Default)]
pub struct RegistryLayout {
    types: Vec<(TypeId, &'static str)>,
    offsets: Vec<usize>,
    sizes: Vec<usize>,
    aligns: Vec<usize>,
    fetchers: Vec<FetchFn>,
    writers: Vec<WriteFn>,
    cloners: Vec<CloneFn>,
    droppers: Vec<DropFn>,
    total: usize,
    /// Maximum slot alignment — the pool allocation's alignment.
    align: usize,
}

impl RegistryLayout {
    /// Append a slot for `T` (alignment-correct), returning its index.
    fn push_slot<T: PlanComponent>(&mut self, name: &'static str) -> usize {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();
        let offset = self.total.next_multiple_of(align);
        self.types.push((TypeId::of::<T>(), name));
        self.offsets.push(offset);
        self.sizes.push(size);
        self.aligns.push(align);
        self.fetchers.push(fetch_fn::<T>());
        self.writers.push(write_fn::<T>());
        self.cloners.push(clone_fn::<T>());
        self.droppers.push(drop_fn::<T>());
        self.total = offset + size;
        self.align = self.align.max(align);
        self.types.len() - 1
    }
}

impl std::fmt::Debug for RegistryLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryLayout")
            .field("slots", &self.types.len())
            .field("total_bytes", &self.total)
            .field("align", &self.align)
            .finish()
    }
}

/// The registry of component types a domain touches, assigning each a dense
/// slot index with an alignment-correct byte region in the [`PlanState`] pool.
///
/// The registry is **frozen** once the domain is baked: any [`PlanState`] (or
/// clone of one) shares the layout `Arc`, after which further registration is
/// a bug caught by `Arc::get_mut`.
#[derive(Clone, Default)]
pub struct ComponentRegistry {
    layout: Arc<RegistryLayout>,
}

impl ComponentRegistry {
    /// The slot index of component `T`, registering it (with its byte region
    /// and monomorphized access fns) on first use.
    ///
    /// Panics if the registry is frozen (a `PlanState` already shares the
    /// layout) — registration happens only during domain recording.
    pub fn index<T: PlanComponent>(&mut self) -> usize {
        let layout = Arc::get_mut(&mut self.layout)
            .expect("component registry is frozen (a PlanState already exists)");
        let name = std::any::type_name::<T>();
        if let Some(idx) = layout
            .types
            .iter()
            .position(|(tid, _)| *tid == TypeId::of::<T>())
        {
            return idx;
        }
        layout.push_slot::<T>(name)
    }

    /// Whether component `T` is registered.
    pub fn contains<T: 'static>(&self) -> bool {
        self.get::<T>().is_some()
    }

    /// The slot index of component `T`, if registered.
    pub fn get<T: 'static>(&self) -> Option<usize> {
        let tid = TypeId::of::<T>();
        self.layout.types.iter().position(|(t, _)| *t == tid)
    }

    /// The number of registered components (the
    /// [`FieldSet`](crate::summaries::FieldSet) universe size).
    pub fn len(&self) -> usize {
        self.layout.types.len()
    }

    /// Whether no components are registered.
    pub fn is_empty(&self) -> bool {
        self.layout.types.is_empty()
    }

    /// The registered component's short type name, by slot index.
    pub fn name_of(&self, idx: usize) -> &'static str {
        let full = self.layout.types[idx].1;
        full.rsplit("::").next().unwrap_or(full)
    }

    /// The maximum slot alignment (the rollback journal's arena alignment).
    pub(crate) fn max_align(&self) -> usize {
        self.layout.align
    }
}

impl std::fmt::Debug for ComponentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentRegistry")
            .field("slots", &self.layout.types.len())
            .field("total_bytes", &self.layout.total)
            .finish()
    }
}

/// The scratchpad's byte pool: one allocation sized to the layout's total and
/// aligned to its maximum slot alignment.
struct Pool {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl Pool {
    /// Allocate `size` bytes with `align` alignment (zero-sized pools use a
    /// dangling, well-aligned pointer — nothing is ever dereferenced).
    fn new(size: usize, align: usize) -> Self {
        if size == 0 {
            return Self {
                ptr: NonNull::dangling(),
                size: 0,
                layout: Layout::from_size_align(1, align.max(1)).expect("valid layout"),
            };
        }
        let layout = Layout::from_size_align(size, align.max(1)).expect("valid pool layout");
        let ptr = unsafe { std::alloc::alloc(layout) };
        Self {
            ptr: NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout)),
            size,
            layout,
        }
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        if self.size != 0 {
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// The pool uniquely owns its allocation and is only ever accessed through
// `&Pool`/`&mut Pool` (never through shared raw pointers), so it is safe to
// move/share like any owned buffer — same reasoning as `Vec<u8>`.
unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

/// The planning scratchpad: one contiguous byte pool holding every registered
/// component, slot regions laid out by the domain's [`ComponentRegistry`].
///
/// The forward search runs entirely on this snapshot — effects mutate slots,
/// backtracking restores them — so planning never mutates the real `World`.
pub struct PlanState {
    layout: Arc<RegistryLayout>,
    pool: Pool,
}

impl PlanState {
    /// Extract a scratchpad from `entity`: every registered component is
    /// cloned out of the world, or materialized as `Default` when absent.
    pub fn extract(world: &World, entity: Entity, registry: &ComponentRegistry) -> Self {
        let layout = Arc::clone(&registry.layout);
        let mut pool = Pool::new(layout.total, layout.align);
        for (i, fetch) in layout.fetchers.iter().enumerate() {
            // Slots are freshly allocated (uninitialized): `write`, no drop.
            unsafe { fetch(world, entity, pool.as_mut_ptr().add(layout.offsets[i])) };
        }
        Self { layout, pool }
    }

    /// Re-extract every slot from `entity` **in place**: drops each current
    /// value, then fetches the component (or its `Default`) into the same
    /// slot region. No allocation — the driver's hot path reuses one
    /// scratchpad across agents and across its plan/validate/re-extract
    /// phases (same discipline as the look-ahead sweep's `copy_from`).
    ///
    /// # Panics (debug)
    /// If `registry` is not this scratchpad's registry.
    pub fn refresh(&mut self, world: &World, entity: Entity, registry: &ComponentRegistry) {
        debug_assert!(Arc::ptr_eq(&self.layout, &registry.layout));
        for i in 0..self.layout.types.len() {
            let off = self.layout.offsets[i];
            unsafe {
                (self.layout.droppers[i])(self.pool.as_mut_ptr().add(off));
                (self.layout.fetchers[i])(world, entity, self.pool.as_mut_ptr().add(off));
            }
        }
    }

    /// Begin building a scratchpad directly from component values (no `World`
    /// needed). Unset slots materialize as `Default` on
    /// [`finish`](PlanStateBuilder::finish).
    pub fn build(registry: &ComponentRegistry) -> PlanStateBuilder {
        PlanStateBuilder {
            layout: Arc::clone(&registry.layout),
            pool: Pool::new(registry.layout.total, registry.layout.align),
            initialized: vec![false; registry.layout.types.len()],
        }
    }

    /// Write the given slots back onto `entity`'s real components, using the
    /// registry's monomorphized writers. Only the listed slots are committed;
    /// call this after mutating the scratchpad to apply simulated effects to
    /// the world.
    pub fn write_back_with(
        &self,
        world: &mut World,
        entity: Entity,
        registry: &ComponentRegistry,
        indices: &[usize],
    ) {
        debug_assert!(Arc::ptr_eq(&self.layout, &registry.layout));
        for &i in indices {
            unsafe {
                (self.layout.writers[i])(
                    world,
                    entity,
                    self.pool.as_ptr().add(self.layout.offsets[i]),
                );
            }
        }
    }

    /// Read component `T` at slot `idx`.
    ///
    /// Panics if `idx` is not `T`'s registered slot — closures capture their
    /// offsets from the same registry that sized the scratchpad, so this can
    /// only fail on a caller bug.
    pub fn get<T: PlanComponent>(&self, idx: usize) -> &T {
        debug_assert_eq!(self.layout.sizes[idx], std::mem::size_of::<T>());
        unsafe { &*self.slot_ptr::<T>(idx) }
    }

    /// Mutably read component `T` at slot `idx` (same contract as [`Self::get`]).
    pub fn get_mut<T: PlanComponent>(&mut self, idx: usize) -> &mut T {
        debug_assert_eq!(self.layout.sizes[idx], std::mem::size_of::<T>());
        unsafe { &mut *self.slot_ptr::<T>(idx) }
    }

    /// Raw pointers to the given slots, proven disjoint by their distinct
    /// registered offsets. Used by the compiled multi-argument effect
    /// closures.
    ///
    /// # Panics
    /// If any index repeats — two `&mut` into the same slot region would
    /// alias. (Same-type duplicate closure parameters are already rejected
    /// when the effect is compiled; this is defense in depth.)
    ///
    /// # Safety contract (enforced by callers)
    /// All indices must hold the caller's component types; every slot must be
    /// initialized.
    pub(crate) fn disjoint_slots<const N: usize>(&mut self, idxs: [usize; N]) -> [*mut u8; N] {
        // Build-time `assert_distinct_slots` already rejects duplicate
        // parameters; this re-check is debug-only defense in depth.
        debug_assert!(
            (0..N).all(|i| (i + 1..N).all(|j| idxs[i] != idxs[j])),
            "disjoint_slots requires distinct slot indices (slots would alias)"
        );
        let base = self.pool.as_mut_ptr();
        let mut out = [std::ptr::null_mut::<u8>(); N];
        for (out_slot, &idx) in out.iter_mut().zip(idxs.iter()) {
            *out_slot = unsafe { base.add(self.layout.offsets[idx]) };
        }
        // Distinct offsets => distinct memory regions => no aliasing.
        out
    }

    /// The typed pointer to slot `idx`.
    ///
    /// # Safety contract (enforced by callers)
    /// `idx` must be `T`'s registered slot and initialized.
    unsafe fn slot_ptr<T>(&self, idx: usize) -> *mut T {
        self.pool.as_ptr().add(self.layout.offsets[idx]) as *mut T
    }

    /// Snapshot one slot's value for rollback by **deep-cloning it into the
    /// journal** — the journal owns its own copy, so the value's heap
    /// allocations stay valid no matter what in-place mutation does to the
    /// slot afterwards (a bitwise copy would dangle the moment a mutation
    /// freed an internal buffer, and reanimating it on restore is a double
    /// free). For plain-data slots the cloner is a memcpy.
    ///
    /// # Safety contract (enforced by the planner's rollback journal)
    /// `dst` must be aligned for the slot's type and hold `size_of::<T>()`
    /// bytes of uninitialized memory.
    pub(crate) fn snapshot_slot(&self, idx: usize, dst: *mut u8) {
        let off = self.layout.offsets[idx];
        unsafe {
            (self.layout.cloners[idx])(self.pool.as_ptr().add(off), dst);
        }
    }

    /// Restore one slot from a journal snapshot: drops the current value and
    /// **clones** the journal's copy back in. The journal keeps ownership of
    /// its copy until [`Self::drop_journaled_slot`] releases it — the two
    /// values never alias.
    ///
    /// # Safety contract (enforced by the planner's rollback journal)
    /// `bytes` must hold exactly `size_of::<T>()` bytes of a live, initialized
    /// value cloned from this slot.
    pub(crate) unsafe fn restore_slot(&mut self, idx: usize, bytes: *const u8) {
        (self.layout.droppers[idx])(self.pool.as_mut_ptr().add(self.layout.offsets[idx]));
        (self.layout.cloners[idx])(bytes, self.pool.as_mut_ptr().add(self.layout.offsets[idx]));
    }

    /// Drop a journal-held value after it has been cloned back into its slot.
    /// Every allocation then has exactly one owner: the slot.
    ///
    /// # Safety contract (enforced by the planner's rollback journal)
    /// `bytes` must point at a live value cloned from slot `idx` that has
    /// already been restored (or will never be restored).
    pub(crate) unsafe fn drop_journaled_slot(&self, idx: usize, bytes: *mut u8) {
        (self.layout.droppers[idx])(bytes);
    }

    pub(crate) fn slot_size(&self, idx: usize) -> usize {
        self.layout.sizes[idx]
    }

    /// The alignment of slot `idx` (the journal aligns each cloned value).
    pub(crate) fn slot_align(&self, idx: usize) -> usize {
        self.layout.aligns[idx]
    }

    /// Deep-copy `src`'s slots into this scratchpad (same layout): drops each
    /// current value, then clones `src`'s in place. No allocation — used by
    /// the look-ahead sweep to reuse its private clone across sweeps.
    ///
    /// # Panics (debug)
    /// If the two scratchpads do not share the same registry layout.
    pub fn copy_from(&mut self, src: &PlanState) {
        debug_assert!(Arc::ptr_eq(&self.layout, &src.layout));
        for i in 0..self.layout.types.len() {
            let off = self.layout.offsets[i];
            unsafe {
                (self.layout.droppers[i])(self.pool.as_mut_ptr().add(off));
                (self.layout.cloners[i])(
                    src.pool.as_ptr().add(off),
                    self.pool.as_mut_ptr().add(off),
                );
            }
        }
    }

    /// The number of slots (registry size at extraction time).
    pub fn len(&self) -> usize {
        self.layout.types.len()
    }

    /// Whether the scratchpad has no slots.
    pub fn is_empty(&self) -> bool {
        self.layout.types.is_empty()
    }
}

impl Default for PlanState {
    /// An empty scratchpad (no slots) — a placeholder; real scratchpads come
    /// from [`PlanState::extract`] or [`PlanState::build`].
    fn default() -> Self {
        Self {
            layout: Arc::new(RegistryLayout::default()),
            pool: Pool::new(0, 1),
        }
    }
}

impl Drop for PlanState {
    fn drop(&mut self) {
        for i in 0..self.layout.types.len() {
            unsafe {
                (self.layout.droppers[i])(self.pool.as_mut_ptr().add(self.layout.offsets[i]));
            }
        }
    }
}

impl Clone for PlanState {
    fn clone(&self) -> Self {
        let mut pool = Pool::new(self.layout.total, self.layout.align);
        for i in 0..self.layout.types.len() {
            unsafe {
                (self.layout.cloners[i])(
                    self.pool.as_ptr().add(self.layout.offsets[i]),
                    pool.as_mut_ptr().add(self.layout.offsets[i]),
                );
            }
        }
        Self {
            layout: Arc::clone(&self.layout),
            pool,
        }
    }
}

impl std::fmt::Debug for PlanState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanState")
            .field("slots", &self.layout.types.len())
            .field("pool_bytes", &self.pool.size)
            .finish()
    }
}

/// Builder for hand-constructed [`PlanState`]s (see [`PlanState::build`]).
pub struct PlanStateBuilder {
    layout: Arc<RegistryLayout>,
    pool: Pool,
    initialized: Vec<bool>,
}

impl PlanStateBuilder {
    /// Set component `T`'s slot to `value` (a no-op if `T` is not registered).
    #[must_use]
    pub fn set<T: PlanComponent>(mut self, value: T) -> Self {
        if let Some(i) = self
            .layout
            .types
            .iter()
            .position(|(tid, _)| *tid == TypeId::of::<T>())
        {
            unsafe {
                let slot = self.pool.as_mut_ptr().add(self.layout.offsets[i]) as *mut T;
                if self.initialized[i] {
                    std::ptr::drop_in_place(slot);
                }
                std::ptr::write(slot, value);
            }
            self.initialized[i] = true;
        }
        self
    }

    /// Materialize every unset slot as `Default` and freeze the scratchpad.
    #[must_use]
    pub fn finish(mut self) -> PlanState {
        let scratch_world = World::new();
        for (i, init) in self.initialized.iter_mut().enumerate() {
            if !*init {
                unsafe {
                    (self.layout.fetchers[i])(
                        &scratch_world,
                        Entity::PLACEHOLDER,
                        self.pool.as_mut_ptr().add(self.layout.offsets[i]),
                    );
                }
                *init = true;
            }
        }
        // Move the fully-initialized pool into the `PlanState` without
        // running the builder's `Drop` (which would double-drop the slots the
        // `PlanState` now owns). The `initialized` bookkeeping Vec has no
        // drop obligation of its own beyond its allocation, so it is dropped
        // explicitly here.
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped normally, so both reads are moves
        // out of a forgotten wrapper; every slot is initialized above, so the
        // `PlanState` takes over the builder's entire drop obligation.
        let layout = unsafe { std::ptr::read(&this.layout) };
        let pool = unsafe { std::ptr::read(&this.pool) };
        let initialized = unsafe { std::ptr::read(&this.initialized) };
        drop(initialized);
        PlanState { layout, pool }
    }
}

impl Drop for PlanStateBuilder {
    fn drop(&mut self) {
        // A builder abandoned before `finish` still owns the values that were
        // `set` (and only those — unset slots are raw bytes, never written).
        for (i, init) in self.initialized.iter().enumerate() {
            if *init {
                unsafe {
                    (self.layout.droppers[i])(self.pool.as_mut_ptr().add(self.layout.offsets[i]));
                }
            }
        }
    }
}
