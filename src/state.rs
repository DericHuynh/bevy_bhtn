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
//!    (registration lives on [`RegistryBuilder`] during domain recording; the
//!    baked domain holds the frozen [`ComponentRegistry`], which has no
//!    mutating API — the phase split is enforced by the type system),
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
use std::collections::HashMap;
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

/// A type-erased per-component default writer: writes `T::default()` into the
/// (uninitialized) slot.
///
/// # Safety
/// `slot` must be uninitialized memory of at least the slot's size and
/// alignment.
pub type DefaultFn = fn(*mut u8);

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

/// Monomorphized default writer for one component type.
fn default_fn<T: PlanComponent>() -> DefaultFn {
    |slot| unsafe { std::ptr::write(slot as *mut T, T::default()) }
}

/// One registered component slot: its identity, byte region in the
/// [`PlanState`] pool, and its monomorphized access fns. One contiguous
/// `Vec<Slot>` (instead of parallel per-kind arrays) keeps every per-slot
/// loop — extract/refresh/clone/`copy_from`/drop — on a single cache-friendly
/// array.
pub(crate) struct Slot {
    name: &'static str,
    pub(crate) offset: usize,
    pub(crate) size: usize,
    pub(crate) align: usize,
    pub(crate) fetch_fn: FetchFn,
    pub(crate) write_fn: WriteFn,
    pub(crate) clone_fn: CloneFn,
    pub(crate) drop_fn: DropFn,
    pub(crate) default_fn: DefaultFn,
}

/// The frozen slot layout of a baked domain: one `Slot` per registered
/// component type, plus the pool's total size and maximum alignment.
#[derive(Default)]
pub struct RegistryLayout {
    slots: Vec<Slot>,
    total: usize,
    /// Maximum slot alignment — the pool allocation's alignment.
    align: usize,
    /// `TypeId -> slot index`, carried from the builder so the frozen
    /// registry resolves components in O(1) (the slot table alone would
    /// force a linear scan per lookup).
    by_type: HashMap<TypeId, usize>,
}

impl RegistryLayout {
    /// The slot index registered for `TypeId`, if any (O(1)).
    fn slot_of(&self, tid: &TypeId) -> Option<usize> {
        self.by_type.get(tid).copied()
    }

    /// Append a slot for `T` (alignment-correct), returning its index.
    fn push_slot<T: PlanComponent>(&mut self, name: &'static str) -> usize {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();
        let offset = self.total.next_multiple_of(align);
        self.slots.push(Slot {
            name,
            offset,
            size,
            align,
            fetch_fn: fetch_fn::<T>(),
            write_fn: write_fn::<T>(),
            clone_fn: clone_fn::<T>(),
            drop_fn: drop_fn::<T>(),
            default_fn: default_fn::<T>(),
        });
        self.total = offset + size;
        self.align = self.align.max(align);
        self.by_type.insert(TypeId::of::<T>(), self.slots.len() - 1);
        self.slots.len() - 1
    }
}

impl std::fmt::Debug for RegistryLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryLayout")
            .field("slots", &self.slots.len())
            .field("total_bytes", &self.total)
            .field("align", &self.align)
            .finish()
    }
}

/// The recording-phase registry: accumulates the component types a domain
/// touches while its task functions are recorded, assigning each a dense slot
/// index with an alignment-correct byte region in the [`PlanState`] pool.
///
/// Registration is only possible here. The baked domain holds the frozen
/// [`ComponentRegistry`], which has no mutating API — so "register after a
/// `PlanState` exists" is a compile error, not a runtime panic.
#[derive(Default)]
pub struct RegistryBuilder {
    layout: RegistryLayout,
    /// Collected domain-authoring errors (e.g. an effect closure taking the
    /// same component type twice). Recording soft-collects instead of
    /// panicking; `HtnDomain::build` drains these into its `Builder` error.
    errors: Vec<String>,
}

impl RegistryBuilder {
    /// The slot index of component `T`, registering it (with its byte region
    /// and monomorphized access fns) on first use. The layout owns the
    /// `TypeId -> slot` map for both the recording and frozen phases.
    pub fn index<T: PlanComponent>(&mut self) -> usize {
        if let Some(idx) = self.layout.slot_of(&TypeId::of::<T>()) {
            return idx;
        }
        self.layout.push_slot::<T>(std::any::type_name::<T>())
    }

    /// Soft-collect a domain-authoring error during recording. Baking turns
    /// every collected error into a single `HtnError::Builder`, so one
    /// `build()` call reports all authoring bugs at once.
    pub(crate) fn push_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
    }

    /// Drain the collected authoring errors (used by domain baking).
    pub(crate) fn take_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.errors)
    }

    /// Freeze the registry: the layout becomes immutably shared with every
    /// [`PlanState`] built from it. Domain baking calls this; it is also the
    /// public path to a hand-constructed registry for [`PlanState::build`].
    pub fn freeze(self) -> ComponentRegistry {
        ComponentRegistry {
            layout: Arc::new(self.layout),
        }
    }

    /// The number of registered components.
    pub fn len(&self) -> usize {
        self.layout.slots.len()
    }

    /// Whether no components are registered.
    pub fn is_empty(&self) -> bool {
        self.layout.slots.is_empty()
    }
}

/// The frozen registry of a baked domain: every component type the domain's
/// preconditions/effects touch, with dense slot indices into the [`PlanState`]
/// pool. Read-only by construction — registration happens on
/// [`RegistryBuilder`] during recording, and [`RegistryBuilder::freeze`]
/// produces this type.
pub struct ComponentRegistry {
    layout: Arc<RegistryLayout>,
}

impl ComponentRegistry {
    /// Whether component `T` is registered.
    pub fn contains<T: 'static>(&self) -> bool {
        self.slot_of::<T>().is_some()
    }

    /// The slot index of component `T`, if registered (O(1) via the frozen
    /// `TypeId` map). The planning-side counterpart of
    /// [`RegistryBuilder::index`](RegistryBuilder::index).
    pub fn slot_of<T: 'static>(&self) -> Option<usize> {
        self.layout.by_type.get(&TypeId::of::<T>()).copied()
    }

    /// The number of registered components (the [`FieldSet`] universe size).
    pub fn len(&self) -> usize {
        self.layout.slots.len()
    }

    /// Whether no components are registered.
    pub fn is_empty(&self) -> bool {
        self.layout.slots.is_empty()
    }

    /// The registered component's short type name, by slot index.
    pub fn name_of(&self, idx: usize) -> &'static str {
        let full = self.layout.slots[idx].name;
        full.rsplit("::").next().unwrap_or(full)
    }

    /// The maximum slot alignment (the rollback journal's arena alignment).
    pub(crate) fn max_align(&self) -> usize {
        self.layout.align
    }

    /// The per-slot table (the planner's rollback journal releases unrestored
    /// copies through each slot's dropper on drop).
    pub(crate) fn slots(&self) -> &[Slot] {
        &self.layout.slots
    }
}

/// A compact bitset over the domain's component-slot indices (the registry's
/// dense slot table above).
///
/// All operations assume both sets share the same universe (the same domain's
/// component table). Slot indices are dense and domains touch only a handful
/// of components, so a `Vec<u64>` bitset keeps set operations to a few word
/// ops in the planner's hot path. This is the vocabulary of the bake-time
/// analysis ([`TaskSummary`](crate::summaries::TaskSummary) read/write sets,
/// the look-ahead sweep's "unknown components" overlay, the backward
/// planner's coverage scoring) — it lives here because the universe it
/// indexes is the registry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldSet {
    bits: Vec<u64>,
}

impl FieldSet {
    /// An empty set over a universe of `universe` slots.
    pub fn new(universe: usize) -> Self {
        let words = universe.div_ceil(64);
        Self {
            bits: vec![0; words],
        }
    }

    /// Add slot `idx` to the set.
    pub fn insert(&mut self, idx: usize) {
        let (word, bit) = (idx / 64, idx % 64);
        if word < self.bits.len() {
            self.bits[word] |= 1 << bit;
        }
    }

    /// Whether slot `idx` is in the set.
    pub fn contains(&self, idx: usize) -> bool {
        let (word, bit) = (idx / 64, idx % 64);
        word < self.bits.len() && self.bits[word] & (1 << bit) != 0
    }

    /// Remove slot `idx` from the set.
    pub fn remove(&mut self, idx: usize) {
        let (word, bit) = (idx / 64, idx % 64);
        if word < self.bits.len() {
            self.bits[word] &= !(1 << bit);
        }
    }

    /// Remove every slot from the set.
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// Add every slot of `other` to this set.
    pub fn union_with(&mut self, other: &Self) {
        for (w, o) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w |= o;
        }
    }

    /// Keep only slots present in both sets.
    pub fn intersect_with(&mut self, other: &Self) {
        for (w, o) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w &= o;
        }
    }

    /// Remove every slot of `other` from this set.
    pub fn subtract(&mut self, other: &Self) {
        for (w, o) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w &= !o;
        }
    }

    /// Whether every slot of this set is also in `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(a, b)| a & !b == 0)
    }

    /// Whether the set contains no slots.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|w| *w == 0)
    }

    /// The number of slots in the set.
    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate the slot indices in the set, in ascending order.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(w, bits)| {
            (0..64)
                .filter(move |bit| bits & (1 << bit) != 0)
                .map(move |bit| w * 64 + bit)
        })
    }
}

impl std::fmt::Debug for ComponentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentRegistry")
            .field("slots", &self.layout.slots.len())
            .field("total_bytes", &self.layout.total)
            .finish()
    }
}

impl std::fmt::Debug for RegistryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryBuilder")
            .field("slots", &self.layout.slots.len())
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
        for slot in &layout.slots {
            // Slots are freshly allocated (uninitialized): `write`, no drop.
            unsafe { (slot.fetch_fn)(world, entity, pool.as_mut_ptr().add(slot.offset)) };
        }
        Self { layout, pool }
    }

    /// Re-extract every slot from `entity` **in place**: drops each current
    /// value, then fetches the component (or its `Default`) into the same
    /// slot region. No allocation — the driver's hot path reuses one
    /// scratchpad across agents and across its plan/validate/re-extract
    /// phases (same discipline as the look-ahead sweep's `copy_from`).
    pub fn refresh(&mut self, world: &World, entity: Entity) {
        for slot in &self.layout.slots {
            unsafe {
                (slot.drop_fn)(self.pool.as_mut_ptr().add(slot.offset));
                (slot.fetch_fn)(world, entity, self.pool.as_mut_ptr().add(slot.offset));
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
            initialized: vec![false; registry.layout.slots.len()],
        }
    }

    /// Write the given slots back onto `entity`'s real components, using the
    /// registry's monomorphized writers. Only the listed slots are committed;
    /// call this after mutating the scratchpad to apply simulated effects to
    /// the world.
    pub fn write_back_with(&self, world: &mut World, entity: Entity, indices: &[usize]) {
        for &i in indices {
            let slot = &self.layout.slots[i];
            unsafe {
                (slot.write_fn)(world, entity, self.pool.as_ptr().add(slot.offset));
            }
        }
    }

    /// Read component `T` at raw slot `idx` — the planner's hot path, where
    /// the slot index was captured at closure-build time. The typed,
    /// lookup-by-name counterpart is [`Self::get`].
    ///
    /// # Panics
    /// If `idx` is out of bounds. Calling it with a slot belonging to
    /// another component type is a caller bug with undefined behavior (the
    /// pointer is cast to `T` unchecked) — the same contract the compiled
    /// closures rely on.
    pub fn get_slot<T: PlanComponent>(&self, idx: usize) -> &T {
        debug_assert_eq!(self.layout.slots[idx].size, std::mem::size_of::<T>());
        unsafe { &*self.slot_ptr::<T>(idx) }
    }

    /// Mutably read component `T` at raw slot `idx` (see [`Self::get_slot`]).
    pub fn get_mut_slot<T: PlanComponent>(&mut self, idx: usize) -> &mut T {
        debug_assert_eq!(self.layout.slots[idx].size, std::mem::size_of::<T>());
        unsafe { &mut *self.slot_ptr::<T>(idx) }
    }

    /// Read component `T` by type, resolving its slot through the frozen
    /// registry's `TypeId` map. `None` if `T` was never registered — the
    /// typed counterpart to the raw slot-index [`Self::get_slot`] the hot
    /// loop uses.
    pub fn get<T: PlanComponent>(&self) -> Option<&T> {
        let idx = self.layout.by_type.get(&TypeId::of::<T>()).copied()?;
        Some(self.get_slot::<T>(idx))
    }

    /// Mutably read component `T` by type (see [`Self::get`]).
    pub fn get_mut<T: PlanComponent>(&mut self) -> Option<&mut T> {
        let idx = self.layout.by_type.get(&TypeId::of::<T>()).copied()?;
        Some(self.get_mut_slot::<T>(idx))
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
            *out_slot = unsafe { base.add(self.layout.slots[idx].offset) };
        }
        // Distinct offsets => distinct memory regions => no aliasing.
        out
    }

    /// The typed pointer to slot `idx`.
    ///
    /// # Safety contract (enforced by callers)
    /// `idx` must be `T`'s registered slot and initialized.
    unsafe fn slot_ptr<T>(&self, idx: usize) -> *mut T {
        self.pool.as_ptr().add(self.layout.slots[idx].offset) as *mut T
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
        let slot = &self.layout.slots[idx];
        unsafe {
            (slot.clone_fn)(self.pool.as_ptr().add(slot.offset), dst);
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
        let slot = &self.layout.slots[idx];
        let ptr = self.pool.as_mut_ptr().add(slot.offset);
        (slot.drop_fn)(ptr);
        (slot.clone_fn)(bytes, ptr);
    }

    /// Drop a journal-held value after it has been cloned back into its slot.
    /// Every allocation then has exactly one owner: the slot.
    ///
    /// # Safety contract (enforced by the planner's rollback journal)
    /// `bytes` must point at a live value cloned from slot `idx` that has
    /// already been restored (or will never be restored).
    pub(crate) unsafe fn drop_journaled_slot(&self, idx: usize, bytes: *mut u8) {
        (self.layout.slots[idx].drop_fn)(bytes);
    }

    pub(crate) fn slot_size(&self, idx: usize) -> usize {
        self.layout.slots[idx].size
    }

    /// The alignment of slot `idx` (the journal aligns each cloned value).
    pub(crate) fn slot_align(&self, idx: usize) -> usize {
        self.layout.slots[idx].align
    }

    /// Deep-copy `src`'s slots into this scratchpad (same layout): drops each
    /// current value, then clones `src`'s in place. No allocation — used by
    /// the look-ahead sweep to reuse its private clone across sweeps.
    ///
    /// # Panics (debug)
    /// If the two scratchpads do not share the same registry layout.
    pub fn copy_from(&mut self, src: &PlanState) {
        debug_assert!(Arc::ptr_eq(&self.layout, &src.layout));
        for slot in &self.layout.slots {
            unsafe {
                (slot.drop_fn)(self.pool.as_mut_ptr().add(slot.offset));
                (slot.clone_fn)(
                    src.pool.as_ptr().add(slot.offset),
                    self.pool.as_mut_ptr().add(slot.offset),
                );
            }
        }
    }

    /// Whether the given slots hold identical bytes in both scratchpads (same
    /// registry layout). Used by the driver's quiet-prefix elision to prove a
    /// step's effects are already reflected in the world.
    ///
    /// Byte comparison is sound *as a change detector* for this purpose:
    /// plain-data components carry their whole value in the slot (a semantic
    /// change always changes bytes; equal bytes mean the planner-certified
    /// value is already in effect), while heap-owning components (Vec/HashMap
    /// members) realloc on every write — a semantically identical write still
    /// moves the pointer, so those slots conservatively read as "changed" and
    /// their steps simply never elide. Alignment padding between slots is
    /// never read — it is uninitialized allocation.
    ///
    /// # Panics (debug)
    /// If the two scratchpads do not share the same registry layout.
    pub(crate) fn slots_unchanged(&self, base: &Self, slots: &[usize]) -> bool {
        debug_assert!(Arc::ptr_eq(&self.layout, &base.layout));
        slots.iter().all(|&i| {
            let slot = &self.layout.slots[i];
            unsafe {
                let mine =
                    std::slice::from_raw_parts(self.pool.as_ptr().add(slot.offset), slot.size);
                let theirs =
                    std::slice::from_raw_parts(base.pool.as_ptr().add(slot.offset), slot.size);
                mine == theirs
            }
        })
    }

    /// The number of slots (registry size at extraction time).
    pub fn len(&self) -> usize {
        self.layout.slots.len()
    }

    /// Whether the scratchpad has no slots.
    pub fn is_empty(&self) -> bool {
        self.layout.slots.is_empty()
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
        for slot in &self.layout.slots {
            unsafe { (slot.drop_fn)(self.pool.as_mut_ptr().add(slot.offset)) };
        }
    }
}

impl Clone for PlanState {
    fn clone(&self) -> Self {
        let mut pool = Pool::new(self.layout.total, self.layout.align);
        for slot in &self.layout.slots {
            unsafe {
                (slot.clone_fn)(
                    self.pool.as_ptr().add(slot.offset),
                    pool.as_mut_ptr().add(slot.offset),
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
            .field("slots", &self.layout.slots.len())
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
    /// Set component `T`'s slot to `value`.
    ///
    /// # Panics (debug builds)
    /// If `T` was never registered — a silently ignored `set` would leave
    /// the slot at `Default` and hide the authoring bug behind a passing
    /// test. Release builds keep the historical no-op.
    #[must_use]
    pub fn set<T: PlanComponent>(mut self, value: T) -> Self {
        debug_assert!(
            self.layout.by_type.contains_key(&TypeId::of::<T>()),
            "PlanStateBuilder::set: `{}` was never registered in this registry",
            std::any::type_name::<T>()
        );
        if let Some(i) = self.layout.by_type.get(&TypeId::of::<T>()).copied() {
            unsafe {
                let slot = self.pool.as_mut_ptr().add(self.layout.slots[i].offset) as *mut T;
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
        for (i, init) in self.initialized.iter_mut().enumerate() {
            if !*init {
                let slot = &self.layout.slots[i];
                unsafe { (slot.default_fn)(self.pool.as_mut_ptr().add(slot.offset)) };
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
                let slot = &self.layout.slots[i];
                unsafe { (slot.drop_fn)(self.pool.as_mut_ptr().add(slot.offset)) };
            }
        }
    }
}
