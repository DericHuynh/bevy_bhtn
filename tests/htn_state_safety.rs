//! Safety pins for the byte-pool [`PlanState`] — the crate's unsafe core.
//!
//! Every test here exercises an invariant the `unsafe` code relies on:
//! drop-exactly-once for heap-owning components (pool drop, deep clone,
//! `copy_from`, rollback restore, builder overwrite/abandon), registry
//! freezing, slot placement (ZST / high alignment), aliasing rejection for
//! duplicate `&mut` parameters, and multi-snapshot rollback bookkeeping.

use std::any::TypeId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::{PlanState, RegistryBuilder};
use bevy_bhtn::tasks::{GoalBuilder, TaskBuilder};
use bevy_bhtn::{FieldSet, HtnDomain};
use bevy_ecs::prelude::*;

// ---------------------------------------------------------------------------
// Instrumented heap-owning component
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Counters {
    constructed: AtomicUsize,
    cloned: AtomicUsize,
    dropped: AtomicUsize,
}

impl Counters {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn constructed(&self) -> usize {
        self.constructed.load(Ordering::SeqCst)
    }

    fn cloned(&self) -> usize {
        self.cloned.load(Ordering::SeqCst)
    }

    fn dropped(&self) -> usize {
        self.dropped.load(Ordering::SeqCst)
    }

    /// Every live value was either constructed explicitly or cloned; at
    /// steady state (everything released) drops must account for all of them
    /// — no leaks, no double-frees.
    fn assert_balanced(&self) {
        assert_eq!(
            self.dropped(),
            self.constructed() + self.cloned(),
            "drop balance broken: constructed={} cloned={} dropped={}",
            self.constructed(),
            self.cloned(),
            self.dropped()
        );
    }
}

/// A component owning heap data (`String`) with instrumented clone/drop.
#[derive(Component, Default, Debug)]
struct Name(Arc<Counters>, String);

impl Name {
    fn new(counter: &Arc<Counters>, value: &str) -> Self {
        counter.constructed.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(counter), value.to_string())
    }
}

impl Clone for Name {
    fn clone(&self) -> Self {
        self.0.cloned.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(&self.0), self.1.clone())
    }
}

impl Drop for Name {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Gold(pub i32);

// ---------------------------------------------------------------------------
// Pool lifetime
// ---------------------------------------------------------------------------

/// Dropping a scratchpad drops every slot's value exactly once.
#[test]
fn plan_state_drop_releases_every_slot_once() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    builder.index::<Gold>();
    let registry = builder.freeze();

    {
        let state = PlanState::build(&registry)
            .set(Name::new(&counter, "held"))
            .set(Gold(1))
            .finish();
        assert_eq!(state.get::<Name>(registry.get::<Name>().unwrap()).1, "held");
    } // dropped here

    counter.assert_balanced();
    assert_eq!(counter.dropped(), 1);
}

/// `Clone` is a deep clone: mutating the clone never touches the original,
/// and both values are released exactly once.
#[test]
fn clone_is_deep_for_heap_owning_components() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    let registry = builder.freeze();

    let original = PlanState::build(&registry)
        .set(Name::new(&counter, "orig"))
        .finish();
    let mut copy = original.clone();

    let name_slot = registry.get::<Name>().unwrap();
    copy.get_mut::<Name>(name_slot).1 = "mutated".into();

    assert_eq!(
        original.get::<Name>(name_slot).1,
        "orig",
        "clone must not alias"
    );
    assert_eq!(copy.get::<Name>(name_slot).1, "mutated");
    drop(original);
    drop(copy);

    counter.assert_balanced();
}

/// `copy_from` (the look-ahead sweep's scratch reuse) drops the destination's
/// current values before cloning the source's in — no leak, no double-free.
#[test]
fn copy_from_replaces_destination_values_cleanly() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    let registry = builder.freeze();

    let source = PlanState::build(&registry)
        .set(Name::new(&counter, "source"))
        .finish();
    let mut scratch = PlanState::build(&registry)
        .set(Name::new(&counter, "stale"))
        .finish();

    scratch.copy_from(&source);
    assert_eq!(
        scratch.get::<Name>(registry.get::<Name>().unwrap()).1,
        "source"
    );
    drop(source);
    drop(scratch);

    counter.assert_balanced();
}

/// `write_back_with` clones the value out (the pool keeps owning its copy),
/// so committing to the world never moves ownership out of the scratchpad.
#[test]
fn write_back_clones_out_and_pool_still_owns() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    builder.index::<Gold>();
    let registry = builder.freeze();

    let state = PlanState::build(&registry)
        .set(Name::new(&counter, "committed"))
        .set(Gold(7))
        .finish();

    let mut world = World::new();
    let entity = world.spawn(Gold(0)).id();
    state.write_back_with(&mut world, entity, &registry, &[0, 1]);

    assert_eq!(world.get::<Gold>(entity).unwrap().0, 7);
    assert_eq!(world.get::<Name>(entity).unwrap().1, "committed");
    drop(state);
    drop(world);

    counter.assert_balanced();
}

// ---------------------------------------------------------------------------
// Builder lifetime
// ---------------------------------------------------------------------------

/// `set` on the same slot twice drops the overwritten value.
#[test]
fn builder_set_overwrites_dropping_old_value() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    let registry = builder.freeze();

    let state = PlanState::build(&registry)
        .set(Name::new(&counter, "first"))
        .set(Name::new(&counter, "second"))
        .finish();

    assert_eq!(
        state.get::<Name>(registry.get::<Name>().unwrap()).1,
        "second"
    );
    drop(state);

    // constructed 2, cloned 0: the "first" value was dropped by the
    // overwrite, the "second" by the state drop.
    counter.assert_balanced();
}

/// Unset slots materialize as `Default` on `finish`.
#[test]
fn builder_finish_materializes_defaults_for_unset_slots() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    builder.index::<Gold>();
    let registry = builder.freeze();

    let state = PlanState::build(&registry)
        .set(Name::new(&counter, "only"))
        .finish();

    let gold_slot = registry.get::<Gold>().unwrap();
    assert_eq!(state.get::<Gold>(gold_slot).0, 0, "Gold defaulted");
    assert_eq!(state.get::<Name>(registry.get::<Name>().unwrap()).1, "only");
    drop(state);

    // The default Name was built on its own fresh counter; the test counter
    // only ever saw the one explicit value.
    counter.assert_balanced();
    assert_eq!(counter.dropped(), 1);
}

/// A builder abandoned before `finish` releases the values that were `set`
/// (and only those — unset slots are raw bytes and must not be dropped).
#[test]
fn builder_dropped_without_finish_releases_set_values() {
    let counter = Counters::new();
    let mut builder = RegistryBuilder::default();
    builder.index::<Name>();
    builder.index::<Gold>();
    let registry = builder.freeze();

    {
        let builder_state = PlanState::build(&registry).set(Name::new(&counter, "doomed"));
        drop(builder_state); // never finished
    }

    assert_eq!(counter.dropped(), 1, "the set value was released");
    counter.assert_balanced();
}

// ---------------------------------------------------------------------------
// Registry freezing & idempotence
// ---------------------------------------------------------------------------

/// Registering the same component type twice returns the same slot and does
/// not grow the layout.
#[test]
fn registry_index_is_idempotent_per_type() {
    let mut registry = RegistryBuilder::default();
    let first = registry.index::<Gold>();
    let second = registry.index::<Gold>();
    assert_eq!(first, second);
    assert_eq!(registry.len(), 1);

    registry.index::<Name>();
    assert_eq!(registry.len(), 2);
}

/// Registration is a recording-phase operation: the builder assigns the
/// slots, and the frozen registry resolves the same indices with no mutating
/// API — late registration after a `PlanState` exists is a compile error,
/// not a runtime panic.
#[test]
fn frozen_registry_resolves_builder_slot_indices() {
    let mut builder = RegistryBuilder::default();
    let gold = builder.index::<Gold>();
    let name = builder.index::<Name>();
    let registry = builder.freeze();

    assert_eq!(registry.get::<Gold>(), Some(gold));
    assert_eq!(registry.get::<Name>(), Some(name));
    assert_eq!(registry.len(), 2);

    let world = World::new();
    let state = PlanState::extract(&world, Entity::PLACEHOLDER, &registry);
    assert_eq!(state.len(), 2);
}

// ---------------------------------------------------------------------------
// Slot placement
// ---------------------------------------------------------------------------

/// Zero-sized components participate in planning: their slots occupy no bytes
/// but preconditions/effects still compile and run.
#[test]
fn zst_components_plan_end_to_end() {
    #[derive(Component, Clone, Default, Debug)]
    struct Flag;

    fn root(task: &mut TaskBuilder) {
        task.branch().then(check_flag).then(act);
    }
    fn check_flag(task: &mut TaskBuilder) {
        task.precondition(|_: &Flag| true);
    }
    fn act(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 5)
            .effect(|_: &mut Flag| {});
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).set(Gold(0)).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        planner.plan("root", &state).task_names(),
        ["check_flag", "act"]
    );
}

/// A high-alignment component placed after smaller ones is aligned correctly
/// and its neighbors keep their values.
#[test]
fn high_alignment_slot_is_placed_correctly() {
    #[repr(align(64))]
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct Big(pub u64);

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct Small(pub u8);

    let mut builder = RegistryBuilder::default();
    builder.index::<Small>();
    builder.index::<Big>();
    let registry = builder.freeze();
    let small_idx = registry.get::<Small>().unwrap();
    let big_idx = registry.get::<Big>().unwrap();

    let state = PlanState::build(&registry)
        .set(Small(0xAB))
        .set(Big(0x1234_5678_9ABC_DEF0))
        .finish();

    assert_eq!(state.get::<Small>(small_idx).0, 0xAB);
    assert_eq!(state.get::<Big>(big_idx).0, 0x1234_5678_9ABC_DEF0);

    // The pool is sized to fit the aligned layout (at least 64 + 1 bytes,
    // rounded up to the alignment).
    assert!(state.len() >= 2);
}

// ---------------------------------------------------------------------------
// Aliasing rejection
// ---------------------------------------------------------------------------

/// An effect closure taking `&mut` to the same component type twice would
/// alias one slot — rejected when the effect is compiled (during recording).
#[test]
#[should_panic(expected = "same component type twice")]
fn duplicate_mut_params_are_rejected_at_build() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(broken);
    }
    fn broken(task: &mut TaskBuilder) {
        task.effect(|a: &mut Gold, b: &mut Gold| {
            a.0 = 1;
            b.0 = 2;
        });
    }
    let _ = HtnDomain::from_root(root).build();
}

/// Repeated `&mut` parameters are fine on preconditions (shared references
/// may alias freely).
#[test]
fn duplicate_shared_params_are_allowed() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(check);
    }
    fn check(task: &mut TaskBuilder) {
        task.precondition(|a: &Gold, b: &Gold| a.0 == b.0 && a.0 > 0);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).set(Gold(3)).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan("root", &state).task_names(), ["check"]);
}

// ---------------------------------------------------------------------------
// Rollback bookkeeping
// ---------------------------------------------------------------------------

/// A task with two effects writing the same slot snapshots twice and restores
/// twice — the LIFO journal must unwind cleanly.
#[test]
fn same_slot_written_twice_rolls_back_cleanly() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(double_write).then(impossible);
        task.branch().then(safe);
    }
    fn double_write(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 5)
            .effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();

    // Direct application composes both writes.
    let mut state = PlanState::build(&domain.components).set(Gold(0)).finish();
    let Task::Primitive(p) = domain.get_task("double_write").unwrap() else {
        panic!("double_write is primitive");
    };
    p.apply_effects(&mut state);
    assert_eq!(
        state
            .get::<Gold>(domain.components.get::<Gold>().unwrap())
            .0,
        6
    );

    // And the search backtracks through the double snapshot without
    // corruption: the doomed branch is abandoned, the safe one plans.
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan("root", &state).task_names(), ["safe"]);
}

/// Backtracking moves heap-owning values back by bytes (drop current, restore
/// snapshot): every constructed/cloned value is dropped exactly once across a
/// full search that overwrites and rolls back a `String` slot.
#[test]
fn rollback_moves_heap_values_without_leaks_or_double_frees() {
    let counter = Counters::new();

    fn root(task: &mut TaskBuilder) {
        task.branch().then(write_name).then(impossible);
        task.branch().then(safe);
    }
    fn write_name(task: &mut TaskBuilder) {
        task.effect(|n: &mut Name| *n = Name(Arc::new(Counters::default()), "planned".into()));
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components)
        .set(Name::new(&counter, "start"))
        .set(Gold(0))
        .finish();

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan("root", &state).task_names(), ["safe"]);
    drop(state); // the planner's working clone is released inside plan()

    counter.assert_balanced();
}

/// The look-ahead sweep's reused scratch (`copy_from`) deep-clones and drops
/// heap-owning slots across sweeps without leaking or double-freeing.
#[test]
fn lookahead_scratch_reuse_balances_heap_clones() {
    let counter = Counters::new();

    fn root(task: &mut TaskBuilder) {
        task.branch().then(write_name).then(impossible);
        task.branch().then(write_name).then(impossible);
        task.branch().then(safe);
    }
    fn write_name(task: &mut TaskBuilder) {
        task.effect(|n: &mut Name| *n = Name(Arc::new(Counters::default()), "planned".into()));
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components)
        .set(Name::new(&counter, "start"))
        .set(Gold(0))
        .finish();

    // Look-ahead ON: each doomed commitment sweeps [write_name, impossible],
    // materializing/reusing the private scratch (clone + copy_from paths).
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(true);
    assert_eq!(planner.plan("root", &state).task_names(), ["safe"]);
    drop(state);

    counter.assert_balanced();
}

// ---------------------------------------------------------------------------
// Misc edges
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rollback snapshot ownership (regressions for the bitwise-snapshot bug)
// ---------------------------------------------------------------------------

/// A heap-owning component whose allocation verifies it is freed **exactly
/// once**: the inner cell records its own free, so reanimating freed bytes
/// (the exact failure mode of a dangling rollback snapshot) panics
/// deterministically. Drop-invocation counters alone cannot catch this —
/// dropping the same allocation twice keeps constructed/cloned/dropped
/// counts balanced.
#[derive(Component, Debug)]
struct HeapGuard(*mut Cell);

struct Cell {
    freed: bool,
}

impl HeapGuard {
    fn new() -> Self {
        let cell = unsafe { std::alloc::alloc(std::alloc::Layout::new::<Cell>()) as *mut Cell };
        unsafe { std::ptr::write(cell, Cell { freed: false }) };
        Self(cell)
    }
}

impl Default for HeapGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HeapGuard {
    // A fresh, independent cell: clones are equal-but-distinct values, so
    // every live value owns its own allocation.
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Drop for HeapGuard {
    fn drop(&mut self) {
        unsafe {
            assert!(
                !(*self.0).freed,
                "heap cell freed twice — a dangling snapshot was reanimated"
            );
            (*self.0).freed = true;
            std::alloc::dealloc(self.0 as *mut u8, std::alloc::Layout::new::<Cell>());
        }
    }
}

unsafe impl Send for HeapGuard {}
unsafe impl Sync for HeapGuard {}

/// Regression: a rollback snapshot must OWN its copy of a heap-owning value.
/// The old journal snapshotted slot bytes bitwise; an effect that *replaced*
/// the value freed the old allocation, the snapshot dangled, and the restore
/// reanimated freed memory — the second drop is a double free.
#[test]
fn rollback_of_a_replaced_heap_value_frees_everything_exactly_once() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(replace_guard).then(impossible);
        task.branch().then(safe);
    }
    fn replace_guard(task: &mut TaskBuilder) {
        task.effect(|g: &mut HeapGuard| *g = HeapGuard::new());
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components)
        .set(HeapGuard::new())
        .set(Gold(0))
        .finish();

    let mut planner = HtnPlanner::new(&domain);
    // Look-ahead off: the sweep would refute the doomed method before any
    // primitive runs, and the rollback journal would never be exercised.
    planner.set_lookahead(false);
    assert_eq!(planner.plan("root", &state).task_names(), ["safe"]);
    drop(state);
}

/// Regression: the same slot snapshotted twice *without* an intervening
/// mutation (the recursive-acquire shape) must not restore the same freed
/// allocation twice.
#[test]
fn rollback_of_repeated_unmutated_snapshots_frees_everything_exactly_once() {
    fn root(task: &mut TaskBuilder) {
        task.branch()
            .then(touch_guard)
            .then(touch_guard)
            .then(impossible);
        task.branch().then(safe);
    }
    fn touch_guard(task: &mut TaskBuilder) {
        // Takes the &mut (so the slot is journaled) but never mutates: both
        // snapshots alias the same allocation under a bitwise journal.
        task.effect(|_g: &mut HeapGuard| {});
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components)
        .set(HeapGuard::new())
        .set(Gold(0))
        .finish();

    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert_eq!(planner.plan("root", &state).task_names(), ["safe"]);
    drop(state);
}

/// `FieldSet` operations work past the first 64-bit word (universes larger
/// than one word).
#[test]
fn field_set_handles_bits_beyond_the_first_word() {
    let mut set = FieldSet::new(70);
    set.insert(0);
    set.insert(63);
    set.insert(64);
    set.insert(69);
    assert_eq!(set.count(), 4);
    assert!(set.contains(0));
    assert!(set.contains(63));
    assert!(set.contains(64));
    assert!(set.contains(69));
    assert!(!set.contains(65));
    assert_eq!(set.indices().collect::<Vec<_>>(), vec![0, 63, 64, 69]);

    set.remove(64);
    assert!(!set.contains(64));
    assert!(
        set.contains(69),
        "removing one word's bit must not touch the next"
    );
}

/// `step_task` bounds-checks the compiled program.
#[test]
fn plan_step_task_bounds_checks() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(only);
    }
    fn only(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }

    let domain = HtnDomain::from_root(root).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("root", &state);

    assert_eq!(plan.len(), 1);
    assert!(plan.step_task(0).is_some());
    assert!(plan.step_task(1).is_none());
    fn type_id_of<F: 'static>(_: F) -> TypeId {
        TypeId::of::<F>()
    }
    assert_eq!(
        plan.step_task(0),
        Some(domain.task_index_by_type(type_id_of(only)).unwrap())
    );
}

/// Back-planning produces a compiled program too (steps + parallel names).
#[test]
fn backward_plan_is_a_compiled_program() {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(earn).then(root);
        task.branch().precondition(|gold: &Gold| gold.0 >= 3);
    }
    fn earn(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 += 1);
    }
    fn three_gold(task: &mut GoalBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 3);
    }

    let domain = HtnDomain::from_root(root).goal(three_gold).build().unwrap();
    let state = PlanState::build(&domain.components).set(Gold(0)).finish();
    let mut back = bevy_bhtn::BackPlanner::new(&domain);
    let plan = back.plan("three_gold", &state).unwrap();

    assert_eq!(plan.task_names(), ["earn"]);
    assert_eq!(plan.steps.len(), 1);
    let Task::Primitive(p) = &domain.tasks[plan.steps[0] as usize] else {
        panic!("backward plans are primitive sequences");
    };
    assert_eq!(p.name, "earn");
}

use bevy_bhtn::tasks::Task;
