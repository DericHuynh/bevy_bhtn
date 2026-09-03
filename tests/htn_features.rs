//! Feature-level coverage for `bevy_bhtn`: the closure precondition/effect
//! variant matrix (1..=8 annotated params), builder-validation error paths,
//! `.action` storage, `expected_effects` chaining vs. execution, MTR/plan
//! ordering, summaries' read/write sets, domain helpers, and error-variant
//! shapes. These pin the *semantics* of each surface so future optimization
//! work can't silently change behaviour.

mod common;

use bevy_bhtn::back_planner::BackPlanner;
use bevy_bhtn::planner::{HtnPlanner, Plan, PlanStatus};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::TaskBuilder;
use bevy_bhtn::{GoalBuilder, HtnDomain, HtnError, Task};
use bevy_ecs::prelude::Component;
use bevy_ecs::system::EntityCommands;

use common::{execute_plan, HtnTestBed};

// ---------------------------------------------------------------------------
// Components covering every value shape the closures can touch.
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Flag(pub bool);

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Count(pub i32);

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Weight(pub f32);

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Maybe(pub Option<i32>);

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Left(pub i32);

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Right(pub i32);

#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
enum Zone {
    #[default]
    Inside,
    Outside,
}

/// Eight distinct components so arity-1..=8 effect closures can take disjoint
/// mutable slots (duplicate slots would panic in `get_disjoint_mut`).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P1(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P2(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P3(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P4(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P5(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P6(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P7(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct P8(pub i32);

// ---------------------------------------------------------------------------
// Condition matrix domain — one primitive per old condition-variant group.
// ---------------------------------------------------------------------------

mod cond_tasks {
    use super::*;

    pub fn cond_root(task: &mut TaskBuilder) {
        task.branch().then(pre_bool);
        task.branch().then(pre_int_float);
        task.branch().then(pre_none);
        task.branch().then(pre_some);
        task.branch().then(pre_enum);
        task.branch().then(pre_not_enum);
        task.branch().then(pre_fields);
        task.branch().then(pre_order);
    }

    pub fn pre_bool(task: &mut TaskBuilder) {
        task.precondition(|f: &Flag| f.0);
    }

    pub fn pre_int_float(task: &mut TaskBuilder) {
        task.precondition(|c: &Count, w: &Weight| c.0 == 7 && w.0 == 2.5);
    }

    pub fn pre_none(task: &mut TaskBuilder) {
        task.precondition(|m: &Maybe| m.0.is_none());
    }

    pub fn pre_some(task: &mut TaskBuilder) {
        task.precondition(|m: &Maybe| m.0.is_some());
    }

    pub fn pre_enum(task: &mut TaskBuilder) {
        task.precondition(|z: &Zone| *z == Zone::Outside);
    }

    pub fn pre_not_enum(task: &mut TaskBuilder) {
        task.precondition(|z: &Zone| *z != Zone::Inside);
    }

    pub fn pre_fields(task: &mut TaskBuilder) {
        task.precondition(|l: &Left, r: &Right| l.0 == r.0);
    }

    pub fn pre_order(task: &mut TaskBuilder) {
        task.precondition(|c: &Count, w: &Weight, l: &Left, r: &Right| {
            c.0 > 3 && c.0 <= 5 && w.0 > 1.0 && w.0 <= 2.0 && l.0 >= r.0 && r.0 <= l.0
        });
    }
}

fn cond_domain() -> HtnDomain {
    use cond_tasks::*;
    HtnDomain::from_root(cond_root)
        .build()
        .expect("condition-matrix domain is well-formed")
}

/// A scratchpad matching every condition's true-case.
fn cond_state_true(domain: &HtnDomain) -> PlanState {
    PlanState::build(&domain.components)
        .set(Flag(true))
        .set(Count(5))
        .set(Weight(1.5))
        .set(Maybe(None))
        .set(Zone::Outside)
        .set(Left(9))
        .set(Right(3))
        .finish()
}

// ---------------------------------------------------------------------------
// Precondition variant matrix (old `cond::*` evaluate pins, closure API)
// ---------------------------------------------------------------------------

#[test]
fn precondition_single_component_evaluates() {
    let domain = cond_domain();
    let on = cond_state_true(&domain);
    let off = PlanState::build(&domain.components)
        .set(Flag(false))
        .finish();

    let pre_bool = domain.get_task("pre_bool").expect("pre_bool recorded");
    let Task::Primitive(p) = pre_bool else {
        panic!("pre_bool must be a primitive");
    };
    assert!(p.preconditions_met(&on));
    assert!(!p.preconditions_met(&off));

    // eq_int + eq_float analogs: exact equality on two numeric components.
    let ints = PlanState::build(&domain.components)
        .set(Count(7))
        .set(Weight(2.5))
        .finish();
    let wrong = PlanState::build(&domain.components)
        .set(Count(6))
        .set(Weight(2.5))
        .finish();
    let Task::Primitive(p) = domain.get_task("pre_int_float").expect("recorded") else {
        panic!("pre_int_float must be a primitive");
    };
    assert!(p.preconditions_met(&ints));
    assert!(!p.preconditions_met(&wrong));
}

#[test]
fn precondition_option_and_enum_evaluates() {
    let domain = cond_domain();
    let none = PlanState::build(&domain.components)
        .set(Maybe(None))
        .finish();
    let some = PlanState::build(&domain.components)
        .set(Maybe(Some(3)))
        .finish();

    let Task::Primitive(p) = domain.get_task("pre_none").expect("recorded") else {
        panic!("pre_none must be a primitive");
    };
    assert!(p.preconditions_met(&none));
    assert!(!p.preconditions_met(&some));

    let Task::Primitive(p) = domain.get_task("pre_some").expect("recorded") else {
        panic!("pre_some must be a primitive");
    };
    assert!(p.preconditions_met(&some));
    assert!(!p.preconditions_met(&none));

    // eq_enum / neq_enum analogs.
    let outside = PlanState::build(&domain.components)
        .set(Zone::Outside)
        .finish();
    let inside = PlanState::build(&domain.components)
        .set(Zone::Inside)
        .finish();
    let Task::Primitive(p) = domain.get_task("pre_enum").expect("recorded") else {
        panic!("pre_enum must be a primitive");
    };
    assert!(p.preconditions_met(&outside));
    assert!(!p.preconditions_met(&inside));
    let Task::Primitive(p) = domain.get_task("pre_not_enum").expect("recorded") else {
        panic!("pre_not_enum must be a primitive");
    };
    assert!(p.preconditions_met(&outside));
    assert!(!p.preconditions_met(&inside)); // Inside != Inside is false
}

#[test]
fn precondition_field_comparison_and_ordering_evaluates() {
    let domain = cond_domain();
    let equal = PlanState::build(&domain.components)
        .set(Left(4))
        .set(Right(4))
        .finish();
    let unequal = PlanState::build(&domain.components)
        .set(Left(4))
        .set(Right(5))
        .finish();

    let Task::Primitive(p) = domain.get_task("pre_fields").expect("recorded") else {
        panic!("pre_fields must be a primitive");
    };
    assert!(p.preconditions_met(&equal));
    assert!(!p.preconditions_met(&unequal));

    // gt/lte int, gt/lte float, gte/lte field-vs-field — all in one closure.
    let ordered = PlanState::build(&domain.components)
        .set(Count(5))
        .set(Weight(1.5))
        .set(Left(9))
        .set(Right(3))
        .finish();
    let unordered = PlanState::build(&domain.components)
        .set(Count(2))
        .set(Weight(1.5))
        .set(Left(9))
        .set(Right(3))
        .finish();
    let Task::Primitive(p) = domain.get_task("pre_order").expect("recorded") else {
        panic!("pre_order must be a primitive");
    };
    assert!(p.preconditions_met(&ordered));
    assert!(!p.preconditions_met(&unordered));
}

// ---------------------------------------------------------------------------
// Arity matrix: preconditions with 1..=8 annotated params compile & evaluate.
// ---------------------------------------------------------------------------

mod arity_tasks {
    use super::*;

    pub fn arity_root(task: &mut TaskBuilder) {
        task.branch()
            .then(pre1)
            .then(pre2)
            .then(pre3)
            .then(pre4)
            .then(pre5)
            .then(pre6)
            .then(pre7)
            .then(pre8);
        task.branch()
            .then(eff1)
            .then(eff2)
            .then(eff3)
            .then(eff4)
            .then(eff5)
            .then(eff6)
            .then(eff7)
            .then(eff8);
    }

    pub fn pre1(task: &mut TaskBuilder) {
        task.precondition(|a: &P1| a.0 > 0);
    }
    pub fn pre2(task: &mut TaskBuilder) {
        task.precondition(|a: &P1, b: &P2| a.0 > 0 && b.0 > 0);
    }
    pub fn pre3(task: &mut TaskBuilder) {
        task.precondition(|a: &P1, b: &P2, c: &P3| a.0 > 0 && b.0 > 0 && c.0 > 0);
    }
    pub fn pre4(task: &mut TaskBuilder) {
        task.precondition(|a: &P1, b: &P2, c: &P3, d: &P4| {
            a.0 > 0 && b.0 > 0 && c.0 > 0 && d.0 > 0
        });
    }
    pub fn pre5(task: &mut TaskBuilder) {
        task.precondition(|a: &P1, b: &P2, c: &P3, d: &P4, e: &P5| {
            a.0 > 0 && b.0 > 0 && c.0 > 0 && d.0 > 0 && e.0 > 0
        });
    }
    pub fn pre6(task: &mut TaskBuilder) {
        task.precondition(|a: &P1, b: &P2, c: &P3, d: &P4, e: &P5, f: &P6| {
            a.0 > 0 && b.0 > 0 && c.0 > 0 && d.0 > 0 && e.0 > 0 && f.0 > 0
        });
    }
    pub fn pre7(task: &mut TaskBuilder) {
        task.precondition(|a: &P1, b: &P2, c: &P3, d: &P4, e: &P5, f: &P6, g: &P7| {
            a.0 > 0 && b.0 > 0 && c.0 > 0 && d.0 > 0 && e.0 > 0 && f.0 > 0 && g.0 > 0
        });
    }
    pub fn pre8(task: &mut TaskBuilder) {
        task.precondition(
            |a: &P1, b: &P2, c: &P3, d: &P4, e: &P5, f: &P6, g: &P7, h: &P8| {
                a.0 > 0 && b.0 > 0 && c.0 > 0 && d.0 > 0 && e.0 > 0 && f.0 > 0 && g.0 > 0 && h.0 > 0
            },
        );
    }

    pub fn eff1(task: &mut TaskBuilder) {
        task.effect(|a: &mut P1| a.0 = 1);
    }
    pub fn eff2(task: &mut TaskBuilder) {
        task.effect(|a: &mut P1, b: &mut P2| {
            a.0 = 1;
            b.0 = 2;
        });
    }
    pub fn eff3(task: &mut TaskBuilder) {
        task.effect(|a: &mut P1, b: &mut P2, c: &mut P3| {
            a.0 = 1;
            b.0 = 2;
            c.0 = 3;
        });
    }
    pub fn eff4(task: &mut TaskBuilder) {
        task.effect(|a: &mut P1, b: &mut P2, c: &mut P3, d: &mut P4| {
            a.0 = 1;
            b.0 = 2;
            c.0 = 3;
            d.0 = 4;
        });
    }
    pub fn eff5(task: &mut TaskBuilder) {
        task.effect(
            |a: &mut P1, b: &mut P2, c: &mut P3, d: &mut P4, e: &mut P5| {
                a.0 = 1;
                b.0 = 2;
                c.0 = 3;
                d.0 = 4;
                e.0 = 5;
            },
        );
    }
    pub fn eff6(task: &mut TaskBuilder) {
        task.effect(
            |a: &mut P1, b: &mut P2, c: &mut P3, d: &mut P4, e: &mut P5, f: &mut P6| {
                a.0 = 1;
                b.0 = 2;
                c.0 = 3;
                d.0 = 4;
                e.0 = 5;
                f.0 = 6;
            },
        );
    }
    pub fn eff7(task: &mut TaskBuilder) {
        task.effect(
            |a: &mut P1, b: &mut P2, c: &mut P3, d: &mut P4, e: &mut P5, f: &mut P6, g: &mut P7| {
                a.0 = 1;
                b.0 = 2;
                c.0 = 3;
                d.0 = 4;
                e.0 = 5;
                f.0 = 6;
                g.0 = 7;
            },
        );
    }
    pub fn eff8(task: &mut TaskBuilder) {
        task.effect(
            |a: &mut P1,
             b: &mut P2,
             c: &mut P3,
             d: &mut P4,
             e: &mut P5,
             f: &mut P6,
             g: &mut P7,
             h: &mut P8| {
                a.0 = 1;
                b.0 = 2;
                c.0 = 3;
                d.0 = 4;
                e.0 = 5;
                f.0 = 6;
                g.0 = 7;
                h.0 = 8;
            },
        );
    }
}

fn arity_domain() -> HtnDomain {
    use arity_tasks::*;
    HtnDomain::from_root(arity_root)
        .build()
        .expect("arity domain is well-formed")
}

#[test]
fn precondition_arity_up_to_eight_compiles_and_evaluates() {
    let domain = arity_domain();
    let all_positive = PlanState::build(&domain.components)
        .set(P1(1))
        .set(P2(2))
        .set(P3(3))
        .set(P4(4))
        .set(P5(5))
        .set(P6(6))
        .set(P7(7))
        .set(P8(8))
        .finish();
    let p4_zero = PlanState::build(&domain.components)
        .set(P1(1))
        .set(P2(2))
        .set(P3(3))
        .set(P4(0))
        .set(P5(5))
        .set(P6(6))
        .set(P7(7))
        .set(P8(8))
        .finish();

    for n in 1..=8 {
        let name = format!("pre{n}");
        let Task::Primitive(p) = domain.get_task(&name).expect("arity precondition recorded")
        else {
            panic!("{name} must be a primitive");
        };
        assert!(p.preconditions_met(&all_positive), "{name} should pass");
        // A zeroed P4 only refutes the closures that read it (pre4..pre8).
        let expected = n < 4;
        assert_eq!(
            p.preconditions_met(&p4_zero),
            expected,
            "{name} vs zeroed P4"
        );
    }
}

// ---------------------------------------------------------------------------
// Effect variant matrix (old `eff::*` apply pins, closure API)
// ---------------------------------------------------------------------------

mod effect_tasks {
    use super::*;

    pub fn effect_root(task: &mut TaskBuilder) {
        task.branch()
            .then(set_all)
            .then(copy_and_increment)
            .then(clear_maybe);
    }

    /// set_int / set_float / set_enum / set_bool analogs in one primitive.
    pub fn set_all(task: &mut TaskBuilder) {
        task.effect(
            |c: &mut Count, w: &mut Weight, z: &mut Zone, f: &mut Flag| {
                c.0 = 42;
                w.0 = 3.25;
                *z = Zone::Outside;
                f.0 = true;
            },
        );
    }

    /// set_from (copy one component's value into another) and inc_int /
    /// inc_float analogs.
    pub fn copy_and_increment(task: &mut TaskBuilder) {
        task.effect(|l: &mut Left, c: &mut Count, w: &mut Weight| {
            l.0 = c.0;
            c.0 += 5;
            w.0 += 0.5;
        });
    }

    /// set_none analog.
    pub fn clear_maybe(task: &mut TaskBuilder) {
        task.effect(|m: &mut Maybe| m.0 = None);
    }
}

fn effect_domain() -> HtnDomain {
    use effect_tasks::*;
    HtnDomain::from_root(effect_root)
        .build()
        .expect("effect-matrix domain is well-formed")
}

#[test]
fn effect_single_component_mutates() {
    let domain = effect_domain();
    let mut state = PlanState::build(&domain.components)
        .set(Maybe(Some(3)))
        .finish();

    let Task::Primitive(p) = domain.get_task("set_all").expect("recorded") else {
        panic!("set_all must be a primitive");
    };
    for e in &p.effects {
        e.apply(&mut state);
    }
    assert_eq!(
        state
            .get::<Count>(domain.components.get::<Count>().unwrap())
            .0,
        42
    );
    assert_eq!(
        state
            .get::<Weight>(domain.components.get::<Weight>().unwrap())
            .0,
        3.25
    );
    assert_eq!(
        state.get::<Zone>(domain.components.get::<Zone>().unwrap()),
        &Zone::Outside
    );
    assert!(
        state
            .get::<Flag>(domain.components.get::<Flag>().unwrap())
            .0
    );

    let Task::Primitive(p) = domain.get_task("clear_maybe").expect("recorded") else {
        panic!("clear_maybe must be a primitive");
    };
    for e in &p.effects {
        e.apply(&mut state);
    }
    assert!(state
        .get::<Maybe>(domain.components.get::<Maybe>().unwrap())
        .0
        .is_none());
}

#[test]
fn effect_multi_component_mutates() {
    let domain = effect_domain();
    let mut state = PlanState::build(&domain.components)
        .set(Count(10))
        .set(Weight(1.0))
        .finish();

    let Task::Primitive(p) = domain.get_task("copy_and_increment").expect("recorded") else {
        panic!("copy_and_increment must be a primitive");
    };
    for e in &p.effects {
        e.apply(&mut state);
    }
    // set_from analog: Left copied from Count *before* Count was incremented.
    assert_eq!(
        state
            .get::<Left>(domain.components.get::<Left>().unwrap())
            .0,
        10
    );
    // inc_int / inc_float analogs.
    assert_eq!(
        state
            .get::<Count>(domain.components.get::<Count>().unwrap())
            .0,
        15
    );
    assert_eq!(
        state
            .get::<Weight>(domain.components.get::<Weight>().unwrap())
            .0,
        1.5
    );
}

#[test]
fn effect_arity_up_to_eight_mutates() {
    let domain = arity_domain();
    let mut state = PlanState::build(&domain.components).finish();

    let slots = [
        domain.components.get::<P1>().unwrap(),
        domain.components.get::<P2>().unwrap(),
        domain.components.get::<P3>().unwrap(),
        domain.components.get::<P4>().unwrap(),
        domain.components.get::<P5>().unwrap(),
        domain.components.get::<P6>().unwrap(),
        domain.components.get::<P7>().unwrap(),
        domain.components.get::<P8>().unwrap(),
    ];
    for n in 1..=8 {
        let name = format!("eff{n}");
        let Task::Primitive(p) = domain.get_task(&name).expect("arity effect recorded") else {
            panic!("{name} must be a primitive");
        };
        p.apply_effects(&mut state);
        for k in 1..=n {
            let value = match k {
                1 => state.get::<P1>(slots[0]).0,
                2 => state.get::<P2>(slots[1]).0,
                3 => state.get::<P3>(slots[2]).0,
                4 => state.get::<P4>(slots[3]).0,
                5 => state.get::<P5>(slots[4]).0,
                6 => state.get::<P6>(slots[5]).0,
                7 => state.get::<P7>(slots[6]).0,
                _ => state.get::<P8>(slots[7]).0,
            };
            assert_eq!(value, k as i32, "{name} wrote P{k}");
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed `&`/`&mut` effect closures: read-only parameters are registered (the
// closure can read them) but are NOT part of the effect's write set — never
// journaled for rollback and never committed to the real entity.
// ---------------------------------------------------------------------------

mod mixed_effect_tasks {
    use super::*;

    pub fn mixed_root(task: &mut TaskBuilder) {
        task.branch().then(mixed_step);
    }

    /// Writes Flag and Maybe; reads Count.
    pub fn mixed_step(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag, c: &Count, m: &mut Maybe| {
            f.0 = c.0 > 0;
            m.0 = Some(c.0);
        });
    }
}

#[test]
fn mixed_effect_reads_are_not_journaled_or_committed() {
    use mixed_effect_tasks::*;
    let domain = HtnDomain::from_root(mixed_root)
        .build()
        .expect("mixed-effect domain is well-formed");

    // The compiled effect's write set excludes the read-only Count slot.
    let Task::Primitive(p) = domain.get_task("mixed_step").expect("recorded") else {
        panic!("mixed_step must be a primitive");
    };
    let flag = domain.components.get::<Flag>().unwrap();
    let count = domain.components.get::<Count>().unwrap();
    let maybe = domain.components.get::<Maybe>().unwrap();
    let writes: Vec<usize> = p.write_slots().collect();
    assert!(writes.contains(&flag) && writes.contains(&maybe));
    assert!(!writes.contains(&count), "read-only params are not writes");

    // Driver execution: the effect reads the world's Count through the
    // scratchpad, but only the written slots are committed back — the world's
    // Count is untouched even though the closure observed it.
    use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};
    use bevy_ecs::world::World;
    let mut world = World::new();
    world.insert_resource(HtnConfig::new(domain));
    let entity = world
        .spawn((Flag(false), Count(7), Maybe(None), HtnAgent::default()))
        .id();
    htn_ai_system(&mut world);
    assert!(world.get::<Flag>(entity).unwrap().0, "Flag committed");
    assert_eq!(
        world.get::<Maybe>(entity).unwrap().0,
        Some(7),
        "Maybe committed"
    );
    assert_eq!(
        world.get::<Count>(entity).unwrap().0,
        7,
        "read-only Count was never committed"
    );
}

// ---------------------------------------------------------------------------
// Builder-validation error paths (old parser-error pins)
// ---------------------------------------------------------------------------

#[test]
fn mixed_declarations_yield_builder_error() {
    fn mixed(task: &mut TaskBuilder) {
        task.branch(); // compound declaration …
        task.precondition(|f: &Flag| f.0); // … mixed with a primitive one
    }
    let err = HtnDomain::from_root(mixed).build().unwrap_err();
    assert!(matches!(err, HtnError::Builder { .. }));
    assert!(err.to_string().contains("mixes"));
}

#[test]
fn branchless_root_yields_builder_error() {
    fn empty_root(_task: &mut TaskBuilder) {}
    let err = HtnDomain::from_root(empty_root).build().unwrap_err();
    assert!(matches!(err, HtnError::Builder { .. }));
    assert!(err.to_string().contains("no branches"));
}

/// Closure subtasks get unique identities: each closure literal is its own
/// type, so two distinct closures record as two distinct tasks and both
/// execute — even though `type_name` mangles both to `{{closure}}`. This is
/// the regression pin for the old lookup-by-name failure (all closures
/// collided on one display name); identity is now the `TypeId` alone.
#[test]
fn closure_subtasks_record_unique_identities() {
    use std::any::TypeId;
    fn tid_of<T: 'static>(_: &T) -> TypeId {
        TypeId::of::<T>()
    }

    let first = |t: &mut TaskBuilder| {
        t.effect(|count: &mut Count| count.0 = 5);
    };
    let second = |t: &mut TaskBuilder| {
        t.effect(|flag: &mut Flag| flag.0 = true);
    };
    let root = move |task: &mut TaskBuilder| {
        task.branch().then(first).then(second);
    };

    let domain = HtnDomain::from_root(root)
        .build()
        .expect("distinct closures are distinct identities");
    // Each closure resolved to its own task (root + 2 closure tasks).
    assert_eq!(domain.tasks.len(), 3);
    let a_idx = domain.task_index_by_type(tid_of(&first)).unwrap();
    let b_idx = domain.task_index_by_type(tid_of(&second)).unwrap();
    assert_ne!(a_idx, b_idx, "the two closures must not alias");

    // Both closure steps execute.
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(root, &state);
    assert_eq!(plan.task_names().len(), 2);
    let mut executed = state.clone();
    for &step in &plan.steps {
        let Task::Primitive(p) = &domain.tasks[step as usize] else {
            panic!("plans are primitive sequences");
        };
        p.apply_effects(&mut executed);
    }
    let count = domain.components.get::<Count>().unwrap();
    let flag = domain.components.get::<Flag>().unwrap();
    assert_eq!(executed.get::<Count>(count).0, 5);
    assert!(executed.get::<Flag>(flag).0);
}

/// Referencing the *same* closure value from several `then` edges records the
/// task exactly once (edge dedup by `TypeId` — identical to repeated fn-item
/// references) while every occurrence still executes.
#[test]
fn same_closure_value_dedupes_to_one_recorded_task() {
    let step = |t: &mut TaskBuilder| {
        t.effect(|count: &mut Count| count.0 += 1);
    };
    let root = move |task: &mut TaskBuilder| {
        task.branch().then(step).then(step).then(step);
    };

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    // root + the single closure task — three references, one recording.
    assert_eq!(domain.tasks.len(), 2, "the closure task is recorded once");

    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(root, &state);
    assert_eq!(plan.task_names().len(), 3, "each edge still executes");
    let mut executed = state.clone();
    for &step in &plan.steps {
        let Task::Primitive(p) = &domain.tasks[step as usize] else {
            panic!("plans are primitive sequences");
        };
        p.apply_effects(&mut executed);
    }
    let count = domain.components.get::<Count>().unwrap();
    assert_eq!(executed.get::<Count>(count).0, 3);
}

/// Closure tasks are displayed by their REFERENCE SITE (`file:line:col`, via
/// `#[track_caller]`) instead of the mangled, collision-prone `{{closure}}`;
/// the braces are stripped. Named functions keep their clean names. Display
/// only — identity is the `TypeId` either way.
#[test]
fn closure_display_names_use_the_reference_site() {
    let first = |t: &mut TaskBuilder| {
        t.effect(|count: &mut Count| count.0 = 1);
    };
    let second = |t: &mut TaskBuilder| {
        t.effect(|count: &mut Count| count.0 += 10);
    };
    let root = move |task: &mut TaskBuilder| {
        task.branch().then(first).then(second);
    };

    let domain = HtnDomain::from_root(root).build().expect("well-formed");
    // No mangled name survives baking — every closure is a location.
    for t in &domain.tasks {
        assert!(
            !t.name().contains("{{closure"),
            "mangled display name leaked: {:?}",
            t.name()
        );
    }
    // All three tasks are closures: root (the `from_root` call site) and the
    // two `then` sites inside the root body. Distinct sites → distinct names,
    // each shaped `htn_features.rs:LINE:COL`.
    let names: Vec<&str> = domain.tasks.iter().map(Task::name).collect();
    assert_eq!(names.len(), 3);
    for n in &names {
        assert!(
            n.contains("htn_features.rs:") && n.matches(':').count() >= 2,
            "expected `file:line:col`, got {n:?}"
        );
    }
    assert_eq!(&names[1..], &names[1..], "sanity");
    assert_ne!(
        names[1], names[2],
        "same-line `then` sites differ by column"
    );

    // And named functions keep their clean type-derived names — both the
    // root fn and subtask fns.
    fn named(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 = 0);
    }
    fn tidy_root(task: &mut TaskBuilder) {
        task.branch().then(named);
    }
    let tidy = HtnDomain::from_root(tidy_root)
        .build()
        .expect("well-formed");
    let tidy_names: Vec<&str> = tidy.tasks.iter().map(Task::name).collect();
    assert_eq!(tidy_names, ["tidy_root", "named"]);
}

/// Coercing task functions to `fn(&mut TaskBuilder)` pointers collapses their
/// identities into ONE `TypeId`: the second body is never recorded, and both
/// references silently execute the first. This is the known fn-pointer trap
/// (the same reason `any_order` takes tuples, not arrays) — pass fn items,
/// never pointers. Pinned so a future identity scheme that fixes it trips
/// this test.
#[test]
fn fn_pointer_coercion_collapses_identity() {
    fn strike(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    fn bash(task: &mut TaskBuilder) {
        task.effect(|flag: &mut Flag| flag.0 = true);
    }
    let root = |task: &mut TaskBuilder| {
        task.branch()
            .then(strike as fn(&mut TaskBuilder))
            .then(bash as fn(&mut TaskBuilder));
    };

    // The collapse is silent: bake succeeds.
    let domain = HtnDomain::from_root(root).build().expect("bakes");
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(root, &state);
    assert_eq!(
        plan.task_names().len(),
        2,
        "two edges — but both point at the first recorded body"
    );
    let mut executed = state.clone();
    for &step in &plan.steps {
        let Task::Primitive(p) = &domain.tasks[step as usize] else {
            panic!("plans are primitive sequences");
        };
        p.apply_effects(&mut executed);
    }
    let count = domain.components.get::<Count>().unwrap();
    assert_eq!(executed.get::<Count>(count).0, 2, "strike's body ran twice");
    // bash's body never even recorded — its Flag effect never registered the
    // component in the domain's registry. Silent, total loss of the task.
    assert!(
        domain.components.get::<Flag>().is_none(),
        "bash's body never ran — the trap"
    );
}

// ---------------------------------------------------------------------------
// `.action` storage on primitives (old operator-verification pins)
// ---------------------------------------------------------------------------

#[test]
fn action_is_stored_on_primitive() {
    fn acted(task: &mut TaskBuilder) {
        task.precondition(|f: &Flag| f.0)
            .effect(|c: &mut Count| c.0 = 1);
        let _ = task.action(|cmds: &mut EntityCommands| {
            cmds.despawn();
        });
    }
    fn plain(task: &mut TaskBuilder) {
        task.effect(|c: &mut Count| c.0 = 2);
    }
    fn action_root(task: &mut TaskBuilder) {
        task.branch().then(acted).then(plain);
    }

    let domain = HtnDomain::from_root(action_root)
        .build()
        .expect("action domain is well-formed");

    let Task::Primitive(p) = domain.get_task("acted").expect("acted recorded") else {
        panic!("acted must be a primitive");
    };
    assert!(p.action.is_some());
    assert_eq!(p.effects.len(), 1);
    assert_eq!(p.preconditions.len(), 1);

    let Task::Primitive(p) = domain.get_task("plain").expect("plain recorded") else {
        panic!("plain must be a primitive");
    };
    assert!(p.action.is_none());
}

// ---------------------------------------------------------------------------
// Plan / MTR ordering
// ---------------------------------------------------------------------------

#[test]
fn plan_mtr_ordering() {
    let low = Plan {
        steps: vec![0],
        names: vec!["A".into()],
        mtr: vec![0],
        status: PlanStatus::Complete,
    };
    let high = Plan {
        steps: vec![1],
        names: vec!["B".into()],
        mtr: vec![1],
        status: PlanStatus::Complete,
    };
    assert!(low.is_preferred_over(&high));
    assert!(!high.is_preferred_over(&low));
    assert_eq!(format!("{:?}", low.mtr()), "[0]");
    assert_eq!(format!("{:?}", high.mtr()), "[1]");
}

// ---------------------------------------------------------------------------
// Domain helpers
// ---------------------------------------------------------------------------

mod helper_tasks {
    use super::*;

    pub fn helper_root(task: &mut TaskBuilder) {
        task.branch().then(helper_leaf);
    }

    pub fn helper_leaf(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag| f.0 = true);
    }

    pub fn helper_goal(goal: &mut GoalBuilder) {
        goal.effect(|f: &mut Flag| f.0 = true);
    }
}

#[test]
fn domain_root_goal_and_primitive_names() {
    use helper_tasks::*;
    let domain = HtnDomain::from_root(helper_root)
        .goal(helper_goal)
        .build()
        .expect("helper domain is well-formed");
    assert_eq!(domain.root_task().name(), "helper_root");
    assert!(domain.goal(helper_goal).is_some());
    assert_eq!(
        domain
            .primitive_names()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>(),
        vec!["helper_leaf"]
    );
    // Unregistered names resolve to nothing.
    assert!(domain.get_task("Missing").is_none());
    fn missing_goal(_goal: &mut GoalBuilder) {}
    assert!(domain.goal(missing_goal).is_none());
}

// ---------------------------------------------------------------------------
// expected_effects chaining: a hoped effect lets a later task's precondition
// pass during planning, even though it isn't a guaranteed effect — and it is
// never applied by execution (`execute_plan` applies `effects` only).
// ---------------------------------------------------------------------------

mod travel_tasks {
    use super::*;

    pub fn travel_root(task: &mut TaskBuilder) {
        task.branch().then(approach).then(arrive);
    }

    pub fn approach(task: &mut TaskBuilder) {
        task.expected(|z: &mut Zone| *z = Zone::Outside);
    }

    pub fn arrive(task: &mut TaskBuilder) {
        task.precondition(|z: &Zone| *z == Zone::Outside)
            .effect(|c: &mut Count| c.0 = 7);
    }
}

#[test]
fn expected_effects_chain_preconditions() {
    use travel_tasks::*;
    let domain = HtnDomain::from_root(travel_root)
        .build()
        .expect("travel domain is well-formed");
    let bed = HtnTestBed::new(domain);
    let state = PlanState::build(&bed.domain().components)
        .set(Zone::Inside)
        .finish();
    // Without expected_effects being simulated, arrive's `Zone == Outside`
    // would block the plan; with it, both tasks plan in sequence.
    assert_eq!(bed.plan_forward(&state), vec!["approach", "arrive"]);
}

#[test]
fn expected_effects_do_not_apply_on_execution() {
    use travel_tasks::*;
    let domain = HtnDomain::from_root(travel_root)
        .build()
        .expect("travel domain is well-formed");
    let mut state = PlanState::build(&domain.components)
        .set(Zone::Inside)
        .set(Count(0))
        .finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(travel_root, &state);
    assert_eq!(plan.task_names(), ["approach", "arrive"]);

    // Execution applies `effects` only: arrive's real Count write lands, but
    // approach's hoped Zone write never does.
    execute_plan(&domain, &mut state, &plan);
    assert_eq!(
        state
            .get::<Count>(domain.components.get::<Count>().unwrap())
            .0,
        7
    );
    assert_eq!(
        state.get::<Zone>(domain.components.get::<Zone>().unwrap()),
        &Zone::Inside
    );
}

// ---------------------------------------------------------------------------
// Summaries: precondition reads and effect writes land in the read/write sets
// (slot indices resolved through the component registry).
// ---------------------------------------------------------------------------

mod summary_tasks {
    use super::*;

    pub fn summary_root(task: &mut TaskBuilder) {
        task.branch().then(summary_leaf);
    }

    pub fn summary_leaf(task: &mut TaskBuilder) {
        task.precondition(|f: &Flag, c: &Count| f.0 && c.0 > 0)
            .effect(|w: &mut Weight| w.0 = 1.0)
            .expected(|m: &mut Maybe| m.0 = Some(1));
    }
}

#[test]
fn summaries_reflect_precondition_reads_and_effect_writes() {
    use summary_tasks::*;
    let domain = HtnDomain::from_root(summary_root)
        .build()
        .expect("summary domain is well-formed");

    let flag = domain.components.get::<Flag>().expect("Flag registered");
    let count = domain.components.get::<Count>().expect("Count registered");
    let weight = domain
        .components
        .get::<Weight>()
        .expect("Weight registered");
    let maybe = domain.components.get::<Maybe>().expect("Maybe registered");

    let leaf = domain.task_summary(summary_leaf).expect("summary present");
    // Reads: both precondition components are required before any write.
    assert!(leaf.required_fields.contains(flag));
    assert!(leaf.required_fields.contains(count));
    assert!(!leaf.required_fields.contains(weight));
    // Possible writes: real + expected effects (over-approximation).
    assert!(leaf.possible_writes.contains(weight));
    assert!(leaf.possible_writes.contains(maybe));
    // Guaranteed writes: real effects only — expected effects are never
    // guaranteed.
    assert!(leaf.guaranteed_writes.contains(weight));
    assert!(!leaf.guaranteed_writes.contains(maybe));

    // The compound root inherits the leaf's write sets.
    let root = domain.task_summary(summary_root).expect("summary present");
    assert!(root.possible_writes.contains(weight));
    assert!(root.possible_writes.contains(maybe));
}

// ---------------------------------------------------------------------------
// Error-variant shapes
// ---------------------------------------------------------------------------

#[test]
fn unknown_goal_yields_unknown_task_error() {
    fn lonely_root(task: &mut TaskBuilder) {
        task.branch().then(lonely_leaf);
    }
    fn lonely_leaf(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag| f.0 = true);
    }
    // A goal fn that was never registered on the domain.
    fn missing_goal(goal: &mut GoalBuilder) {
        goal.effect(|f: &mut Flag| f.0 = true);
    }
    let domain = HtnDomain::from_root(lonely_root)
        .build()
        .expect("lonely domain is well-formed");
    let state = PlanState::build(&domain.components).finish();
    let mut planner = BackPlanner::new(&domain);
    let err = planner.plan(missing_goal, &state).unwrap_err();
    assert!(matches!(err, HtnError::UnregisteredTask { .. }));
}

#[test]
fn unreachable_goal_yields_no_plan_error() {
    fn stranded_root(task: &mut TaskBuilder) {
        task.branch().then(stranded_leaf);
    }
    fn stranded_leaf(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag| f.0 = true);
    }
    fn stranded_goal(goal: &mut GoalBuilder) {
        goal.effect(|m: &mut Maybe| m.0 = Some(1));
    }
    let domain = HtnDomain::from_root(stranded_root)
        .goal(stranded_goal)
        .build()
        .expect("stranded domain is well-formed");
    let state = PlanState::build(&domain.components).finish();
    let mut planner = BackPlanner::new(&domain);
    // No primitive writes `Maybe` -> the goal can never be advanced.
    let err = planner.plan(stranded_goal, &state).unwrap_err();
    assert!(matches!(err, HtnError::NoPlan));
}

// ---------------------------------------------------------------------------
// BackPlanner greedy tie-breaking: a leaf covering multiple goal fields wins.
// ---------------------------------------------------------------------------

mod tiebreak_tasks {
    use super::*;

    pub fn tie_root(task: &mut TaskBuilder) {
        task.branch().then(single_shot).then(double_shot);
    }

    pub fn single_shot(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag| f.0 = true);
    }

    pub fn double_shot(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag| f.0 = true)
            .effect(|c: &mut Count| c.0 = 7);
    }

    pub fn both_goal(goal: &mut GoalBuilder) {
        goal.effect(|f: &mut Flag| f.0 = true)
            .effect(|c: &mut Count| c.0 = 7);
    }
}

#[test]
fn backward_plan_commits_full_coverage_compound() {
    use tiebreak_tasks::*;
    let domain = HtnDomain::from_root(tie_root)
        .goal(both_goal)
        .build()
        .expect("tiebreak domain is well-formed");
    let state = PlanState::build(&domain.components).finish();
    let mut planner = BackPlanner::new(&domain);
    // Compound candidates participate: `tie_root`'s method guarantees both
    // needed slots, so it is committed (earlier task index wins the coverage
    // tie with `double_shot`), and its mandatory sequence brings the
    // redundant `single_shot` along. The plan still reaches the goal's
    // values — which the old primitive-only greedy also did here, but not in
    // value-recursive domains (see the htn_builder end-to-end pin).
    let plan = planner.plan(both_goal, &state).expect("back plan");
    let names = plan.task_names();
    assert!(!names.is_empty());
    assert_eq!(names, ["single_shot", "double_shot"]);
    let mut executed = state.clone();
    for &s in &plan.steps {
        if let Task::Primitive(p) = &domain.tasks[s as usize] {
            p.apply_effects(&mut executed);
        }
    }
    let flag = domain.components.get::<Flag>().unwrap();
    let count = domain.components.get::<Count>().unwrap();
    assert!(executed.get::<Flag>(flag).0);
    assert_eq!(executed.get::<Count>(count).0, 7);
}

mod combine_tasks {
    use super::*;

    pub fn combine_root(task: &mut TaskBuilder) {
        task.branch().then(set_flag).then(set_count);
    }

    pub fn set_flag(task: &mut TaskBuilder) {
        task.effect(|f: &mut Flag| f.0 = true);
    }

    pub fn set_count(task: &mut TaskBuilder) {
        task.effect(|c: &mut Count| c.0 = 1);
    }

    pub fn combine_goal(goal: &mut GoalBuilder) {
        goal.effect(|f: &mut Flag| f.0 = true)
            .effect(|c: &mut Count| c.0 = 1);
    }
}

#[test]
fn back_plan_combines_leaves_for_distinct_fields() {
    use combine_tasks::*;
    let domain = HtnDomain::from_root(combine_root)
        .goal(combine_goal)
        .build()
        .expect("combine domain is well-formed");
    let state = PlanState::build(&domain.components).finish();
    let mut planner = BackPlanner::new(&domain);
    // No single leaf covers both slots -> both are chained.
    let plan = planner.plan(combine_goal, &state).expect("back plan");
    let names = plan.task_names();
    assert_eq!(names.len(), 2);
    assert_ne!(names[0], names[1]);
}
