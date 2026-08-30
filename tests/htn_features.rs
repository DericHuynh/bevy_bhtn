//! Feature-level coverage for `cdda_htn`: the full condition/effect variant
//! matrix, verification error paths, `expected_effects` chaining, MTR/plan
//! ordering, domain helpers, and error-variant shapes. These pin the *semantics*
//! of each surface so future optimization work can't silently change behaviour.

use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_reflect::TypeRegistry;
use cdda_htn::operators::{construct_operator, verify_operator, Operator};
use cdda_htn::planner::Plan;
use cdda_htn::{Effect, HtnCondition, HtnError};

// ---------------------------------------------------------------------------
// A state covering every field type the conditions/effects can touch.
// ---------------------------------------------------------------------------

#[derive(Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Default)]
enum Zone {
    #[default]
    Inside,
    Outside,
}

#[derive(Reflect, Default, Clone, Debug, PartialEq)]
struct State {
    flag: bool,
    count: i32,
    weight: f32,
    zone: Zone,
    maybe: Option<i32>,
    left: i32,
    right: i32,
}

fn reg(registry: &mut TypeRegistry) {
    registry.register::<State>();
    registry.register::<Zone>();
}

fn treg() -> TypeRegistry {
    let mut registry = TypeRegistry::default();
    reg(&mut registry);
    registry
}

// ---------------------------------------------------------------------------
// Condition variant matrix
// ---------------------------------------------------------------------------

#[test]
fn condition_equals_bool() {
    let mut s = State::default();
    let on = HtnCondition::EqualsBool {
        field: "flag".into(),
        value: true,
        notted: false,
    };
    assert!(!on.evaluate(s.as_reflect()));
    s.flag = true;
    assert!(on.evaluate(s.as_reflect()));
}

#[test]
fn condition_equals_int_and_float() {
    let s = State {
        count: 7,
        weight: 2.5,
        ..Default::default()
    };
    assert!(HtnCondition::EqualsInt {
        field: "count".into(),
        value: 7,
        notted: false,
    }
    .evaluate(s.as_reflect()));
    assert!(HtnCondition::EqualsFloat {
        field: "weight".into(),
        value: 2.5,
        notted: false,
    }
    .evaluate(s.as_reflect()));
}

#[test]
fn condition_equals_none_and_some() {
    let none = State::default();
    let some = State {
        maybe: Some(3),
        ..Default::default()
    };
    assert!(HtnCondition::EqualsNone {
        field: "maybe".into(),
        notted: false,
    }
    .evaluate(none.as_reflect()));
    assert!(!HtnCondition::EqualsNone {
        field: "maybe".into(),
        notted: false,
    }
    .evaluate(some.as_reflect()));
    assert!(HtnCondition::EqualsNone {
        field: "maybe".into(),
        notted: true,
    }
    .evaluate(some.as_reflect()));
}

#[test]
fn condition_equals_enum() {
    let s = State {
        zone: Zone::Outside,
        ..Default::default()
    };
    let c = HtnCondition::EqualsEnum {
        field: "zone".into(),
        enum_type: "Zone".into(),
        enum_variant: "Outside".into(),
        notted: false,
    };
    assert!(c.evaluate(s.as_reflect()));
    let not = HtnCondition::EqualsEnum {
        field: "zone".into(),
        enum_type: "Zone".into(),
        enum_variant: "Inside".into(),
        notted: true,
    };
    assert!(not.evaluate(s.as_reflect()));
}

#[test]
fn condition_equals_identifier() {
    let s = State {
        left: 4,
        right: 4,
        ..Default::default()
    };
    assert!(HtnCondition::EqualsIdentifier {
        field: "left".into(),
        other_field: "right".into(),
        notted: false,
    }
    .evaluate(s.as_reflect()));
}

#[test]
fn condition_order_comparisons() {
    let s = State {
        count: 5,
        weight: 1.5,
        left: 9,
        right: 3,
        ..Default::default()
    };
    assert!(HtnCondition::GreaterThanInt {
        field: "count".into(),
        threshold: 3,
        orequals: false,
    }
    .evaluate(s.as_reflect()));
    assert!(HtnCondition::LessThanInt {
        field: "count".into(),
        threshold: 5,
        orequals: true,
    }
    .evaluate(s.as_reflect()));
    assert!(HtnCondition::GreaterThanFloat {
        field: "weight".into(),
        threshold: 1.0,
        orequals: false,
    }
    .evaluate(s.as_reflect()));
    assert!(HtnCondition::LessThanFloat {
        field: "weight".into(),
        threshold: 2.0,
        orequals: true,
    }
    .evaluate(s.as_reflect()));
    assert!(HtnCondition::GreaterThanIdentifier {
        field: "left".into(),
        other_field: "right".into(),
        orequals: true,
    }
    .evaluate(s.as_reflect()));
    assert!(HtnCondition::LessThanIdentifier {
        field: "right".into(),
        other_field: "left".into(),
        orequals: true,
    }
    .evaluate(s.as_reflect()));
}

// ---------------------------------------------------------------------------
// Effect variant matrix
// ---------------------------------------------------------------------------

#[test]
fn effect_set_int_float_none_enum_identifier() {
    let mut s = State::default();
    Effect::SetInt {
        field: "count".into(),
        value: 42,
    }
    .apply(&mut s, &treg());
    Effect::SetFloat {
        field: "weight".into(),
        value: 3.25,
    }
    .apply(&mut s, &treg());
    Effect::SetEnum {
        field: "zone".into(),
        enum_type: "Zone".into(),
        enum_variant: "Outside".into(),
    }
    .apply(&mut s, &treg());
    Effect::SetNone {
        field: "maybe".into(),
    }
    .apply(&mut s, &treg());
    Effect::SetIdentifier {
        field: "left".into(),
        field_source: "count".into(),
    }
    .apply(&mut s, &treg());
    assert_eq!(s.count, 42);
    assert_eq!(s.weight, 3.25);
    assert_eq!(s.zone, Zone::Outside);
    assert!(s.maybe.is_none());
    assert_eq!(s.left, 42);
}

#[test]
fn effect_increment_int_and_float() {
    let mut s = State {
        count: 10,
        weight: 1.0,
        ..Default::default()
    };
    Effect::IncrementInt {
        field: "count".into(),
        by: 5,
    }
    .apply(&mut s, &treg());
    Effect::IncrementInt {
        field: "count".into(),
        by: -4,
    }
    .apply(&mut s, &treg());
    Effect::IncrementFloat {
        field: "weight".into(),
        by: 0.5,
    }
    .apply(&mut s, &treg());
    assert_eq!(s.count, 11);
    assert_eq!(s.weight, 1.5);
}

// ---------------------------------------------------------------------------
// verify error paths
// ---------------------------------------------------------------------------

#[test]
fn verify_rejects_unknown_field() {
    let state = State::default();
    let c = HtnCondition::EqualsBool {
        field: "nope".into(),
        value: true,
        notted: false,
    };
    assert!(matches!(
        c.verify(state.as_reflect(), &treg()),
        Err(HtnError::Condition { .. })
    ));
}

#[test]
fn verify_rejects_unregistered_enum() {
    let state = State::default();
    let c = HtnCondition::EqualsEnum {
        field: "zone".into(),
        enum_type: "UnregisteredEnum".into(),
        enum_variant: "X".into(),
        notted: false,
    };
    assert!(matches!(
        c.verify(state.as_reflect(), &treg()),
        Err(HtnError::Condition { .. })
    ));
}

// ---------------------------------------------------------------------------
// Operator verification
// ---------------------------------------------------------------------------

#[derive(Reflect, Default, Clone, Debug, PartialEq, Eq)]
#[reflect(Default)]
struct NoopOp;

#[test]
fn verify_operator_rejects_missing_registration() {
    let registry = TypeRegistry::default();
    let err = verify_operator(&registry, "UnregisteredOp", &[]).unwrap_err();
    assert!(matches!(err, HtnError::Operator { .. }));
}

#[test]
fn construct_operator_returns_registered_default() {
    let mut registry = TypeRegistry::default();
    registry.register::<NoopOp>();
    let op = Operator {
        name: "NoopOp".into(),
        params: vec![],
    };
    let boxed = construct_operator(&op, &State::default(), &registry).expect("construct");
    assert!(boxed.is::<NoopOp>());
}

// ---------------------------------------------------------------------------
// Plan / MTR ordering
// ---------------------------------------------------------------------------

#[test]
fn plan_mtr_ordering() {
    let low = Plan {
        tasks: vec!["A".into()],
        mtr: cdda_htn::planner::Mtr(vec![0]),
    };
    let high = Plan {
        tasks: vec!["B".into()],
        mtr: cdda_htn::planner::Mtr(vec![1]),
    };
    assert!(low.is_preferred_over(&high));
    assert!(!high.is_preferred_over(&low));
    assert_eq!(low.mtr().to_string(), "0");
    assert_eq!(high.mtr().to_string(), "1");
}

// ---------------------------------------------------------------------------
// Domain helpers
// ---------------------------------------------------------------------------

#[test]
fn domain_root_goal_and_primitive_names() {
    let src = r#"
schema {
    version: 0.1.0
}

goal_task "G" {
    effects: [flag = true]
}

primitive_task "P" {
    operator: NoopOp
}

compound_task "Root" {
    method "m" {
        subtasks: [P]
    }
}
"#;
    let domain = cdda_htn::parse_htn(src).expect("parse");
    assert_eq!(domain.root_task().map(|t| t.name()), Some("Root"));
    assert!(domain.goal("G").is_some());
    assert_eq!(domain.primitive_names(), vec!["P"]);
    // First-task fallback when no compound exists.
    let only_src = r#"
schema {
    version: 0.1.0
}

primitive_task "X" {
    operator: NoopOp
}
"#;
    let only = cdda_htn::parse_htn(only_src).expect("parse only-primitive domain");
    assert_eq!(only.root_task().map(|t| t.name()), Some("X"));
}

// ---------------------------------------------------------------------------
// expected_effects chaining: a hoped effect lets a later task's precondition
// pass during planning, even though it isn't a guaranteed effect.
// ---------------------------------------------------------------------------

#[test]
fn expected_effects_chain_preconditions() {
    let src = r#"
schema {
    version: 0.1.0
}

compound_task "Travel" {
    method "go" {
        subtasks: [Approach, Arrive]
    }
}

primitive_task "Approach" {
    operator: NoopOp
    expected_effects: [zone = Zone::Outside]
}

primitive_task "Arrive" {
    operator: NoopOp
    preconditions: [zone == Zone::Outside]
}
"#;
    let registry = treg();
    let domain = cdda_htn::parse_htn(src).expect("parse");
    let mut planner = cdda_htn::planner::HtnPlanner::new(&domain, &registry);
    let plan = planner.plan("Travel", &State::default());
    // Without expected_effects being simulated, Arrive's `zone == Outside`
    // would block the plan; with it, both tasks plan in sequence.
    assert_eq!(plan.task_names(), vec!["Approach", "Arrive"]);
}

// ---------------------------------------------------------------------------
// Error-variant shapes
// ---------------------------------------------------------------------------

#[test]
fn malformed_htn_yields_parser_error() {
    let err = cdda_htn::parse_htn("this is not an htn").unwrap_err();
    assert!(matches!(err, HtnError::Parser { .. }));
}

#[test]
fn unknown_goal_yields_unknown_task_error() {
    let src = r#"
schema {
    version: 0.1.0
}

primitive_task "P" {
    operator: NoopOp
}
"#;
    let domain = cdda_htn::parse_htn(src).expect("parse");
    let registry = treg();
    let mut planner = cdda_htn::back_planner::BackPlanner::new(&domain, &registry);
    let err = planner.plan("MissingGoal", &State::default()).unwrap_err();
    assert!(matches!(err, HtnError::UnknownTask { .. }));
}

// ---------------------------------------------------------------------------
// BackPlanner greedy tie-breaking: a leaf covering multiple goal fields wins.
// ---------------------------------------------------------------------------

#[test]
fn backward_plan_prefers_multi_field_leaf() {
    let src = r#"
schema {
    version: 0.1.0
}

primitive_task "SingleShot" {
    operator: NoopOp
    effects: [flag = true]
}

primitive_task "DoubleShot" {
    operator: NoopOp
    effects: [flag = true, count = 7]
}

goal_task "Both" {
    effects: [flag = true, count = 7]
}
"#;
    let registry = treg();
    let domain = cdda_htn::parse_htn(src).expect("parse");
    let mut planner = cdda_htn::back_planner::BackPlanner::new(&domain, &registry);
    // Greedy heuristic: DoubleShot covers 2 of 2 needed fields, SingleShot only
    // 1 -> DoubleShot is selected first (and satisfies the whole goal).
    let plan = planner.plan("Both", &State::default()).expect("back plan");
    let names = plan.task_names();
    // The greedy heuristic prefers the max-coverage leaf first: DoubleShot
    // covers 2 of 2 needed fields (flag + count), SingleShot only 1.
    assert!(!names.is_empty());
    assert_eq!(names[0], "DoubleShot");
}

#[test]
fn back_plan_combines_leaves_for_distinct_fields() {
    let src = r#"
schema {
    version: 0.1.0
}

primitive_task "SetFlag" {
    operator: NoopOp
    effects: [flag = true]
}

primitive_task "SetCount" {
    operator: NoopOp
    effects: [count = 1]
}

goal_task "Both" {
    effects: [flag = true, count = 1]
}
"#;
    let registry = treg();
    let domain = cdda_htn::parse_htn(src).expect("parse");
    let mut planner = cdda_htn::back_planner::BackPlanner::new(&domain, &registry);
    // No single leaf covers both fields -> both are chained.
    let plan = planner.plan("Both", &State::default()).expect("back plan");
    let names = plan.task_names();
    assert_eq!(names.len(), 2);
    assert_ne!(names[0], names[1]);
}

// ---------------------------------------------------------------------------
// SetIdentifier verify round-trip (source field must exist on the state).
// ---------------------------------------------------------------------------

#[test]
fn set_identifier_verify_checks_source_field() {
    let state = State::default();
    // Both target and source exist -> ok.
    let ok_effect = Effect::SetIdentifier {
        field: "left".into(),
        field_source: "count".into(),
    };
    assert!(ok_effect.verify(state.as_reflect(), &treg()).is_ok());

    // Missing source field -> HtnError::Effect.
    let bad_effect = Effect::SetIdentifier {
        field: "left".into(),
        field_source: "does_not_exist".into(),
    };
    assert!(matches!(
        bad_effect.verify(state.as_reflect(), &treg()),
        Err(HtnError::Effect { .. })
    ));
}
