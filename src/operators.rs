//! Strongly-typed HTN operators, resolved through reflection.
//!
//! An operator is the leaf "action" of a primitive task. In the planner it flows
//! as a stable string id plus parameter names (pulled from the plan state by
//! reflection). At execution time a Rust type implements [`HtnOperator`] and is
//! registered in the `bevy_reflect::TypeRegistry`, so a task's `operator: Foo`
//! line resolves to the registered `Foo` type, initialised from matching state
//! fields, and executed.
//!
//! This keeps operators **data** in the `.htn` file and **code** only at
//! execution, so domains stay moddable without editing Rust.

use bevy_reflect::{std_traits::ReflectDefault, FromType, Reflect, TypeRegistry};
use ustr::Ustr;

use crate::error::{HtnError, HtnResult};

/// An operator reference, as written in the `.htn` file.
///
/// Named operators are resolved against the type registry at execution time.
/// The parameter list names state fields (or literals) used to initialise the
/// operator's reflected value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operator {
    /// The operator type's name (must match a registered `Reflect` type).
    pub name: Ustr,
    /// State field names used to initialise the operator's data.
    pub params: Vec<Ustr>,
}

impl Operator {
    /// The operator's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The operator's parameter names.
    pub fn params(&self) -> &[Ustr] {
        &self.params
    }
}

/// A plan-step execution result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorStatus {
    /// Operator completed and the plan advances.
    Success,
    /// Operator is still executing across subsequent frames/turns.
    Ongoing,
    /// Operator failed; the plan should abort or backtrack.
    Failure,
}

/// A rust-side, strongly-typed operator implementation.
///
/// Implementors `#[derive(Reflect)]` (optionally `#[reflect(Default)]`), call
/// `registry_mut().register::<Self>()` (or `app.register_type::<Self>()`), and
/// implement this trait. The method runs at *execution* time.
pub trait HtnOperator: Reflect + Default + Clone + std::fmt::Debug {
    /// Execute one step. Return [`OperatorStatus`] to drive the plan forward.
    fn run(&self, _ctx: &dyn Reflect) -> OperatorStatus {
        OperatorStatus::Success
    }
}

/// Reflection type-data that lets a registered `Box<dyn Reflect>` operator be
/// executed after downcasting to its concrete type.
#[derive(Clone)]
pub struct ReflectHtnOperator(ReflectHtnOperatorFns);

/// The raw function pointers needed to make up a [`ReflectHtnOperator`].
#[derive(Clone)]
pub struct ReflectHtnOperatorFns {
    /// Type-erased [`HtnOperator::run`].
    run: fn(&dyn Reflect) -> OperatorStatus,
}

impl ReflectHtnOperator {
    /// Create custom operator reflection data from a type-erased `run` fn.
    pub fn new(run: fn(&dyn Reflect) -> OperatorStatus) -> Self {
        Self(ReflectHtnOperatorFns { run })
    }

    /// Run the operator (type-erased).
    pub fn run(&self, op: &dyn Reflect) -> OperatorStatus {
        (self.0.run)(op)
    }

    /// The underlying function pointers.
    pub fn fn_pointers(&self) -> &ReflectHtnOperatorFns {
        &self.0
    }
}

impl ReflectHtnOperatorFns {
    /// Build the default fns for a concrete `HtnOperator`.
    pub fn new<T: HtnOperator>() -> Self {
        <ReflectHtnOperator as FromType<T>>::from_type().0
    }
}

impl<E: HtnOperator> FromType<E> for ReflectHtnOperator {
    fn from_type() -> Self {
        ReflectHtnOperator::new(|op| {
            let typed = op
                .downcast_ref::<E>()
                .expect("ReflectHtnOperator executed on the wrong type");
            typed.run(op)
        })
    }
}

/// Verify a named operator exists in the type registry with the reflection data
/// needed to execute it (Default + ReflectHtnOperator). Call after parsing a
/// domain, before running it.
pub fn verify_operator(registry: &TypeRegistry, name: &str, params: &[Ustr]) -> HtnResult<()> {
    let Some(reg) = registry
        .get_with_short_type_path(name)
        .or_else(|| registry.get_with_type_path(name))
    else {
        return Err(HtnError::Operator {
            name: name.to_string(),
            params: params.iter().map(ToString::to_string).collect(),
            details: "No type registry entry. Call `app.register_type::<T>()` (or the
            registry equivalent) for this operator."
                .into(),
        });
    };
    if reg.data::<ReflectDefault>().is_none() {
        return Err(HtnError::Operator {
            name: name.to_string(),
            params: params.iter().map(ToString::to_string).collect(),
            details: "Missing ReflectDefault. Add `#[reflect(Default)]` to the operator type."
                .into(),
        });
    }
    if reg.data::<ReflectHtnOperator>().is_none() {
        return Err(HtnError::Operator {
            name: name.to_string(),
            params: params.iter().map(ToString::to_string).collect(),
            details: "Missing ReflectHtnOperator. Implement `HtnOperator` and register it.".into(),
        });
    }
    Ok(())
}

/// Construct a `Box<dyn Reflect>` operator from a registered type, initialising
/// its fields from matching plan-state fields (by name).
///
/// If any param has no matching state field or operator field, that param is
/// skipped (matching the reference crate's permissive behaviour).
pub fn construct_operator<S>(
    operator: &Operator,
    state: &S,
    registry: &TypeRegistry,
) -> HtnResult<Box<dyn Reflect>>
where
    S: Reflect,
{
    let Some(reg) = registry
        .get_with_short_type_path(operator.name())
        .or_else(|| registry.get_with_type_path(operator.name()))
    else {
        return Err(HtnError::Operator {
            name: operator.name().to_string(),
            params: operator.params().iter().map(ToString::to_string).collect(),
            details: "No type registry entry for operator".into(),
        });
    };
    let Some(reflect_default) = reg.data::<ReflectDefault>() else {
        return Err(HtnError::Operator {
            name: operator.name().to_string(),
            params: operator.params().iter().map(ToString::to_string).collect(),
            details: "Missing ReflectDefault".into(),
        });
    };
    let mut boxed: Box<dyn Reflect> = reflect_default.default();

    let Ok(state_struct) = state.reflect_ref().as_struct() else {
        return Ok(boxed);
    };

    for param in operator.params() {
        let Some(state_value) = state_struct.field(param) else {
            continue;
        };
        // Copy the state value into the operator via reflection.
        if let Ok(op_struct) = boxed.reflect_mut().as_struct() {
            if let Some(pr) = op_struct.field_mut(param) {
                pr.apply(state_value);
            }
        } else if let Ok(op_tuple) = boxed.reflect_mut().as_tuple_struct() {
            if let Some(pr) = op_tuple.field_mut(0) {
                pr.apply(state_value);
            }
        }
    }

    Ok(boxed)
}
