//! [`Effect`] — a reflection-applied state mutation, written in `.htn`.
//!
//! Effects are applied to the plan state during forward planning (as
//! "anticipated" effects) and at execution time. `goal_task` bodies use effects
//! to declare the desired end state for **back-planning**.
//!
//! All application happens through `bevy_reflect` 0.18 APIs: we coerce the state
//! to `&mut dyn Struct`, then mutate fields via `downcast_mut` (primitives),
//! `Enum::set_variant` (unit enum variants / `None`), or clone-and-`apply`
//! (identifier copies).

use bevy_reflect::{DynamicVariant, PartialReflect, Reflect, ReflectFromReflect, TypeRegistry};
use ustr::Ustr;

use crate::error::{HtnError, HtnResult};

/// A write to a state field. Applied to the plan state during forward planning
/// (as an "anticipated" effect) and at execution time. `goal_task` bodies use
/// effects to declare the desired end state for **back-planning**.
///
/// Field names are interned [`Ustr`]s, so constructing/comparing an effect is a
/// single pointer compare and effect lists stay compact.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// Set a boolean field to a literal.
    SetBool {
        /// The state field to write.
        field: Ustr,
        /// The boolean value to assign.
        value: bool,
    },
    /// Set an integer field to a literal.
    SetInt {
        /// The state field to write.
        field: Ustr,
        /// The integer value to assign.
        value: i32,
    },
    /// Set a float field to a literal.
    SetFloat {
        /// The state field to write.
        field: Ustr,
        /// The float value to assign.
        value: f32,
    },
    /// Switch an enum field to the named unit variant.
    SetEnum {
        /// The state field to write.
        field: Ustr,
        /// The enum type's name (used only for diagnostics/verification).
        enum_type: Ustr,
        /// The unit variant to switch to.
        enum_variant: Ustr,
    },
    /// Set an `Option` field to `None`.
    SetNone {
        /// The state field to write.
        field: Ustr,
    },
    /// Copy one field's value into another.
    SetIdentifier {
        /// The state field to write.
        field: Ustr,
        /// The source field whose value is copied.
        field_source: Ustr,
    },
    /// Add a constant to an integer field.
    IncrementInt {
        /// The state field to write.
        field: Ustr,
        /// Amount to add (negative for subtraction).
        by: i32,
    },
    /// Add a constant to a float field.
    IncrementFloat {
        /// The state field to write.
        field: Ustr,
        /// Amount to add (negative for subtraction).
        by: f32,
    },
}

impl Effect {
    /// The `.htn` representation (for diagnostics).
    pub fn syntax(&self) -> String {
        match self {
            Effect::SetBool { field, value } => format!("{field} = {value}"),
            Effect::SetInt { field, value } => format!("{field} = {value}"),
            Effect::SetFloat { field, value } => format!("{field} = {value}"),
            Effect::SetEnum {
                field,
                enum_type,
                enum_variant,
            } => format!("{field} = {enum_type}::{enum_variant}"),
            Effect::SetNone { field } => format!("{field} = None"),
            Effect::SetIdentifier {
                field,
                field_source,
            } => format!("{field} = {field_source}"),
            Effect::IncrementInt { field, by } => format!("{field} += {by}"),
            Effect::IncrementFloat { field, by } => format!("{field} += {by}"),
        }
    }

    /// The field name this effect writes to.
    pub fn field(&self) -> &str {
        match self {
            Effect::SetBool { field, .. }
            | Effect::SetInt { field, .. }
            | Effect::SetFloat { field, .. }
            | Effect::SetEnum { field, .. }
            | Effect::SetNone { field }
            | Effect::SetIdentifier { field, .. }
            | Effect::IncrementInt { field, .. }
            | Effect::IncrementFloat { field, .. } => field,
        }
    }

    /// The optional second field this effect reads from, if any.
    pub fn source_field(&self) -> Option<&str> {
        match self {
            Effect::SetIdentifier { field_source, .. } => Some(field_source),
            _ => None,
        }
    }

    /// Verify that every referenced field exists on `state` and (for enum
    /// effects) that the referenced enum type is registered.
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        let Ok(s) = state.reflect_ref().as_struct() else {
            return Err(HtnError::Effect {
                syntax: self.syntax(),
                details: "Plan state must be a struct".into(),
            });
        };
        for name in [Some(self.field()), self.source_field()]
            .into_iter()
            .flatten()
        {
            if s.field(name).is_none() {
                return Err(HtnError::Effect {
                    syntax: self.syntax(),
                    details: format!("State has no field `{name}`"),
                });
            }
        }
        if let Effect::SetEnum { enum_type, .. } = self {
            let registered = registry
                .get_with_type_path(enum_type)
                .or_else(|| registry.get_with_short_type_path(enum_type))
                .is_some();
            if !registered {
                return Err(HtnError::Effect {
                    syntax: self.syntax(),
                    details: format!("Enum type `{enum_type}` is not registered"),
                });
            }
        }
        Ok(())
    }

    /// Apply this effect to `state` via reflection. `registry` is used to look
    /// up the `ReflectFromReflect` cloner for identifier copies.
    pub fn apply(&self, state: &mut dyn Reflect, registry: &TypeRegistry) {
        if let Ok(s) = state.reflect_mut().as_struct() {
            self.apply_to_struct(s, registry);
        }
    }

    /// Same as [`Effect::apply`] but for an erased `&mut dyn Reflect`.
    pub fn apply_dyn(&self, state: &mut dyn Reflect, registry: &TypeRegistry) {
        if let Ok(s) = state.reflect_mut().as_struct() {
            self.apply_to_struct(s, registry);
        }
    }

    /// The shared mutation routine used by both [`Effect::apply`] and
    /// [`Effect::apply_dyn`].
    fn apply_to_struct(&self, s: &mut dyn bevy_reflect::Struct, registry: &TypeRegistry) {
        match self {
            Effect::SetBool { field, value } => {
                if let Some(f) = s.field_mut(field) {
                    if let Some(v) = f.try_downcast_mut::<bool>() {
                        *v = *value;
                    }
                }
            }
            Effect::SetInt { field, value } => {
                if let Some(f) = s.field_mut(field) {
                    if let Some(v) = f.try_downcast_mut::<i32>() {
                        *v = *value;
                    }
                }
            }
            Effect::SetFloat { field, value } => {
                if let Some(f) = s.field_mut(field) {
                    if let Some(v) = f.try_downcast_mut::<f32>() {
                        *v = *value;
                    }
                }
            }
            Effect::SetEnum {
                field,
                enum_variant,
                ..
            } => {
                apply_unit_variant(s, field, enum_variant, registry);
            }
            Effect::SetNone { field } => apply_unit_variant(s, field, "None", registry),
            Effect::SetIdentifier {
                field,
                field_source,
            } => {
                let Some(src) = s.field(field_source) else {
                    return;
                };
                if let Some(cloned) = clone_value(src, registry) {
                    if let Some(f) = s.field_mut(field) {
                        f.apply(cloned.as_partial_reflect());
                    }
                }
            }
            Effect::IncrementInt { field, by } => {
                if let Some(f) = s.field_mut(field) {
                    if let Some(v) = f.try_downcast_mut::<i32>() {
                        *v += *by;
                    }
                }
            }
            Effect::IncrementFloat { field, by } => {
                if let Some(f) = s.field_mut(field) {
                    if let Some(v) = f.try_downcast_mut::<f32>() {
                        *v += *by;
                    }
                }
            }
        }
    }
}

/// Copy a reflected field into an owned `Box<dyn Reflect>`.
///
/// Prefers the registry's `ReflectFromReflect` data so registered custom types
/// keep their concrete form; falls back to `reflect_clone` otherwise.
fn clone_value(
    value: &dyn bevy_reflect::PartialReflect,
    registry: &TypeRegistry,
) -> Option<Box<dyn Reflect>> {
    let type_path = value.reflect_type_path();
    if let Some(reg) = registry.get_with_type_path(type_path) {
        if let Some(cloner) = reg.data::<ReflectFromReflect>() {
            if let Some(cloned) = cloner.from_reflect(value) {
                return Some(cloned);
            }
        }
    }
    value.reflect_clone().ok()
}

/// Switch a field to a unit enum variant by constructing a `DynamicEnum` and
/// applying it to the field.
///
/// The variant index (for ordering) is resolved from the field's reflected enum
/// info when available; this works for unit variants (including `None` on
/// `Option` fields).
fn apply_unit_variant(
    s: &mut dyn bevy_reflect::Struct,
    field: &str,
    variant: &str,
    registry: &TypeRegistry,
) {
    let Some(f) = s.field_mut(field) else {
        return;
    };
    // Determine the variant's index (used only for faithful round-tripping) via
    // the field's represented enum info, resolved through the registry.
    let mut dyn_enum = bevy_reflect::DynamicEnum::new(variant.to_string(), DynamicVariant::Unit);
    let index = registry
        .get_with_type_path(f.reflect_type_path())
        .and_then(|reg| match reg.type_info() {
            bevy_reflect::TypeInfo::Enum(info) => info.index_of(variant),
            _ => None,
        });
    if let Some(index) = index {
        dyn_enum.set_variant_with_index(index, variant.to_string(), DynamicVariant::Unit);
    }
    f.apply(dyn_enum.as_partial_reflect());
}
