//! [`HtnCondition`] — a reflection-evaluated predicate over a plan state struct.
//!
//! Conditions are read-only: they never mutate state. Every variant is evaluated
//! against a `&dyn PartialReflect` struct using Bevy 0.18 reflection APIs
//! (`field`, `try_downcast_ref`, `reflect_partial_eq`, `reflect_ref().as_enum()`).

use bevy_reflect::{Enum, Reflect, Struct, TypeRegistry};
use ustr::Ustr;

use crate::error::{HtnError, HtnResult};

/// A predicate over state fields, written in the `.htn` file and evaluated by
/// reflection against the plan state at plan time.
///
/// Field/enum names are interned [`Ustr`]s: comparing a condition's referenced
/// field to a known name is a single pointer compare.
#[derive(Debug, Clone, PartialEq)]
pub enum HtnCondition {
    /// Field equals a boolean literal.
    EqualsBool {
        /// The state field to read.
        field: Ustr,
        /// The boolean to compare against.
        value: bool,
        /// Whether the comparison is negated (`!=`).
        notted: bool,
    },
    /// Field equals an integer literal.
    EqualsInt {
        /// The state field to read.
        field: Ustr,
        /// The integer to compare against.
        value: i32,
        /// Whether the comparison is negated (`!=`).
        notted: bool,
    },
    /// Field equals a float literal.
    EqualsFloat {
        /// The state field to read.
        field: Ustr,
        /// The float to compare against.
        value: f32,
        /// Whether the comparison is negated (`!=`).
        notted: bool,
    },
    /// Field is (or is not) `None` — only valid on `Option` fields.
    EqualsNone {
        /// The state field to read.
        field: Ustr,
        /// Whether the comparison is negated (`!= None`, i.e. is `Some`).
        notted: bool,
    },
    /// Field equals an enum variant.
    EqualsEnum {
        /// The state field to read.
        field: Ustr,
        /// The enum type's name.
        enum_type: Ustr,
        /// The variant to compare against.
        enum_variant: Ustr,
        /// Whether the comparison is negated (`!=`).
        notted: bool,
    },
    /// Two fields hold equal values.
    EqualsIdentifier {
        /// The first state field to read.
        field: Ustr,
        /// The second state field to read.
        other_field: Ustr,
        /// Whether the comparison is negated (`!=`).
        notted: bool,
    },
    /// Integer field is greater than a literal (optionally including equality).
    GreaterThanInt {
        /// The state field to read.
        field: Ustr,
        /// The integer threshold.
        threshold: i32,
        /// `true` for `>=`, `false` for `>`.
        orequals: bool,
    },
    /// Float field is greater than a literal (optionally including equality).
    GreaterThanFloat {
        /// The state field to read.
        field: Ustr,
        /// The float threshold.
        threshold: f32,
        /// `true` for `>=`, `false` for `>`.
        orequals: bool,
    },
    /// Integer field is greater than another field's value.
    GreaterThanIdentifier {
        /// The state field to read.
        field: Ustr,
        /// The other field to compare against.
        other_field: Ustr,
        /// `true` for `>=`, `false` for `>`.
        orequals: bool,
    },
    /// Integer field is less than a literal (optionally including equality).
    LessThanInt {
        /// The state field to read.
        field: Ustr,
        /// The integer threshold.
        threshold: i32,
        /// `true` for `<=`, `false` for `<`.
        orequals: bool,
    },
    /// Float field is less than a literal (optionally including equality).
    LessThanFloat {
        /// The state field to read.
        field: Ustr,
        /// The float threshold.
        threshold: f32,
        /// `true` for `<=`, `false` for `<`.
        orequals: bool,
    },
    /// Integer field is less than another field's value.
    LessThanIdentifier {
        /// The state field to read.
        field: Ustr,
        /// The other field to compare against.
        other_field: Ustr,
        /// `true` for `<=`, `false` for `<`.
        orequals: bool,
    },
}

impl HtnCondition {
    /// The `.htn` representation (for diagnostics).
    pub fn syntax(&self) -> String {
        match self {
            HtnCondition::EqualsBool {
                field,
                value,
                notted,
            } => {
                format!("{field} {} {value}", eq(*notted))
            }
            HtnCondition::EqualsInt {
                field,
                value,
                notted,
            } => {
                format!("{field} {} {value}", eq(*notted))
            }
            HtnCondition::EqualsFloat {
                field,
                value,
                notted,
            } => format!("{field} {} {value}", eq(*notted)),
            HtnCondition::EqualsNone { field, notted } => {
                format!("{field} {} None", eq(*notted))
            }
            HtnCondition::EqualsEnum {
                field,
                enum_type,
                enum_variant,
                notted,
            } => format!("{field} {} {enum_type}::{enum_variant}", eq(*notted)),
            HtnCondition::EqualsIdentifier {
                field,
                other_field,
                notted,
            } => format!("{field} {} {other_field}", eq(*notted)),
            HtnCondition::GreaterThanInt {
                field, threshold, ..
            } => format!("{field} > {threshold}"),
            HtnCondition::GreaterThanFloat {
                field, threshold, ..
            } => format!("{field} > {threshold}"),
            HtnCondition::GreaterThanIdentifier {
                field, other_field, ..
            } => format!("{field} > {other_field}"),
            HtnCondition::LessThanInt {
                field, threshold, ..
            } => format!("{field} < {threshold}"),
            HtnCondition::LessThanFloat {
                field, threshold, ..
            } => format!("{field} < {threshold}"),
            HtnCondition::LessThanIdentifier {
                field, other_field, ..
            } => format!("{field} < {other_field}"),
        }
    }

    /// Verify the referenced fields exist on `state` (and, for enum conditions,
    /// that the enum type is registered).
    pub fn verify(&self, state: &dyn Reflect, registry: &TypeRegistry) -> HtnResult<()> {
        let Ok(s) = state.reflect_ref().as_struct() else {
            return Err(HtnError::Condition {
                syntax: self.syntax(),
                details: "Plan state must be a struct".into(),
            });
        };
        let fields = [Some(self.field()), self.other_field()]
            .into_iter()
            .flatten();
        for name in fields {
            if s.field(name).is_none() {
                return Err(HtnError::Condition {
                    syntax: self.syntax(),
                    details: format!("State has no field `{name}`"),
                });
            }
        }
        if let HtnCondition::EqualsEnum { enum_type, .. } = self {
            let registered = registry
                .get_with_type_path(enum_type)
                .or_else(|| registry.get_with_short_type_path(enum_type))
                .is_some();
            if !registered {
                return Err(HtnError::Condition {
                    syntax: self.syntax(),
                    details: format!("Enum type `{enum_type}` is not registered"),
                });
            }
        }
        Ok(())
    }

    /// Evaluate this condition against `state`. Assumes [`HtnCondition::verify`]
    /// has already passed so field access is safe to fall back to `false` on.
    pub fn evaluate(&self, state: &dyn Reflect) -> bool {
        let Ok(s) = state.reflect_ref().as_struct() else {
            return false;
        };
        self.predicate(s)
    }

    /// The primary field this condition reads.
    pub(crate) fn field(&self) -> &str {
        match self {
            HtnCondition::EqualsBool { field, .. }
            | HtnCondition::EqualsInt { field, .. }
            | HtnCondition::EqualsFloat { field, .. }
            | HtnCondition::EqualsNone { field, .. }
            | HtnCondition::EqualsEnum { field, .. }
            | HtnCondition::EqualsIdentifier { field, .. }
            | HtnCondition::GreaterThanInt { field, .. }
            | HtnCondition::GreaterThanFloat { field, .. }
            | HtnCondition::GreaterThanIdentifier { field, .. }
            | HtnCondition::LessThanInt { field, .. }
            | HtnCondition::LessThanFloat { field, .. }
            | HtnCondition::LessThanIdentifier { field, .. } => field,
        }
    }

    /// The optional second field this condition compares against (if any).
    pub(crate) fn other_field(&self) -> Option<&str> {
        match self {
            HtnCondition::EqualsIdentifier { other_field, .. }
            | HtnCondition::GreaterThanIdentifier { other_field, .. }
            | HtnCondition::LessThanIdentifier { other_field, .. } => Some(other_field),
            _ => None,
        }
    }

    /// Every field this condition reads (primary + comparison field).
    pub(crate) fn read_fields(&self) -> impl Iterator<Item = &str> {
        [Some(self.field()), self.other_field()]
            .into_iter()
            .flatten()
    }

    fn predicate(&self, s: &dyn Struct) -> bool {
        let Some(value) = s.field(self.field()) else {
            return false;
        };
        match self {
            HtnCondition::EqualsBool {
                value: want,
                notted,
                ..
            } => value.try_downcast_ref::<bool>().map(|v| v == want) == Some(!*notted),
            HtnCondition::EqualsInt {
                value: want,
                notted,
                ..
            } => value.try_downcast_ref::<i32>().map(|v| *v == *want) == Some(!*notted),
            HtnCondition::EqualsFloat {
                value: want,
                notted,
                ..
            } => value.try_downcast_ref::<f32>().map(|v| *v == *want) == Some(!*notted),
            HtnCondition::EqualsNone { notted, .. } => {
                let is_none = value
                    .reflect_ref() // we need to check if current variant name is "None"
                    .as_enum()
                    .map(|e| e.variant_name() == "None" && represents_option(e))
                    .unwrap_or(false);
                is_none == !*notted
            }
            HtnCondition::EqualsEnum {
                enum_variant,
                notted,
                ..
            } => {
                let matches = value
                    .reflect_ref()
                    .as_enum()
                    .map(|e| e.variant_name() == enum_variant)
                    .unwrap_or(false);
                matches != *notted
            }
            HtnCondition::EqualsIdentifier {
                other_field,
                notted,
                ..
            } => {
                let Some(other) = s.field(other_field) else {
                    return false;
                };
                let equal = value.reflect_partial_eq(other).unwrap_or(false);
                equal == !*notted
            }
            HtnCondition::GreaterThanInt {
                threshold,
                orequals,
                ..
            } => numeral_int_cmp(
                value.try_downcast_ref::<i32>().copied(),
                *threshold,
                *orequals,
                true,
            ),
            HtnCondition::GreaterThanFloat {
                threshold,
                orequals,
                ..
            } => numeral_f32_cmp(
                value.try_downcast_ref::<f32>().copied(),
                *threshold,
                *orequals,
                true,
            ),
            HtnCondition::GreaterThanIdentifier {
                other_field,
                orequals,
                ..
            } => {
                let Some(other) = s.field(other_field) else {
                    return false;
                };
                if let (Some(a), Some(b)) = (
                    value.try_downcast_ref::<i32>(),
                    other.try_downcast_ref::<i32>(),
                ) {
                    return cmp_int_pair(*a, *b, *orequals, true);
                }
                false
            }
            HtnCondition::LessThanInt {
                threshold,
                orequals,
                ..
            } => numeral_int_cmp(
                value.try_downcast_ref::<i32>().copied(),
                *threshold,
                *orequals,
                false,
            ),
            HtnCondition::LessThanFloat {
                threshold,
                orequals,
                ..
            } => numeral_f32_cmp(
                value.try_downcast_ref::<f32>().copied(),
                *threshold,
                *orequals,
                false,
            ),
            HtnCondition::LessThanIdentifier {
                other_field,
                orequals,
                ..
            } => {
                let Some(other) = s.field(other_field) else {
                    return false;
                };
                if let (Some(a), Some(b)) = (
                    value.try_downcast_ref::<i32>(),
                    other.try_downcast_ref::<i32>(),
                ) {
                    return cmp_int_pair(*a, *b, *orequals, false);
                }
                false
            }
        }
    }
}

/// Is this enum an `Option` (i.e. has just `None`/`Some` variants)?
fn represents_option(e: &dyn Enum) -> bool {
    let Some(info) = e.get_represented_enum_info() else {
        return false;
    };
    let names = info.variant_names();
    names.len() == 2 && names[0] == "None" && names[1] == "Some"
}

/// Integer field-vs-literal bucket comparison.
fn numeral_int_cmp(v: Option<i32>, threshold: i32, orequals: bool, greater: bool) -> bool {
    let Some(v) = v else {
        return false;
    };
    if greater {
        if orequals {
            v >= threshold
        } else {
            v > threshold
        }
    } else if orequals {
        v <= threshold
    } else {
        v < threshold
    }
}

/// Float field-vs-literal bucket comparison.
fn numeral_f32_cmp(v: Option<f32>, threshold: f32, orequals: bool, greater: bool) -> bool {
    let Some(v) = v else {
        return false;
    };
    if greater {
        if orequals {
            v >= threshold
        } else {
            v > threshold
        }
    } else if orequals {
        v <= threshold
    } else {
        v < threshold
    }
}

/// Integer field-vs-field bucket comparison.
fn cmp_int_pair(a: i32, b: i32, orequals: bool, greater: bool) -> bool {
    if greater {
        if orequals {
            a >= b
        } else {
            a > b
        }
    } else if orequals {
        a <= b
    } else {
        a < b
    }
}

fn eq(notted: bool) -> &'static str {
    if notted {
        "!="
    } else {
        "=="
    }
}
