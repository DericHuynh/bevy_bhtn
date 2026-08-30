//! `.htn` DSL parser — build a domain from text (e.g. an imported asset).

use std::collections::HashMap;

use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use ustr::Ustr;

use crate::conditions::HtnCondition;
use crate::domain::HtnDomain;
use crate::effects::Effect;
use crate::error::{HtnError, HtnResult};
use crate::operators::Operator;
use crate::summaries::FieldSet;
use crate::tasks::{CompoundTask, GoalTask, Method, PrimitiveTask, Task};

#[derive(Parser)]
#[grammar = "src/htn.pest"]
pub struct HtnParser;

fn parse_i32(s: &str, ctx: &str) -> HtnResult<i32> {
    s.parse::<i32>().map_err(|_| HtnError::Condition {
        syntax: ctx.into(),
        details: format!("Invalid int `{s}`"),
    })
}
fn parse_f32(s: &str, ctx: &str) -> HtnResult<f32> {
    s.parse::<f32>().map_err(|_| HtnError::Effect {
        syntax: ctx.into(),
        details: format!("Invalid float `{s}`"),
    })
}
fn parse_bool(s: &str, ctx: &str) -> HtnResult<bool> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(HtnError::Condition {
            syntax: ctx.into(),
            details: format!("Invalid bool `{s}`"),
        }),
    }
}
fn parse_enum(s: &str, ctx: &str) -> HtnResult<(Ustr, Ustr)> {
    let mut parts = s.split("::");
    let ty = parts.next();
    let variant = parts.next();
    if parts.next().is_some() || ty.is_none() || variant.is_none() {
        return Err(HtnError::Condition {
            syntax: ctx.into(),
            details: format!("Invalid enum `{s}` — expected `Enum::Variant`"),
        });
    }
    Ok((ty.unwrap().into(), variant.unwrap().into()))
}

fn notted(op: Rule) -> bool {
    matches!(op, Rule::op_neq)
}

fn parse_condition(pair: Pair<Rule>) -> HtnResult<HtnCondition> {
    let syntax = pair.as_str().to_string();
    let mut it = pair.into_inner();
    let field: Ustr = it.next().unwrap().as_str().into();
    let op = it.next().unwrap().as_rule();
    let value = it.next().unwrap();
    let val_rule = value.as_rule();
    let val_str = value.as_str();
    let notted = notted(op);

    let cond = match (op, val_rule) {
        (Rule::op_gte | Rule::op_gt, Rule::int_value) => HtnCondition::GreaterThanInt {
            field,
            threshold: parse_i32(val_str, &syntax)?,
            orequals: op == Rule::op_gte,
        },
        (Rule::op_lte | Rule::op_lt, Rule::int_value) => HtnCondition::LessThanInt {
            field,
            threshold: parse_i32(val_str, &syntax)?,
            orequals: op == Rule::op_lte,
        },
        (Rule::op_gte | Rule::op_gt, Rule::float_value) => HtnCondition::GreaterThanFloat {
            field,
            threshold: parse_f32(val_str, &syntax)?,
            orequals: op == Rule::op_gte,
        },
        (Rule::op_lte | Rule::op_lt, Rule::float_value) => HtnCondition::LessThanFloat {
            field,
            threshold: parse_f32(val_str, &syntax)?,
            orequals: op == Rule::op_lte,
        },
        (Rule::op_gte | Rule::op_gt, Rule::identifier) => HtnCondition::GreaterThanIdentifier {
            field,
            other_field: val_str.into(),
            orequals: op == Rule::op_gte,
        },
        (Rule::op_lte | Rule::op_lt, Rule::identifier) => HtnCondition::LessThanIdentifier {
            field,
            other_field: val_str.into(),
            orequals: op == Rule::op_lte,
        },
        (Rule::op_eq | Rule::op_neq, Rule::bool_value) => HtnCondition::EqualsBool {
            field,
            value: parse_bool(val_str, &syntax)?,
            notted,
        },
        (Rule::op_eq | Rule::op_neq, Rule::none_value) => {
            HtnCondition::EqualsNone { field, notted }
        }
        (Rule::op_eq | Rule::op_neq, Rule::int_value) => HtnCondition::EqualsInt {
            field,
            value: parse_i32(val_str, &syntax)?,
            notted,
        },
        (Rule::op_eq | Rule::op_neq, Rule::float_value) => HtnCondition::EqualsFloat {
            field,
            value: parse_f32(val_str, &syntax)?,
            notted,
        },
        (Rule::op_eq | Rule::op_neq, Rule::enum_value) => {
            let (enum_type, enum_variant) = parse_enum(val_str, &syntax)?;
            HtnCondition::EqualsEnum {
                field,
                enum_type,
                enum_variant,
                notted,
            }
        }
        (Rule::op_eq | Rule::op_neq, Rule::identifier) => HtnCondition::EqualsIdentifier {
            field,
            other_field: val_str.into(),
            notted,
        },
        _ => {
            return Err(HtnError::Condition {
                syntax,
                details: "Unsupported condition".into(),
            })
        }
    };
    Ok(cond)
}

fn parse_effect(pair: Pair<Rule>) -> HtnResult<Effect> {
    let syntax = pair.as_str().to_string();
    let effect_pair = pair.into_inner().next().unwrap();
    let effect_rule = effect_pair.as_rule();
    let mut parts = effect_pair.into_inner();
    let field: Ustr = parts.next().unwrap().as_str().into();
    let val_pair = parts.next().unwrap();
    let val_rule = val_pair.as_rule();
    let val_str = val_pair.as_str();

    let effect = match (effect_rule, val_rule) {
        (Rule::set_effect_literal, Rule::bool_value) => Effect::SetBool {
            field,
            value: parse_bool(val_str, &syntax)?,
        },
        (Rule::set_effect_literal, Rule::int_value) => Effect::SetInt {
            field,
            value: parse_i32(val_str, &syntax)?,
        },
        (Rule::set_effect_literal, Rule::float_value) => Effect::SetFloat {
            field,
            value: parse_f32(val_str, &syntax)?,
        },
        (Rule::set_effect_literal, Rule::enum_value) => {
            let (enum_type, enum_variant) = parse_enum(val_str, &syntax)?;
            Effect::SetEnum {
                field,
                enum_type,
                enum_variant,
            }
        }
        (Rule::set_effect_literal, Rule::none_value) => Effect::SetNone { field },
        (Rule::set_effect_identifier, Rule::identifier) => Effect::SetIdentifier {
            field,
            field_source: val_str.into(),
        },
        (Rule::set_effect_inc_literal, Rule::int_value) => Effect::IncrementInt {
            field,
            by: parse_i32(val_str, &syntax)?,
        },
        (Rule::set_effect_dec_literal, Rule::int_value) => Effect::IncrementInt {
            field,
            by: -parse_i32(val_str, &syntax)?,
        },
        (Rule::set_effect_inc_literal, Rule::float_value) => Effect::IncrementFloat {
            field,
            by: parse_f32(val_str, &syntax)?,
        },
        (Rule::set_effect_dec_literal, Rule::float_value) => Effect::IncrementFloat {
            field,
            by: -parse_f32(val_str, &syntax)?,
        },
        (Rule::set_effect_inc_identifier, Rule::identifier)
        | (Rule::set_effect_dec_identifier, Rule::identifier) => {
            // `+= identifier` copies the source field's value into the target.
            // (Addition by reference isn't expressible as a single operation, so
            // mirror the reference crate: treat it as an identifier copy.)
            Effect::SetIdentifier {
                field,
                field_source: val_str.into(),
            }
        }
        _ => {
            return Err(HtnError::Effect {
                syntax,
                details: "Unsupported effect".into(),
            })
        }
    };
    Ok(effect)
}

fn parse_primitive(pair: Pair<Rule>) -> HtnResult<PrimitiveTask> {
    let mut inner = pair.into_inner();
    let name: Ustr = inner.next().unwrap().as_str().trim_matches('"').into();
    let mut operator = None;
    let mut preconditions = Vec::new();
    let mut effects = Vec::new();
    let mut expected_effects = Vec::new();

    for stmt in inner {
        match stmt.as_rule() {
            Rule::operator_statement => {
                let op_def = stmt.into_inner().next().unwrap();
                let mut op_parts = op_def.into_inner();
                let op_name = op_parts.next().unwrap();
                let params: Vec<Ustr> = op_parts.map(|p| p.as_str().into()).collect();
                operator = Some(Operator {
                    name: op_name.as_str().into(),
                    params,
                });
            }
            Rule::preconditions_statement => {
                for c in stmt.into_inner().filter(|p| p.as_rule() == Rule::condition) {
                    preconditions.push(parse_condition(c)?);
                }
            }
            Rule::effects_statement => {
                for e in stmt.into_inner().filter(|p| p.as_rule() == Rule::effect) {
                    effects.push(parse_effect(e)?);
                }
            }
            Rule::expected_effects_statement => {
                for e in stmt.into_inner().filter(|p| p.as_rule() == Rule::effect) {
                    expected_effects.push(parse_effect(e)?);
                }
            }
            _ => {}
        }
    }

    Ok(PrimitiveTask {
        name,
        operator: operator.expect("primitive task requires an operator"),
        preconditions,
        effects,
        expected_effects,
        prec_reads: Vec::new(),
    })
}

fn parse_method(pair: Pair<Rule>) -> HtnResult<Method> {
    let mut inner = pair.into_inner().peekable();
    let mut name = None;
    if let Some(p) = inner.peek() {
        if p.as_rule() == Rule::STRING {
            name = Some(inner.next().unwrap().as_str().trim_matches('"').into());
        }
    }
    let mut preconditions = Vec::new();
    let mut subtasks = Vec::new();
    for stmt in inner {
        match stmt.as_rule() {
            Rule::preconditions_statement => {
                for c in stmt.into_inner().filter(|p| p.as_rule() == Rule::condition) {
                    preconditions.push(parse_condition(c)?);
                }
            }
            Rule::subtasks_statement => {
                for t in stmt
                    .into_inner()
                    .filter(|p| p.as_rule() == Rule::identifier)
                {
                    subtasks.push(t.as_str().into());
                }
            }
            _ => {}
        }
    }
    Ok(Method {
        name,
        preconditions,
        subtasks,
        possible_writes: FieldSet::default(),
        prec_reads: Vec::new(),
    })
}

fn parse_compound(pair: Pair<Rule>) -> HtnResult<CompoundTask> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().trim_matches('"').into();
    let mut methods = Vec::new();
    for m in inner.filter(|p| p.as_rule() == Rule::method) {
        methods.push(parse_method(m)?);
    }
    Ok(CompoundTask { name, methods })
}

fn parse_goal(pair: Pair<Rule>) -> HtnResult<GoalTask> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().trim_matches('"').into();
    let mut effects = Vec::new();
    for stmt in inner {
        if stmt.as_rule() == Rule::effects_statement {
            for e in stmt.into_inner().filter(|p| p.as_rule() == Rule::effect) {
                effects.push(parse_effect(e)?);
            }
        }
    }
    Ok(GoalTask { name, effects })
}

fn parse_schema(pair: Pair<Rule>) -> HtnResult<String> {
    let mut inner = pair.into_inner();
    let ver = inner.next().unwrap();
    if ver.as_rule() == Rule::schema_version_statement {
        Ok(ver.into_inner().next().unwrap().as_str().to_string())
    } else {
        Err(HtnError::schema(ver.as_str()))
    }
}

/// Parse an `.htn` domain from `input`. Task declaration order matters: the
/// first compound task is the default root for forward planning.
pub fn parse_htn(input: &str) -> HtnResult<HtnDomain> {
    let pairs =
        HtnParser::parse(Rule::domain, input).map_err(|e| HtnError::parser(e.to_string()))?;
    let root = pairs.into_iter().next().expect("domain rule matches");
    let mut schema = None;
    let mut tasks = Vec::new();

    for pair in root.into_inner() {
        match pair.as_rule() {
            Rule::schema => schema = Some(parse_schema(pair)?),
            Rule::primitive_task => tasks.push(Task::Primitive(parse_primitive(pair)?)),
            Rule::compound_task => tasks.push(Task::Compound(parse_compound(pair)?)),
            Rule::goal_task => tasks.push(Task::Goal(parse_goal(pair)?)),
            _ => {}
        }
    }
    let mut domain = HtnDomain {
        schema: schema.unwrap_or_else(|| "0.0.0".into()),
        tasks,
        index_of: HashMap::new(),
        fields: Vec::new(),
        field_index: HashMap::new(),
        summaries: Vec::new(),
    };
    domain.rebuild_index();
    Ok(domain)
}
