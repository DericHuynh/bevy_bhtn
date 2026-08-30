//! Nested-plan and replanning tests for `cdda_htn`.
//!
//! The planner is stateless — `HtnPlanner::plan` clones the state and returns a
//! fresh [`Plan`]. That makes "changing requirements mid-execution" a matter of
//! re-planning against an updated world state each turn, which is exactly how a
//! turn-based game (and the reference `bevy_htn` "ReplanRequest" flow) consumes
//! it. These tests pin that behaviour: deep nesting decomposes in the right
//! order, and a plan adapts when the world changes between turns.

mod common;
use common::HtnTestBed;

use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_reflect::TypeRegistry;
use cdda_htn::Task;

// ---------------------------------------------------------------------------
// A three-level nested domain: root -> middle -> leaves.
// ---------------------------------------------------------------------------

#[derive(Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Default)]
enum Room {
    #[default]
    Outside,
    Hall,
    Vault,
}

#[derive(Reflect, Clone, Debug, Default)]
struct NestedState {
    room: Room,
    torch_lit: bool,
    chest_open: bool,
}

fn register_nested(registry: &mut TypeRegistry) {
    registry.register::<NestedState>();
    registry.register::<Room>();
}

const NESTED_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "GetTreasure" {
    method "go" {
        subtasks: [Traverse, OpenChest]
    }
}

compound_task "Traverse" {
    method "light hall" {
        subtasks: [EnterHall, LightTorch]
    }
}

primitive_task "EnterHall" {
    operator: NoopOperator
    preconditions: [room == Room::Outside]
    effects: [room = Room::Hall]
}

primitive_task "LightTorch" {
    operator: NoopOperator
    effects: [torch_lit = true]
}

primitive_task "OpenChest" {
    operator: NoopOperator
    preconditions: [torch_lit == true, room != Room::Outside]
    effects: [chest_open = true]
}
"#;

#[test]
fn deep_nested_plan_decomposes_in_order() {
    let bed = HtnTestBed::new(NESTED_HTN, "GetTreasure", register_nested);
    let plan = bed.plan_forward(&NestedState::default());
    // Level 1 (GetTreasure) -> Level 2 (Entrance) -> leaves, in order.
    assert_eq!(plan, vec!["EnterHall", "LightTorch", "OpenChest"]);
}

// ---------------------------------------------------------------------------
// Mid-execution replanning: the world changes between turns.
// ---------------------------------------------------------------------------

const SURVIVAL_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Survive" {
    method "cold and unsheltered" {
        preconditions: [temperature < 15, sheltered == false]
        subtasks: [SeekShelter, WarmUp]
    }
    method "cold but sheltered" {
        preconditions: [temperature < 15, sheltered == true]
        subtasks: [WarmUp]
    }
    method "warm enough" {
        preconditions: [temperature >= 15]
        subtasks: [Scavenge]
    }
    method "fallback" {
        subtasks: [Wait]
    }
}

primitive_task "SeekShelter" {
    operator: NoopOperator
    preconditions: [sheltered == false]
    effects: [sheltered = true]
}

primitive_task "WarmUp" {
    operator: NoopOperator
    preconditions: [sheltered == true]
    effects: [temperature += 20]
}

primitive_task "Scavenge" {
    operator: NoopOperator
    preconditions: [temperature >= 15]
    effects: [supplies += 1]
}

primitive_task "Wait" {
    operator: NoopOperator
}
"#;

#[derive(Reflect, Clone, Debug, Default)]
struct SurvivalState {
    temperature: i32,
    sheltered: bool,
    supplies: i32,
}

fn register_survival(registry: &mut TypeRegistry) {
    registry.register::<SurvivalState>();
}

#[test]
fn plan_adapts_when_world_changes_mid_execution() {
    let bed = HtnTestBed::new(SURVIVAL_HTN, "Survive", register_survival);

    // Turn 1: it's cold — plan to take shelter then warm up.
    let mut state = SurvivalState {
        temperature: 5,
        ..Default::default()
    };
    let plan_1 = bed.plan_forward(&state);
    assert_eq!(plan_1, vec!["SeekShelter", "WarmUp"]);

    // Execute the shelter step (the world changes).
    apply_effects(&bed, &plan_1[0], &mut state);

    // Turn 2: still cold (warm-up not yet done) — replan should still aim to
    // warm up, now shelter is secured.
    let plan_2 = bed.plan_forward(&state);
    assert_eq!(plan_2, vec!["WarmUp"]);

    // Turn 3: warm up executes, temperature rises above the threshold.
    apply_effects(&bed, &plan_2[0], &mut state);
    assert!(state.temperature >= 15);

    // Turn 4: requirement changed — now it's warm, so switch to scavenging.
    let plan_3 = bed.plan_forward(&state);
    assert_eq!(plan_3, vec!["Scavenge"]);
}

#[test]
fn changing_requirements_force_a_new_root_branch() {
    let bed = HtnTestBed::new(SURVIVAL_HTN, "Survive", register_survival);
    // Start warm -> scavenge immediately.
    let warm = SurvivalState {
        temperature: 20,
        ..Default::default()
    };
    // A requirement *becomes* unmet: weather drops mid-execution.
    let mut state = warm.clone();
    let plan_warm = bed.plan_forward(&state);
    assert_eq!(plan_warm, vec!["Scavenge"]);

    // Requirements change (cold snap) mid-plan -> replan to shelter.
    state.temperature = 5;
    state.sheltered = false;
    let plan_cold = bed.plan_forward(&state);
    assert_eq!(plan_cold, vec!["SeekShelter", "WarmUp"]);
}

/// Apply a single planned primitive's effects to `state` (simulating executing
/// that step of the plan).
fn apply_effects<S: Reflect + Default + Clone + std::fmt::Debug>(
    bed: &HtnTestBed,
    task_name: &str,
    state: &mut S,
) {
    let Some(Task::Primitive(p)) = bed.domain().get_task(task_name) else {
        panic!("task `{task_name}` not a primitive");
    };
    for e in p.effects.iter() {
        e.apply(state.as_reflect_mut(), bed.registry());
    }
}
