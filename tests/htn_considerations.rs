//! Pins for the multi-axis utility layer (`bevy_bhtn::considerations`):
//! per-axis evaluate semantics (clamp → curve → invert → weight), the
//! blend combinators' weighted average, and end-to-end branch selection
//! through [`HighestUtility`](bevy_bhtn::domain::SelectionPolicy) with
//! blended scores.

use bevy_bhtn::considerations::{blend2, blend3, curves, Consideration};
use bevy_bhtn::planner::{HtnPlanner, PlanStatus};
use bevy_bhtn::prelude::*;
use bevy_ecs::prelude::Component;

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Hunger(pub f32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Morale(pub f32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Fatigue(pub f32);

/// Evaluate order is clamp → curve → invert → weight, and the scorer is
/// clamped to 0–1 before the curve regardless of what it returns.
#[test]
fn consideration_evaluate_order_and_clamping() {
    // Raw 2.0 clamps to 1.0 before the curve.
    let c = Consideration::new(|_: &Hunger| 2.0);
    assert_eq!(c.evaluate(&Hunger(0.0)), 1.0);

    // Curve applies after the clamp: quadratic(0.5) = 0.25.
    let curved = Consideration::new(|h: &Hunger| h.0 / 100.0).curve(curves::quadratic);
    assert_eq!(curved.evaluate(&Hunger(50.0)), 0.25);

    // Invert applies after the curve: 1 - quadratic(0.5) = 0.75.
    let inverted = curved.clone().invert();
    assert_eq!(inverted.evaluate(&Hunger(50.0)), 0.75);

    // Weight scales last.
    let weighted = curved.weight(2.0);
    assert_eq!(weighted.evaluate(&Hunger(50.0)), 0.5);
}

/// Named curves produce their documented shapes.
#[test]
fn curves_have_their_documented_shapes() {
    assert_eq!(curves::linear(0.5), 0.5);
    assert_eq!(curves::quadratic(0.5), 0.25);
    assert_eq!(curves::smoothstep(0.5), 0.5);
    assert_eq!(curves::ease_out(0.5), 0.75);
    // Endpoints are stable for every curve.
    for f in [
        curves::linear,
        curves::quadratic,
        curves::smoothstep,
        curves::ease_out,
    ] {
        assert_eq!(f(0.0), 0.0);
        assert_eq!(f(1.0), 1.0);
    }
}

/// A blend is the weighted average of its considerations — 0–1 scale
/// preserved — and an all-zero weight set degenerates to 0 instead of NaN.
#[test]
fn blend_is_a_weighted_average() {
    let hunger = Consideration::new(|h: &Hunger| h.0 / 100.0).weight(3.0);
    let morale = Consideration::new(|m: &Morale| m.0).weight(1.0);
    let blend = blend2(hunger, morale);
    // (3 * 0.8 + 1 * 0.2) / 4 = 0.65
    assert!((blend(&Hunger(80.0), &Morale(0.2)) - 0.65).abs() < 1e-5);

    let zero_weights = blend2(
        Consideration::<Hunger>::new(|_| 1.0).weight(0.0),
        Consideration::<Morale>::new(|_| 1.0).weight(0.0),
    );
    assert_eq!(zero_weights(&Hunger(1.0), &Morale(1.0)), 0.0);

    // blend3 keeps the same contract across three axes.
    let b3 = blend3(
        Consideration::<Hunger>::new(|h: &Hunger| h.0 / 100.0),
        Consideration::<Morale>::new(|m: &Morale| m.0),
        Consideration::<Fatigue>::new(|f: &Fatigue| f.0),
    );
    assert!((b3(&Hunger(100.0), &Morale(0.5), &Fatigue(0.0)) - 0.5).abs() < 1e-5);
}

/// End-to-end: blended considerations drive `HighestUtility` selection, and
/// the chosen branch flips as the underlying components change — with the
/// blend's score always the weighted average of its axes.
#[test]
fn blended_utility_selects_branches_by_state() {
    fn eat(task: &mut TaskBuilder) {
        task.effect(|h: &mut Hunger| h.0 -= 1.0);
    }
    fn patrol(task: &mut TaskBuilder) {
        task.effect(|m: &mut Morale| m.0 += 0.1);
    }
    fn decide(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::HighestUtility);
        let hunger = Consideration::new(|h: &Hunger| (h.0 / 100.0).clamp(0.0, 1.0))
            .curve(curves::quadratic)
            .weight(2.0);
        let morale = Consideration::new(|m: &Morale| m.0).invert();
        task.branch()
            .named("eat")
            .utility_fn(blend2(hunger, morale))
            .then(eat);
        // The fixed-utility competitor sits between the blend's two states.
        task.branch().named("patrol").utility(0.5).then(patrol);
    }

    let domain = HtnDomain::from_root(decide).build().unwrap();

    // Low hunger (quadratic(0.1)² ≈ 0.01), mid morale (inverted 0.5):
    // blend = (2*0.01 + 0.5) / 3 ≈ 0.173 < 0.5 → patrol.
    let state = PlanState::build(&domain.components)
        .set(Hunger(10.0))
        .set(Morale(0.5))
        .finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(decide, &state).expect("plan");
    assert_eq!(plan.status(), PlanStatus::Complete);
    assert_eq!(plan.task_names(&domain), ["patrol"]);

    // High hunger (1.0), low morale (inverted 1.0): blend = (2*1 + 1) / 3
    // ≈ 1.0 > 0.5 → eat. Same domain, same code — only the components moved.
    let state = PlanState::build(&domain.components)
        .set(Hunger(100.0))
        .set(Morale(0.0))
        .finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan(decide, &state).expect("plan");
    assert_eq!(plan.task_names(&domain), &["eat"]);
}

