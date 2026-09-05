//! Multi-axis utility scoring: [`Consideration`] scorers and [`blend2`]-style
//! combinators for [`HighestUtility`](crate::domain::SelectionPolicy) and
//! [`WeightedRandom`](crate::domain::SelectionPolicy) selection.
//!
//! A single hand-written `utility_fn` closure scores one opaque `f32` — fine
//! for simple branches, but emergent NPC decisions (hunger vs. morale vs.
//! fatigue) want per-axis "considerations": each factor normalized to 0–1,
//! shaped by a curve, weighted, and blended into one score with a shared
//! vocabulary (Infinite-Axis-Utility style). Each axis stays individually
//! inspectable — log `Consideration::evaluate` per axis instead of reverse-
//! engineering one magic scalar.
//!
//! ```
//! use bevy_bhtn::considerations::{Consideration, curves, blend2};
//! use bevy_bhtn::prelude::*;
//! use bevy_ecs::prelude::*;
//!
//! #[derive(Component, Clone, Default, Debug)]
//! struct Hunger(pub f32);
//! #[derive(Component, Clone, Default, Debug)]
//! struct Morale(pub f32);
//!
//! fn eat(task: &mut TaskBuilder) { task.branch(); }
//! fn patrol(task: &mut TaskBuilder) { task.branch(); }
//!
//! fn decide(task: &mut TaskBuilder) {
//!     // `eat` when hunger dominates; `patron` otherwise.
//!     let hunger = Consideration::new(|h: &Hunger| h.0 / 100.0)
//!         .curve(curves::quadratic)
//!         .weight(2.0);
//!     let morale = Consideration::new(|m: &Morale| m.0).invert();
//!     task.branch().named("eat")
//!         .utility_fn(blend2(hunger, morale))
//!         .then(eat);
//!     task.branch().named("patrol").utility(0.4).then(patrol);
//! }
//! # HtnDomain::from_root(decide).build().unwrap();
//! ```
//!
//! A blend's score is the **weighted average** of its considerations
//! (`Σ weightᵢ · valueᵢ / Σ weightᵢ`), so the result stays in 0–1 and the
//! [`C` exploration constant](crate::domain::SelectionPolicy) of any
//! downstream policy keeps a stable scale. Each consideration's `evaluate`
//! clamps its scorer to 0–1 *before* the curve and applies `invert` after it,
//! so `curve(quadratic).invert()` means "1 − x²", not "(1 − x)²".

use std::sync::Arc;

use crate::state::PlanComponent;

/// Response curves shaping a normalized 0–1 input. Plain `fn` pointers —
/// pick one per axis; write your own for exotic shapes.
pub mod curves {
    /// Identity: the input passes through unchanged.
    pub fn linear(x: f32) -> f32 {
        x
    }

    /// Quadratic ease-in: low values matter less, high values matter more.
    pub fn quadratic(x: f32) -> f32 {
        x * x
    }

    /// Smooth S-curve (3x² − 2x³): soft threshold around the midpoint.
    pub fn smoothstep(x: f32) -> f32 {
        x * x * (3.0 - 2.0 * x)
    }

    /// Inverse ease (1 − (1 − x)²): steep early, flattening late.
    pub fn ease_out(x: f32) -> f32 {
        1.0 - (1.0 - x) * (1.0 - x)
    }
}

/// One scoring axis over a single planning component: a normalized scorer
/// plus curve, weight, and inversion. Cheap to clone (`Arc`'d scorer) and
/// `Send + Sync`, so one consideration set can be shared across a population.
///
/// The scorer reads one component and returns its raw value; author it
/// normalized to 0–1 when you can — anything outside is clamped before the
/// curve, so the blend's scale stays stable regardless.
pub struct Consideration<A: PlanComponent> {
    score: Arc<dyn Fn(&A) -> f32 + Send + Sync>,
    curve: fn(f32) -> f32,
    weight: f32,
    invert: bool,
}

impl<A: PlanComponent> Clone for Consideration<A> {
    fn clone(&self) -> Self {
        Self {
            score: Arc::clone(&self.score),
            curve: self.curve,
            weight: self.weight,
            invert: self.invert,
        }
    }
}

impl<A: PlanComponent> Consideration<A> {
    /// A full-weight, linear, non-inverted consideration over `A`.
    pub fn new(score: impl Fn(&A) -> f32 + Send + Sync + 'static) -> Self {
        Self {
            score: Arc::new(score),
            curve: curves::linear,
            weight: 1.0,
            invert: false,
        }
    }

    /// Shape the normalized value (applied after the 0–1 clamp).
    pub fn curve(mut self, curve: fn(f32) -> f32) -> Self {
        self.curve = curve;
        self
    }

    /// Scale this axis's contribution inside a blend (default 1.0; negative
    /// weights are allowed and simply push the average down).
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Flip the axis: "1 − shaped(x)". Applied after the curve, so
    /// `curve(quadratic).invert()` scores `1 − x²`.
    pub fn invert(mut self) -> Self {
        self.invert = true;
        self
    }

    /// Evaluate the axis: clamp → curve → (optional) invert → weight.
    /// Useful for per-axis logging/debugging independent of any blend.
    pub fn evaluate(&self, component: &A) -> f32 {
        let v = (self.curve)((self.score)(component).clamp(0.0, 1.0));
        let v = if self.invert { 1.0 - v } else { v };
        v * self.weight
    }
}

/// Blend two considerations over different components into one
/// [`utility_fn`](crate::tasks::MethodBuilder::utility_fn)-compatible scorer
/// (weighted average). The returned closure flows straight into the existing
/// `IntoUtility` machinery — parameters are resolved by component type as
/// with any scorer closure.
pub fn blend2<A: PlanComponent, B: PlanComponent>(
    a: Consideration<A>,
    b: Consideration<B>,
) -> impl Fn(&A, &B) -> f32 + Send + Sync + Clone + 'static {
    move |x, y| {
        let (w_a, w_b) = (a.weight, b.weight);
        let total = w_a + w_b;
        if total == 0.0 {
            0.0
        } else {
            (a.evaluate(x) + b.evaluate(y)) / total
        }
    }
}

/// Blend three considerations (see [`blend2`]).
pub fn blend3<A: PlanComponent, B: PlanComponent, C: PlanComponent>(
    a: Consideration<A>,
    b: Consideration<B>,
    c: Consideration<C>,
) -> impl Fn(&A, &B, &C) -> f32 + Send + Sync + Clone + 'static {
    move |x, y, z| {
        let total = a.weight + b.weight + c.weight;
        if total == 0.0 {
            0.0
        } else {
            (a.evaluate(x) + b.evaluate(y) + c.evaluate(z)) / total
        }
    }
}

/// Blend four considerations (see [`blend2`]).
pub fn blend4<A: PlanComponent, B: PlanComponent, C: PlanComponent, D: PlanComponent>(
    a: Consideration<A>,
    b: Consideration<B>,
    c: Consideration<C>,
    d: Consideration<D>,
) -> impl Fn(&A, &B, &C, &D) -> f32 + Send + Sync + Clone + 'static {
    move |w, x, y, z| {
        let total = a.weight + b.weight + c.weight + d.weight;
        if total == 0.0 {
            0.0
        } else {
            (a.evaluate(w) + b.evaluate(x) + c.evaluate(y) + d.evaluate(z)) / total
        }
    }
}
