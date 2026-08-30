//! # cdda_htn — Hierarchical Task Networks for CDDA AI
//!
//! A self-contained, headless [`bevy_reflect`]-backed HTN planner. It combines
//! the two classic shapes:
//!
//! - **Forward planning** (bevy_htn style) — decompose a root compound task into
//!   an ordered list of primitive tasks, applying expected effects to a working
//!   copy of the state and backtracking when a method's preconditions fail. The
//!   chosen methods are recorded as an MTR (Method Traversal Record).
//! - **Backward / goal-state planning** — given a `goal_task` (a set of desired
//!   [`Effect`]s) and an initial state, walk the domain's primitive tasks in
//!   reverse to find a dependency-ordered plan that reaches the goal.
//!
//! Tasks, methods, conditions, and effects are plain data (parsed from `.htn`
//! files via [`dsl::parse_htn`]); operators are user types registered in a
//! `bevy_reflect::TypeRegistry` and executed at run time. The crate has **no
//! ECS / `Component` dependency and no `cdda_sim` / `cdda_components`
//! dependency**, so it is a leaf that any AI layer can adopt. Its only Bevy
//! ties are `bevy_reflect` (for reflection) and `bevy_asset` (for the optional
//! `.htn` asset loader).

#![deny(missing_docs)]

use bevy_reflect::Reflect;

mod conditions;
mod domain;
mod dsl;
mod effects;
mod error;
mod lookahead;
mod summaries;
mod tasks;

pub mod asset_loader;
pub mod back_planner;
pub mod operators;
pub mod planner;

/// A reflected plan state that the planners act on.
///
/// Any `struct` that derives `Reflect` and is `Default` (so operators can be
/// initialised) plus `Clone + Debug` satisfies this via the blanket impl. There
/// is deliberately **no `Component` requirement** so the crate stays usable
/// headless and is trivially testable outside the ECS.
pub trait HtnState: Reflect + Default + Clone + std::fmt::Debug {}

impl<T: Reflect + Default + Clone + std::fmt::Debug> HtnState for T {}

pub use conditions::*;
pub use domain::*;
pub use dsl::parse_htn;
pub use effects::*;
pub use error::*;
pub use summaries::{FieldSet, TaskSummary};
pub use tasks::*;

/// Convenience re-exports for crate-wide use.
pub mod prelude {
    pub use crate::asset_loader::*;
    pub use crate::back_planner::*;
    pub use crate::conditions::*;
    pub use crate::domain::*;
    pub use crate::dsl::parse_htn;
    pub use crate::effects::*;
    pub use crate::error::*;
    pub use crate::operators::*;
    pub use crate::planner::*;
    pub use crate::tasks::*;
    pub use crate::HtnState;
}
