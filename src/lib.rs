//! # bevy_bhtn — Hierarchical Task Networks for Bevy AI
//!
//! A headless HTN planner with an idiomatic [`bevy_ecs`] execution layer,
//! combining the two classic planning shapes:
//!
//! - **Forward planning** — decompose a root compound task into an ordered
//!   list of primitive tasks, applying effects to a [`PlanState`] scratchpad
//!   and backtracking (with look-ahead pruning) when a method's preconditions
//!   fail. The chosen methods are recorded as an MTR (Method Traversal
//!   Record).
//! - **Backward / goal-state planning** — given a goal task (a set of desired
//!   [`Effect`]s) and an initial state, walk the domain's primitive tasks in
//!   reverse to find a dependency-ordered plan that reaches the goal.
//!
//! # The architecture: function-graph reflection
//!
//! Tasks are **plain Rust functions**. Every named function has a unique
//! zero-sized type, so the function itself is the task's identity — no marker
//! structs, no string ids, no registration. At startup,
//! [`HtnDomain::from_root`] records the root function (and everything it
//! references) and **bakes** the graph into flat `Vec`s with contiguous task
//! indices: the runtime planner searches the arrays directly (O(1) node
//! lookups, allocation-light backtracking), while the recorded names
//! (`std::any::type_name`) keep the domain fully inspectable.
//!
//! State is **ordinary Bevy components**. Preconditions and effects are
//! closures over component references (`|ammo: &Ammo| ...`), monomorphized at
//! build time into type-erased checkers/mutators over a dense [`PlanState`]
//! scratchpad — no reflection anywhere. The ECS driver
//! ([`htn_ai_system`]) extracts the scratchpad from the entity's real
//! components, plans, then executes one step per tick: re-validating
//! preconditions against the world, dispatching the task's action commands,
//! and committing effects back to the real components.
//!
//! ```
//! use bevy_bhtn::prelude::*;
//! use bevy_ecs::prelude::*;
//!
//! #[derive(Component, Clone, Default, Debug)]
//! struct Ammo(pub u32);
//!
//! fn reload(task: &mut TaskBuilder) {
//!     task.effect(|ammo: &mut Ammo| ammo.0 = 30);
//! }
//!
//! fn engage(task: &mut TaskBuilder) {
//!     task.branch()
//!         .precondition(|ammo: &Ammo| ammo.0 == 0)
//!         .then(reload);
//! }
//!
//! let domain = HtnDomain::from_root(engage).build().unwrap();
//! # let _ = domain;
//! ```

#![deny(missing_docs)]

pub mod back_planner;
pub mod domain;
pub mod ecs;
pub mod error;
pub mod lookahead;
pub mod planner;
pub mod selection;
pub mod state;
pub mod summaries;
pub mod tasks;

pub use back_planner::*;
pub use domain::*;
pub use error::*;
pub use summaries::{FieldSet, TaskSummary};
pub use tasks::*;

/// Convenience re-exports for crate-wide use.
pub mod prelude {
    pub use crate::back_planner::*;
    pub use crate::domain::*;
    pub use crate::ecs::*;
    pub use crate::error::*;
    pub use crate::planner::*;
    pub use crate::selection::*;
    pub use crate::state::*;
    pub use crate::summaries::{FieldSet, TaskSummary};
    pub use crate::tasks::*;
}
