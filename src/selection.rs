//! Search strategies and the decomposition-trace contract.
//!
//! - **Search strategies** ([`HtnSearchStrategy`]) govern *how the task
//!   network is expanded* (the planner's algorithm). They live on
//!   [`HtnConfig`](crate::ecs::HtnConfig) and can be overridden per agent via
//!   the [`SearchOverride`] component.
//! - **Tracing** ([`DecompositionTrace`]) is the driver bridge: one event per
//!   branch commitment when tracing is enabled.
//!
//! (Branch-*selection* policies — the baked per-task branch ranking — live
//! with the baked network in [`crate::domain`].)

use std::sync::Arc;

use bevy_ecs::message::Message;
use bevy_ecs::prelude::Component;

use crate::domain::HtnDomain;
use crate::planner::Plan;
use crate::state::PlanState;

// ---------------------------------------------------------------------------
// Search strategies
// ---------------------------------------------------------------------------

/// The search algorithm used to expand the task network.
#[derive(Clone, Default)]
pub enum HtnSearchStrategy {
    /// Left-to-right DFS with full MTR backtracking and look-ahead pruning.
    /// The default; what the planner has always done.
    #[default]
    DepthFirst,

    /// DFS that abandons the branch on first downstream failure and returns
    /// the partial plan immediately (no backtracking). Cheaper per tick,
    /// less complete — for tight per-frame budgets.
    DepthFirstFailFast,

    /// Depth-first with branch-and-bound on accumulated primitive cost:
    /// keeps the cheapest *complete* plan found within the sanity budget and
    /// prunes any branch whose accumulated cost plus the bake-time
    /// `min_cost` lower bound of its remaining sequence cannot strictly beat
    /// the best plan found so far. Requires `cost`/`cost_fn` annotations
    /// (unannotated primitives count 0, so with none declared this behaves
    /// exactly like [`DepthFirst`](HtnSearchStrategy::DepthFirst)). Anytime
    /// and deterministic: the first complete plan found is returned if the
    /// budget runs out, and the cost-optimal plan when the budget suffices
    /// to exhaust the space.
    CostBounded,

    /// Caller-supplied strategy. The strategy object owns any persistent
    /// state (MCTS statistics, ACO pheromone tables, seeded RNGs) — wrap it
    /// in `Arc` and share it via `HtnConfig` or a per-agent
    /// [`SearchOverride`] so a population of agents shares one table.
    Custom(Arc<dyn Searcher>),
}

/// A pluggable search strategy. Implementations own their statistics and
/// randomness; they must be internally synchronized (`&self`).
pub trait Searcher: Send + Sync {
    /// Search the domain from `state`. Returns the best plan found (`None`
    /// if nothing decomposes).
    fn search(&self, domain: &HtnDomain, state: &PlanState) -> Option<Plan>;
}

/// Per-agent search override. Agents opt in by inserting this component; the
/// driver uses it instead of [`HtnConfig`](crate::ecs::HtnConfig)'s strategy
/// for that entity.
#[derive(Component, Clone, Default)]
pub struct SearchOverride {
    /// Overrides the global strategy for this agent.
    pub strategy: Option<HtnSearchStrategy>,
    /// Overrides the global sanity budget for this agent.
    pub sanity_limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Tracing
// ---------------------------------------------------------------------------

/// One branch-selection decision, emitted per *commitment* (not per
/// precondition attempt) when tracing is enabled via
/// [`HtnPlanner::plan_traced`](crate::planner::HtnPlanner::plan_traced).
///
/// The driver forwards these into `Messages<DecompositionTrace>` when
/// `HtnConfig::debug_trace` is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Message)]
pub struct DecompositionTrace {
    /// Declaration index of the compound task.
    pub compound: u32,
    /// Declaration index of the branch the event is about.
    pub branch: u32,
    /// The branch's declared name, if any.
    pub branch_name: Option<&'static str>,
    /// What happened to the branch.
    pub outcome: TraceOutcome,
}

/// What happened to a traced branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceOutcome {
    /// The branch was selected and its subtasks committed.
    Selected,
    /// The branch's preconditions failed (it was never offered to the search).
    PrecondFailed,
    /// The branch's subtree failed and the search backtracked past it.
    Backtracked,
}
