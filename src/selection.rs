//! Branch-selection policies, search strategies, and the trace contract.
//!
//! Two orthogonal axes live here:
//!
//! - **Selection policies** ([`SelectionPolicy`]) are baked into compound
//!   tasks and govern the *order in which valid branches are offered* to the
//!   search. They never override preconditions — only valid branches are
//!   ranked.
//! - **Search strategies** ([`HtnSearchStrategy`]) govern *how the task
//!   network is expanded* (the planner's algorithm). They live on
//!   [`HtnConfig`](crate::ecs::HtnConfig) and can be overridden per agent via
//!   the [`SearchOverride`] component.

use std::sync::Arc;

use bevy_ecs::message::Message;
use bevy_ecs::prelude::Component;

use crate::domain::HtnDomain;
use crate::planner::Plan;
use crate::state::PlanState;

// ---------------------------------------------------------------------------
// Selection policies
// ---------------------------------------------------------------------------

/// How a compound task's *valid* branches are ranked before the planner
/// descends. Applied after precondition evaluation — only valid branches are
/// ranked, and a look-ahead pin (unique surviving method) overrides ranking
/// entirely.
#[derive(Clone, Default)]
pub enum SelectionPolicy {
    /// Branches are tried in declaration order (the default; what the
    /// planner has always done).
    #[default]
    FirstMatch,

    /// All valid branches are scored by their `utility` closure (branches
    /// without one score 0); the highest wins. Ties break by declaration
    /// order. Deterministic: backtracking re-derives the same order.
    HighestUtility,

    /// Valid branches are sampled without replacement, proportional to their
    /// utility scores (branches without one weigh 1.0). The sampled order is
    /// snapshotted into the decomposition frame, so backtracking exhausts the
    /// sampled order instead of re-sampling — completeness is preserved.
    ///
    /// Sampling is stateless and deterministic: the permutation is derived
    /// from `seed` and the choice point's position in the plan, so the same
    /// state always yields the same order (stable replans).
    WeightedRandom {
        /// Seed for the deterministic weighted sampler.
        seed: u64,
    },

    /// The caller supplies the comparator at domain-build time. The ranker
    /// must be deterministic for a given `(candidates, state)` pair, and its
    /// output must be a permutation of the candidate indices (missing
    /// candidates are appended in declaration order so no branch is lost).
    Custom(Arc<dyn BranchRanker>),
}

/// One valid branch offered to a [`BranchRanker`].
pub struct BranchCandidate<'a> {
    /// The branch's declaration index (its MTR identity).
    pub index: u32,
    /// The branch's declared name, if any.
    pub name: Option<&'static str>,
    /// The branch's declared static utility, if any.
    pub utility: Option<f32>,
    /// The branch's subtask list (task indices, for structural heuristics).
    pub subtasks: &'a [u32],
}

/// Ranks *valid* branch candidates. Implementations must be deterministic
/// for a given `(candidates, state)` pair.
pub trait BranchRanker: Send + Sync {
    /// Appends the candidate indices to `out` in preferred order. Scratch-
    /// buffer signature: no per-node allocation inside the planner.
    fn rank(&self, candidates: &[BranchCandidate<'_>], state: &PlanState, out: &mut Vec<u32>);

}

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
