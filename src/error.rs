//! Errors produced by building, verifying, or planning an HTN domain.

use thiserror::Error;

/// Errors produced by building, verifying, or planning an HTN domain.
#[derive(Debug, Error)]
pub enum HtnError {
    /// The domain builder was given an invalid specification (a task mixing
    /// compound and primitive declarations, a methodless compound task, a
    /// missing root, ...).
    #[error("Invalid HTN domain: {details}")]
    Builder {
        /// Details of the builder problem.
        details: String,
    },

    /// A task or goal function that was never recorded in the domain (e.g. a
    /// back-planning goal or an adversarial root not registered via
    /// `from_root`/`root`). Resolution is by the function's `TypeId`; the
    /// string is its `type_name` for diagnosis.
    #[error("Task function `{type_name}` was never recorded in this domain")]
    UnregisteredTask {
        /// The `type_name` of the unregistered function.
        type_name: String,
    },

    /// Planning failed: the search space was exhausted with no complete
    /// decomposition. For [`BackPlanner`](crate::back_planner) this means no
    /// chain of primitives reaches the goal; for the forward planner it means
    /// the domain genuinely cannot solve the root task in this state (an
    /// empty *successful* plan is `Ok`, never this error). Distinct from
    /// budget truncation, which is a `Partial` [`Plan`](crate::planner::Plan).
    #[error("No plan could be found: the search space was exhausted with no valid decomposition")]
    NoPlan,

    /// The planner exceeded its internal step budget (defensive).
    #[error("Planning exceeded the sanity limit ({limit} steps)")]
    SanityLimit {
        /// The step budget that was exceeded.
        limit: usize,
    },
}

impl HtnError {
    /// Create a [`HtnError::Builder`].
    pub fn builder(details: impl Into<String>) -> Self {
        HtnError::Builder {
            details: details.into(),
        }
    }
}

/// Convenience alias for results that fail with [`HtnError`].
pub type HtnResult<T> = std::result::Result<T, HtnError>;
