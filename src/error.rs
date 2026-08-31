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

    /// A referenced task name was not defined in the domain.
    #[error("Task `{name}` was not found in the HTN domain")]
    UnknownTask {
        /// The missing task name.
        name: String,
    },

    /// A back-plan could not reach the goal state.
    #[error("No plan reaches the goal state from the initial state")]
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

    /// Create a [`HtnError::UnknownTask`].
    pub fn unknown_task(name: impl Into<String>) -> Self {
        HtnError::UnknownTask { name: name.into() }
    }
}

/// Convenience alias for results that fail with [`HtnError`].
pub type HtnResult<T> = std::result::Result<T, HtnError>;
