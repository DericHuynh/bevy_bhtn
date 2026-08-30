//! Errors produced by parsing, verifying, or planning an HTN domain.

use thiserror::Error;

/// Errors produced by parsing, verifying, or planning an HTN domain.
#[derive(Debug, Error)]
pub enum HtnError {
    /// The `.htn` text could not be parsed by the pest grammar.
    #[error("Failed to parse HTN: {details}")]
    Parser {
        /// The pest parser error message.
        details: String,
    },

    /// The `.htn` schema block was malformed.
    #[error("Invalid HTN schema: {details}")]
    Schema {
        /// Details of the schema problem.
        details: String,
    },

    /// A referenced task name was not defined in the domain.
    #[error("Task `{name}` was not found in the HTN domain")]
    UnknownTask {
        /// The missing task name.
        name: String,
    },

    /// A condition referenced a field/type that does not exist on the state.
    #[error("Condition `{syntax}` is invalid: {details}")]
    Condition {
        /// The condition's `.htn` syntax.
        syntax: String,
        /// Why it failed verification.
        details: String,
    },

    /// An effect referenced a field/type that does not exist on the state.
    #[error("Effect `{syntax}` is invalid: {details}")]
    Effect {
        /// The effect's `.htn` syntax.
        syntax: String,
        /// Why it failed verification.
        details: String,
    },

    /// An operator is missing its type-registry registration (Reflect).
    #[error("Operator `{name}` is not registered: {details}")]
    Operator {
        /// The operator's name as written in the domain.
        name: String,
        /// The parameter names referenced by the operator.
        params: Vec<String>,
        /// Details of what registration is missing.
        details: String,
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
    /// Create a [`HtnError::Parser`].
    pub fn parser(details: impl Into<String>) -> Self {
        HtnError::Parser {
            details: details.into(),
        }
    }

    /// Create a [`HtnError::Schema`].
    pub fn schema(details: impl Into<String>) -> Self {
        HtnError::Schema {
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
