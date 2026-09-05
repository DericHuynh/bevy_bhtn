//! Errors produced by building, verifying, or planning an HTN domain.

use thiserror::Error;

/// Errors produced by building, verifying, or planning an HTN domain.
#[derive(Debug, Error)]
pub enum HtnError {
    /// The domain builder was given an invalid specification (a task mixing
    /// compound and primitive declarations, a methodless compound task, a
    /// missing root, ...). Every authoring bug one `build()` call can
    /// collect is reported together — one entry per bug, joined in the
    /// `Display` form.
    #[error("Invalid HTN domain: {}", errors.join("; "))]
    Builder {
        /// Each collected authoring error, in collection order.
        errors: Vec<String>,
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
    /// chain of primitives reaches the goal (or its expansion budget ran
    /// out); for the forward planner it means the domain genuinely cannot
    /// solve the root task in this state (an empty *successful* plan is `Ok`,
    /// never this error). Distinct from budget truncation, which is a
    /// `Partial` [`Plan`](crate::planner::Plan) value, not an error.
    #[error("No plan could be found: the search space was exhausted with no valid decomposition")]
    NoPlan,
}

impl HtnError {
    /// Create a [`HtnError::Builder`] carrying a single error entry (the
    /// multi-error form comes from `HtnDomain::build`, which collects every
    /// authoring bug before failing).
    pub fn builder(details: impl Into<String>) -> Self {
        HtnError::Builder {
            errors: vec![details.into()],
        }
    }
}

/// Convenience alias for results that fail with [`HtnError`].
pub type HtnResult<T> = std::result::Result<T, HtnError>;
