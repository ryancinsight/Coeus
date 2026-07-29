use std::error::Error;

/// Contract failures returned by neural-network module execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModuleError<E>
where
    E: Error + 'static,
{
    /// A backend operation failed while evaluating the module.
    #[error("{module} backend operation failed")]
    Backend {
        /// Module family performing the operation.
        module: &'static str,
        /// Backend-owned typed failure.
        #[source]
        source: E,
    },
    /// The input rank violates the module contract.
    #[error("{module} expected input rank {expected}, got {actual}")]
    InvalidRank {
        /// Module family rejecting the input.
        module: &'static str,
        /// Accepted rank or rank range.
        expected: &'static str,
        /// Observed input rank.
        actual: usize,
    },
    /// A parameter or input has an incompatible shape.
    #[error("{module} {parameter} shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Module family rejecting the shape.
        module: &'static str,
        /// Input or parameter role.
        parameter: &'static str,
        /// Required shape.
        expected: Vec<usize>,
        /// Observed shape.
        actual: Vec<usize>,
    },
    /// The input channel count differs from the configured module width.
    #[error("{module} channel mismatch: expected {expected}, got {actual}")]
    ChannelMismatch {
        /// Module family rejecting the channel count.
        module: &'static str,
        /// Configured channel count.
        expected: usize,
        /// Observed channel count.
        actual: usize,
    },
    /// A group count cannot partition the configured channels.
    #[error("{module} group count {groups} does not divide {channels} channels")]
    InvalidGroupCount {
        /// Module family rejecting the configuration.
        module: &'static str,
        /// Requested normalization groups.
        groups: usize,
        /// Configured channel count.
        channels: usize,
    },
    /// The normalization epsilon is not strictly positive.
    #[error("{module} epsilon must be positive")]
    NonPositiveEpsilon {
        /// Module family rejecting epsilon.
        module: &'static str,
    },
    /// Interior module state is already borrowed.
    #[error("{module} state {state} is already borrowed")]
    StateBorrow {
        /// Module family attempting the borrow.
        module: &'static str,
        /// Conflicting state field.
        state: &'static str,
    },
}

/// Contract failures when loading optimizer-owned named parameters.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ParameterLoadError {
    /// The incoming inventory has a different parameter count.
    #[error("named parameter count mismatch: expected {expected}, got {actual}")]
    Count {
        /// Module inventory length.
        expected: usize,
        /// Incoming inventory length.
        actual: usize,
    },
    /// A hierarchical name differs at a stable inventory position.
    #[error("named parameter mismatch at index {index}: expected {expected}, got {actual}")]
    Name {
        /// Inventory position.
        index: usize,
        /// Module-owned path.
        expected: String,
        /// Incoming path.
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::ModuleError;
    use coeus_core::BackendError;

    #[test]
    fn backend_failure_preserves_module_and_source() {
        let source = BackendError::AxisOutOfRange {
            operation: "sum",
            axis: 3,
            rank: 2,
        };

        let error = ModuleError::Backend {
            module: "LayerNorm",
            source: source.clone(),
        };

        match error {
            ModuleError::Backend {
                module,
                source: actual,
            } => {
                assert_eq!(module, "LayerNorm");
                assert_eq!(actual, source);
            }
            other => panic!("expected backend failure, got {other:?}"),
        }
    }
}
