use coeus_core::BackendError;

pub(super) fn map_leto_error(operation: &'static str, error: leto::LetoError) -> BackendError {
    match error {
        leto::LetoError::ShapeMismatch { lhs, rhs } => BackendError::ShapeMismatch {
            operation,
            lhs,
            rhs,
        },
        leto::LetoError::IncompatibleBroadcast { from, to } => {
            BackendError::IncompatibleBroadcast {
                operation,
                from,
                to,
            }
        }
        leto::LetoError::Overflow { reason } => BackendError::Overflow { operation, reason },
        leto::LetoError::StorageError { reason } => BackendError::Storage { operation, reason },
        other => BackendError::Storage {
            operation,
            reason: other.to_string(),
        },
    }
}
