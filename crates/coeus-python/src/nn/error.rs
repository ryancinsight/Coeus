use coeus_core::BackendError;
use coeus_nn::ModuleError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

pub(crate) fn map_backend_error(error: BackendError) -> PyErr {
    match error {
        BackendError::UnsupportedRank { .. }
        | BackendError::LayoutRankMismatch { .. }
        | BackendError::ShapeMismatch { .. }
        | BackendError::AxisOutOfRange { .. }
        | BackendError::IncompatibleBroadcast { .. }
        | BackendError::Storage { .. } => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

pub(crate) fn map_module_error(error: ModuleError<BackendError>) -> PyErr {
    match error {
        ModuleError::Backend { module, source } => {
            PyRuntimeError::new_err(format!("{module} backend operation failed: {source}"))
        }
        ModuleError::StateBorrow { .. } => PyRuntimeError::new_err(error.to_string()),
        ModuleError::InvalidRank { .. }
        | ModuleError::InvalidAxis { .. }
        | ModuleError::UnevenSplit { .. }
        | ModuleError::ShapeMismatch { .. }
        | ModuleError::ChannelMismatch { .. }
        | ModuleError::InvalidGroupCount { .. }
        | ModuleError::InvalidEpsilon { .. }
        | ModuleError::InsufficientElements { .. } => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::map_module_error;
    use coeus_core::BackendError;
    use coeus_nn::ModuleError;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::Python;

    #[test]
    fn contract_failure_maps_to_value_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let error = map_module_error(ModuleError::InvalidRank {
                module: "LayerNorm",
                expected: "2",
                actual: 1,
            });

            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error
                .to_string()
                .contains("LayerNorm expected input rank 2"));
        });
    }

    #[test]
    fn backend_failure_maps_to_runtime_error_with_source() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let error = map_module_error(ModuleError::Backend {
                module: "LayerNorm",
                source: BackendError::AxisOutOfRange {
                    operation: "sum",
                    axis: 3,
                    rank: 2,
                },
            });

            assert!(error.is_instance_of::<PyRuntimeError>(py));
            let message = error.to_string();
            assert!(message.contains("LayerNorm backend operation failed"));
            assert!(message.contains("sum"));
            assert!(message.contains("axis 3"));
        });
    }
}
