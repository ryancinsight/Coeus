//! Distributed training bindings: the local and TCP communicators and the
//! gradient synchronization that runs over them.

mod local;
mod tcp;

pub use local::{create_local_cluster, PyLocalCommunicator};
pub use tcp::{create_tcp_loopback_cluster, PyTcpCommunicator, PyTcpMesh};

use crate::error::map_backend_error;
use crate::tensor::PyTensor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Synchronize and average gradients across all ranks in a process group (releasing GIL).
#[pyfunction]
pub fn synchronize_gradients(
    py: Python<'_>,
    params: Vec<Py<PyTensor>>,
    comm: &PyLocalCommunicator,
) -> PyResult<()> {
    let mut rust_params: Vec<coeus_autograd::Var<f64>> = params
        .iter()
        .map(|p| p.bind(py).borrow().inner.clone())
        .collect();

    let comm_inner = comm.inner.clone();
    py.allow_threads(move || coeus_dist::synchronize_gradients(&mut rust_params, &comm_inner))
        .map_err(|error| match error {
            coeus_dist::GradientSyncError::Backend(source) => map_backend_error(source),
            other => PyRuntimeError::new_err(other.to_string()),
        })
}
