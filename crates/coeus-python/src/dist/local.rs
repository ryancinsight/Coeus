//! The in-process communicator: simulated ranks sharing one address space.

use crate::tensor::PyTensor;
use coeus_dist::Communicator;
use pyo3::prelude::*;

/// Python-exposed LocalCommunicator.
#[pyclass(name = "LocalCommunicator")]
#[derive(Clone)]
pub struct PyLocalCommunicator {
    /// Underlying Rust LocalCommunicator handle.
    pub inner: coeus_dist::LocalCommunicator,
}

#[pymethods]
impl PyLocalCommunicator {
    /// Get the rank of the current process within the process group.
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Get the total number of processes in the process group.
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Synchronize all ranks in the process group (blocking barrier, releasing GIL).
    fn barrier(&self, py: Python<'_>) {
        let comm = self.inner.clone();
        py.allow_threads(move || {
            use coeus_dist::Communicator;
            let Ok(()) = comm.barrier();
        });
    }

    /// Reduce and distribute a tensor to all processes in-place (releasing GIL).
    fn all_reduce(&self, tensor: &Bound<'_, PyTensor>, py: Python<'_>) -> PyResult<()> {
        let mut t_borrow = tensor.try_borrow_mut()?;
        let comm = self.inner.clone();
        let t_val = t_borrow.inner.tensor.clone();
        let t_val = py.allow_threads(move || {
            let backend = coeus_core::MoiraiBackend::new();
            let mut t_val = t_val;
            let Ok(()) = comm.all_reduce::<f64, _, coeus_dist::Sum>(&mut t_val, &backend);
            t_val
        });
        t_borrow.inner.tensor = t_val;
        Ok(())
    }

    /// Broadcast a tensor from the root process rank to all other processes in-place (releasing GIL).
    fn broadcast(&self, tensor: &Bound<'_, PyTensor>, root: usize, py: Python<'_>) -> PyResult<()> {
        let mut t_borrow = tensor.try_borrow_mut()?;
        let comm = self.inner.clone();
        let t_val = t_borrow.inner.tensor.clone();
        let t_val = py.allow_threads(move || {
            let backend = coeus_core::MoiraiBackend::new();
            let mut t_val = t_val;
            let Ok(()) = comm.broadcast::<f64, _>(&mut t_val, root, &backend);
            t_val
        });
        t_borrow.inner.tensor = t_val;
        Ok(())
    }

    /// Gather tensors from all processes into a list of output tensors (releasing GIL).
    fn all_gather(
        &self,
        tensor: &PyTensor,
        output: Vec<Py<PyTensor>>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let comm = self.inner.clone();
        let size = comm.size();
        if output.len() != size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Output list length ({}) must equal communicator size ({})",
                output.len(),
                size
            )));
        }

        let input_tensor = tensor.inner.tensor.clone();
        let mut rust_tensors = Vec::with_capacity(size);
        for item in &output {
            rust_tensors.push(item.bind(py).borrow().inner.tensor.clone());
        }

        let rust_tensors = py.allow_threads(move || {
            let backend = coeus_core::MoiraiBackend::new();
            let mut rust_tensors = rust_tensors;
            let Ok(()) = comm.all_gather::<f64, _>(&input_tensor, &mut rust_tensors, &backend);
            rust_tensors
        });

        for (item, rust_t) in output.iter().zip(rust_tensors) {
            item.bind(py).borrow_mut().inner.tensor = rust_t;
        }

        Ok(())
    }

    /// Reduce a tensor from all processes to a single root process in-place (releasing GIL).
    fn reduce(&self, tensor: &Bound<'_, PyTensor>, root: usize, py: Python<'_>) -> PyResult<()> {
        let mut t_borrow = tensor.try_borrow_mut()?;
        let comm = self.inner.clone();
        let t_val = t_borrow.inner.tensor.clone();
        let t_val = py.allow_threads(move || {
            let backend = coeus_core::MoiraiBackend::new();
            let mut t_val = t_val;
            let Ok(()) = comm.reduce::<f64, _, coeus_dist::Sum>(&mut t_val, root, &backend);
            t_val
        });
        t_borrow.inner.tensor = t_val;
        Ok(())
    }

    /// Gather tensors from all processes into a single slice on the root process (releasing GIL).
    fn gather(
        &self,
        tensor: &PyTensor,
        output: Vec<Py<PyTensor>>,
        root: usize,
        py: Python<'_>,
    ) -> PyResult<()> {
        let comm = self.inner.clone();
        let size = comm.size();
        let rank = comm.rank();
        if rank == root && output.len() != size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Output list length ({}) must equal communicator size ({}) on root",
                output.len(),
                size
            )));
        }

        let input_tensor = tensor.inner.tensor.clone();
        let mut rust_tensors = Vec::with_capacity(output.len());
        for item in &output {
            rust_tensors.push(item.bind(py).borrow().inner.tensor.clone());
        }

        let rust_tensors = py.allow_threads(move || {
            let backend = coeus_core::MoiraiBackend::new();
            let mut rust_tensors = rust_tensors;
            let Ok(()) = comm.gather::<f64, _>(&input_tensor, &mut rust_tensors, root, &backend);
            rust_tensors
        });

        if rank == root {
            for (item, rust_t) in output.iter().zip(rust_tensors) {
                item.bind(py).borrow_mut().inner.tensor = rust_t;
            }
        }

        Ok(())
    }

    /// Scatter a slice of tensors from the root process to all processes in-place (releasing GIL).
    fn scatter(
        &self,
        tensor: &Bound<'_, PyTensor>,
        input: Vec<Py<PyTensor>>,
        root: usize,
        py: Python<'_>,
    ) -> PyResult<()> {
        let mut t_borrow = tensor.try_borrow_mut()?;
        let comm = self.inner.clone();
        let size = comm.size();
        let rank = comm.rank();
        if rank == root && input.len() != size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input list length ({}) must equal communicator size ({}) on root",
                input.len(),
                size
            )));
        }

        let t_val = t_borrow.inner.tensor.clone();
        let mut rust_tensors = Vec::with_capacity(input.len());
        for item in &input {
            rust_tensors.push(item.bind(py).borrow().inner.tensor.clone());
        }

        let (t_val, _rust_tensors) = py.allow_threads(move || {
            let backend = coeus_core::MoiraiBackend::new();
            let mut t_val = t_val;
            let Ok(()) = comm.scatter::<f64, _>(&mut t_val, &rust_tensors, root, &backend);
            (t_val, rust_tensors)
        });

        t_borrow.inner.tensor = t_val;
        Ok(())
    }
}

/// Create a new process cluster with `world_size` simulated ranks.
#[pyfunction]
pub fn create_local_cluster(world_size: usize) -> PyResult<Vec<PyLocalCommunicator>> {
    let communicators = coeus_dist::LocalCommunicator::create_cluster(world_size);
    Ok(communicators
        .into_iter()
        .map(|comm| PyLocalCommunicator { inner: comm })
        .collect())
}
