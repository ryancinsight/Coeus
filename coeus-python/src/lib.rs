// ── Coeus Python bindings entry point ──

use crate::tensor::PyTensor;
use coeus_dist::Communicator;
use pyo3::prelude::*;

/// Mnemosyne is the ecosystem allocation SSOT. Registering it as the extension's
/// global allocator routes *all* Rust-side allocations (tensor buffers already go
/// through it explicitly in `coeus-core::storage`, plus every incidental `Vec`/
/// `Box`/intermediate) through one allocator. Disable with `--no-default-features`
/// to fall back to the system allocator (e.g. for sanitizers or allocator profiling).
#[cfg(feature = "mnemosyne-global")]
#[global_allocator]
static GLOBAL: mnemosyne::Mnemosyne = mnemosyne::Mnemosyne;

pub mod nn;
pub mod optim;
pub mod tensor;

/// Element-wise ReLU activation.
#[pyfunction]
fn relu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::relu(&input.inner));
    PyTensor { inner }
}

/// Element-wise Sigmoid activation.
#[pyfunction]
fn sigmoid(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sigmoid(&input.inner));
    PyTensor { inner }
}

/// Element-wise Tanh activation.
#[pyfunction]
fn tanh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::tanh(&input.inner));
    PyTensor { inner }
}

/// Element-wise GELU activation.
#[pyfunction]
fn gelu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::activation::gelu(&input.inner));
    PyTensor { inner }
}

/// Element-wise SiLU activation.
#[pyfunction]
fn silu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::silu(&input.inner));
    PyTensor { inner }
}

/// Element-wise Mish activation.
#[pyfunction]
fn mish(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::activation::mish(&input.inner));
    PyTensor { inner }
}

/// Element-wise ELU activation.
#[pyfunction]
fn elu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::elu(&input.inner));
    PyTensor { inner }
}

/// Element-wise Softplus activation.
#[pyfunction]
fn softplus(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softplus(&input.inner));
    PyTensor { inner }
}

/// Element-wise GELU tanh approximation activation.
#[pyfunction]
fn gelu_tanh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::gelu_tanh(&input.inner));
    PyTensor { inner }
}

/// Element-wise LeakyReLU activation.
#[pyfunction]
#[pyo3(signature = (input, negative_slope = 0.01))]
fn leaky_relu(input: &PyTensor, negative_slope: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::leaky_relu(&input.inner, negative_slope));
    PyTensor { inner }
}

/// Mean Squared Error loss.
#[pyfunction]
fn mse_loss(pred: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::mse_loss(&pred.inner, &target.inner));
    PyTensor { inner }
}

/// Cross-entropy loss.
#[pyfunction]
fn cross_entropy_loss(logits: &PyTensor, targets: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::cross_entropy_loss(&logits.inner, &targets));
    PyTensor { inner }
}

/// Binary Cross-Entropy Loss.
#[pyfunction]
#[pyo3(signature = (pred, target, eps = 1e-7))]
fn binary_cross_entropy(pred: &PyTensor, target: &PyTensor, eps: f64, py: Python<'_>) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::binary_cross_entropy(&pred.inner, &target.inner, eps));
    PyTensor { inner }
}

/// Negative Log-Likelihood Loss.
#[pyfunction]
fn nll_loss(log_probs: &PyTensor, targets: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::nll_loss(&log_probs.inner, &targets));
    PyTensor { inner }
}

/// Huber Loss.
#[pyfunction]
#[pyo3(signature = (pred, target, delta = 1.0))]
fn huber_loss(pred: &PyTensor, target: &PyTensor, delta: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::huber_loss(&pred.inner, &target.inner, delta));
    PyTensor { inner }
}

/// Cosine Embedding Loss.
#[pyfunction]
#[pyo3(signature = (x1, x2, y, margin = 0.0))]
fn cosine_embedding_loss(
    x1: &PyTensor,
    x2: &PyTensor,
    y: Vec<f64>,
    margin: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner = py
        .allow_threads(|| coeus_nn::loss::cosine_embedding_loss(&x1.inner, &x2.inner, &y, margin));
    PyTensor { inner }
}

/// Element-wise exponential.
#[pyfunction]
fn exp(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::exp(&input.inner));
    PyTensor { inner }
}

/// Element-wise natural logarithm.
#[pyfunction]
fn log(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log(&input.inner));
    PyTensor { inner }
}

/// Sum along the specified axis.
#[pyfunction]
fn sum_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sum_axis(&input.inner, axis));
    PyTensor { inner }
}

/// Mean along the specified axis.
#[pyfunction]
fn mean_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::mean_axis(&input.inner, axis));
    PyTensor { inner }
}

/// Compute log-softmax along the specified axis.
#[pyfunction]
fn log_softmax(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log_softmax(&input.inner, axis));
    PyTensor { inner }
}

/// Cumulative sum along the specified axis.
#[pyfunction]
fn cumsum(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::cumsum(&input.inner, dim));
    PyTensor { inner }
}

/// Constant padding.
#[pyfunction]
#[pyo3(signature = (input, pads, value = 0.0))]
fn pad(input: &PyTensor, pads: Vec<(usize, usize)>, value: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::pad(&input.inner, &pads, value));
    PyTensor { inner }
}

/// Concatenate a sequence of tensors along the specified dimension.
#[pyfunction]
fn cat(inputs: Vec<Py<PyTensor>>, dim: usize, py: Python<'_>) -> PyTensor {
    // Extract all Rust Var<f64> values while the GIL is held, before allow_threads.
    // Python<'_> is !Ungil — it must not be captured inside the allow_threads closure.
    let rust_inputs: Vec<coeus_autograd::Var<f64>> = inputs
        .iter()
        .map(|t| t.bind(py).borrow().inner.clone())
        .collect();
    let inner = py.allow_threads(move || {
        let ref_inputs: Vec<&coeus_autograd::Var<f64>> = rust_inputs.iter().collect();
        coeus_autograd::cat(&ref_inputs, dim)
    });
    PyTensor { inner }
}

/// Split a tensor into chunks of `chunk_size` along the specified dimension.
#[pyfunction]
fn split(input: &PyTensor, chunk_size: usize, dim: usize, py: Python<'_>) -> Vec<PyTensor> {
    let inner_chunks = py.allow_threads(|| coeus_autograd::split(&input.inner, chunk_size, dim));
    inner_chunks
        .into_iter()
        .map(|inner| PyTensor { inner })
        .collect()
}

/// Python-exposed MockCommunicator.
#[pyclass(name = "MockCommunicator")]
#[derive(Clone)]
pub struct PyMockCommunicator {
    pub inner: coeus_dist::MockCommunicator,
}

#[pymethods]
impl PyMockCommunicator {
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
            comm.barrier();
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
            comm.all_reduce::<f64, _, coeus_dist::Sum>(&mut t_val, &backend);
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
            comm.broadcast::<f64, _>(&mut t_val, root, &backend);
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
            comm.all_gather::<f64, _>(&input_tensor, &mut rust_tensors, &backend);
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
            comm.reduce::<f64, _, coeus_dist::Sum>(&mut t_val, root, &backend);
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
            comm.gather::<f64, _>(&input_tensor, &mut rust_tensors, root, &backend);
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
            comm.scatter::<f64, _>(&mut t_val, &rust_tensors, root, &backend);
            (t_val, rust_tensors)
        });

        t_borrow.inner.tensor = t_val;
        Ok(())
    }
}

/// Create a new process cluster with `world_size` simulated ranks.
#[pyfunction]
fn create_mock_cluster(world_size: usize) -> PyResult<Vec<PyMockCommunicator>> {
    let communicators = coeus_dist::MockCommunicator::create_cluster(world_size);
    Ok(communicators
        .into_iter()
        .map(|comm| PyMockCommunicator { inner: comm })
        .collect())
}

/// Synchronize and average gradients across all ranks in a process group (releasing GIL).
#[pyfunction]
fn synchronize_gradients(
    py: Python<'_>,
    params: Vec<Py<PyTensor>>,
    comm: &PyMockCommunicator,
) -> PyResult<()> {
    let mut rust_params: Vec<coeus_autograd::Var<f64>> = params
        .iter()
        .map(|p| p.bind(py).borrow().inner.clone())
        .collect();

    let comm_inner = comm.inner.clone();
    py.allow_threads(move || {
        coeus_dist::synchronize_gradients(&mut rust_params, &comm_inner);
    });

    Ok(())
}

/// Python-exposed TcpMesh.
#[pyclass(name = "TcpMesh")]
pub struct PyTcpMesh {
    pub inner: std::sync::Mutex<Option<coeus_dist::TcpMesh>>,
}

#[pymethods]
impl PyTcpMesh {
    /// Create a new TcpMesh.
    #[new]
    fn new(rank: usize, size: usize, addresses: Vec<String>, py: Python<'_>) -> PyResult<Self> {
        let addrs: Vec<std::net::SocketAddr> = addresses
            .iter()
            .map(|s| {
                s.parse().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid SocketAddr: {e}"))
                })
            })
            .collect::<Result<_, _>>()?;

        let inner = py.allow_threads(move || coeus_dist::TcpMesh::new(rank, size, &addrs));
        Ok(Self {
            inner: std::sync::Mutex::new(Some(inner)),
        })
    }
}

/// Python-exposed TcpCommunicator.
#[pyclass(name = "TcpCommunicator")]
#[derive(Clone)]
pub struct PyTcpCommunicator {
    pub inner: std::sync::Arc<coeus_dist::TcpCommunicator>,
}

#[pymethods]
impl PyTcpCommunicator {
    /// Create a new TcpCommunicator wrapping a TcpMesh.
    #[new]
    fn new(mesh: &PyTcpMesh) -> PyResult<Self> {
        let mut guard = mesh.inner.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Mutex poisoned: {e}"))
        })?;
        let raw_mesh = guard.take().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "TcpMesh has already been used to construct a Communicator",
            )
        })?;
        Ok(Self {
            inner: std::sync::Arc::new(coeus_dist::TcpCommunicator::new(raw_mesh)),
        })
    }

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
            comm.barrier();
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
            comm.all_reduce::<f64, _, coeus_dist::Sum>(&mut t_val, &backend);
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
            comm.broadcast::<f64, _>(&mut t_val, root, &backend);
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
            comm.all_gather::<f64, _>(&input_tensor, &mut rust_tensors, &backend);
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
            comm.reduce::<f64, _, coeus_dist::Sum>(&mut t_val, root, &backend);
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
            comm.gather::<f64, _>(&input_tensor, &mut rust_tensors, root, &backend);
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
            comm.scatter::<f64, _>(&mut t_val, &rust_tensors, root, &backend);
            (t_val, rust_tensors)
        });

        t_borrow.inner.tensor = t_val;
        Ok(())
    }
}

/// PyCoeus extension module definition.
#[pymodule]
pub fn pycoeus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<nn::PyLinear>()?;
    m.add_class::<tensor::PyStateDict>()?;
    m.add_class::<optim::PyLrScheduler>()?;
    m.add_class::<nn::PyConv1d>()?;
    m.add_class::<nn::PyConv2d>()?;
    m.add_class::<nn::PyConv3d>()?;
    m.add_class::<nn::PyLayerNorm>()?;
    m.add_class::<nn::PyRMSNorm>()?;
    m.add_class::<nn::PyBatchNorm3d>()?;
    m.add_class::<nn::PyAvgPool2d>()?;
    m.add_class::<nn::PyMaxPool2d>()?;
    m.add_class::<nn::PyAvgPool3d>()?;
    m.add_class::<nn::PyMaxPool3d>()?;
    m.add_class::<nn::PyEmbedding>()?;
    m.add_class::<nn::PyDropout>()?;
    m.add_class::<nn::PyBatchNorm1d>()?;
    m.add_class::<nn::PyBatchNorm2d>()?;
    m.add_class::<optim::PySGD>()?;
    m.add_class::<optim::PyAdam>()?;
    m.add_class::<optim::PyAdamW>()?;
    m.add_class::<optim::PyRMSProp>()?;
    m.add_class::<optim::PyAdaGrad>()?;
    m.add_class::<nn::PyGroupNorm>()?;
    m.add_class::<nn::PyInstanceNorm1d>()?;
    m.add_class::<nn::PyInstanceNorm2d>()?;
    m.add_class::<nn::PyMultiHeadAttention>()?;
    m.add_class::<nn::PyRotaryEmbedding>()?;
    m.add_class::<PyMockCommunicator>()?;
    m.add_class::<PyTcpMesh>()?;
    m.add_class::<PyTcpCommunicator>()?;
    m.add_function(wrap_pyfunction!(create_mock_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(mish, m)?)?;
    m.add_function(wrap_pyfunction!(elu, m)?)?;
    m.add_function(wrap_pyfunction!(softplus, m)?)?;
    m.add_function(wrap_pyfunction!(gelu_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(huber_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(sum_axis, m)?)?;
    m.add_function(wrap_pyfunction!(mean_axis, m)?)?;
    m.add_function(wrap_pyfunction!(pad, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    Ok(())
}
