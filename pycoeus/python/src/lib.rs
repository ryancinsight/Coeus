use pyo3::prelude::*;
use pyo3::types::PyList;
use coeus_tensor::{Tensor, Dtype, Backend, CpuBackend};
use coeus_nn::{Linear, Conv1d, BatchNorm1d /* add normalization, pooling, attention */};

// Stub generic via f32 for bindings (full T,B defer maturin feature)
#[pyclass]
struct PyTensor {
    inner: Tensor<f32, CpuBackend>,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: &PyList, shape: Vec<usize>) -> PyResult<Self> {
        let py = Python::with_gil(|py| {
            let data_vec: Vec<f32> = data.extract(py)?;
            let backend = CpuBackend::default();
            let tensor = Tensor::from_vec(backend, data_vec, shape).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Tensor create error: {}", e)))?;
            Ok(Self { inner: tensor })
        })
    }

    fn add(&self, other: &Self) -> PyResult<Self> {
        Ok(Self { inner: (&self.inner + &other.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Add error: {}", e)))? })
    }

    // Add mul/sub/div etc. similar

    fn forward_linear(&self, linear: &PyLinear) -> PyResult<Self> {
        linear.forward(&self.inner).map(|t| Self { inner: t }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Linear forward: {}", e)))
    }

    // Add requires_grad, grad access via pyproperty
}

#[pyclass]
struct PyLinear(Linear<f32, CpuBackend>);

#[pymethods]
impl PyLinear {
    #[new]
    fn new(input_size: usize, output_size: usize) -> PyResult<Self> {
        let backend = CpuBackend::default();
        let linear = Linear::new(input_size, output_size, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Linear create: {}", e)))?;
        Ok(Self(linear))
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.0.forward(&input.inner).map(|t| PyTensor { inner: t }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Forward: {}", e)))
    }
}

// Similar for Conv1d
#[pyclass]
struct PyConv1d(/* impl Conv1d<f32,CpuBackend> */);

#[pymethods]
impl PyConv1d {
    #[new]
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> PyResult<Self> {
        // create with backend, .expect or map_err PyErr
        Ok(Self(/* ... */))
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // conv forward .expect map_err
    }
}

// Add BatchNorm1d, AdaptiveAvgPool1d/2d, MultiheadAttention etc. for 50% API (normalization/pooling/attention)

// py_module! for nn
#[pymodule]
fn pycoeus_nn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyConv1d>()?;
    // add others: BatchNorm1d, etc.
    Ok(())
}

// In build.rs or pyproject: maturin develop for bindings
