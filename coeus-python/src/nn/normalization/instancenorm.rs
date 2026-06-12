use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed Instance Normalization 1D layer.
///
/// Input shape: `[N, C]` or `[N, C, L]`.
#[pyclass(name = "InstanceNorm1d")]
pub struct PyInstanceNorm1d {
    /// Trainable scale (gamma), shape `[num_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape `[num_features]`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub num_features: usize,
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyInstanceNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5))]
    pub fn new(py: Python<'_>, num_features: usize, eps: f64) -> PyResult<Self> {
        let inst = coeus_nn::normalization::instancenorm::InstanceNorm1d::<
            f64,
            coeus_core::MoiraiBackend,
        >::new(num_features, eps);
        let weight = Py::new(py, PyTensor { inner: inst.weight })?;
        let bias = Py::new(py, PyTensor { inner: inst.bias })?;
        Ok(Self {
            weight,
            bias,
            num_features,
            eps,
        })
    }

    /// Forward pass through InstanceNorm1d.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_features = self.num_features;
        let eps = self.eps;

        let inner = py.allow_threads(move || {
            let mut inst = coeus_nn::normalization::instancenorm::InstanceNorm1d::<
                f64,
                coeus_core::MoiraiBackend,
            >::new(num_features, eps);
            inst.weight = w_var;
            inst.bias = b_var;
            inst.forward(&input_var)
        });
        Ok(PyTensor { inner })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            self.bias.bind(py).borrow_mut().inner.tensor = b.clone();
        }
        Ok(())
    }
}

/// Python-exposed Instance Normalization 2D layer.
///
/// Input shape: `[N, C, H, W]`.
#[pyclass(name = "InstanceNorm2d")]
pub struct PyInstanceNorm2d {
    /// Trainable scale (gamma), shape `[num_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape `[num_features]`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub num_features: usize,
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyInstanceNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5))]
    pub fn new(py: Python<'_>, num_features: usize, eps: f64) -> PyResult<Self> {
        let inst = coeus_nn::normalization::instancenorm::InstanceNorm2d::<
            f64,
            coeus_core::MoiraiBackend,
        >::new(num_features, eps);
        let weight = Py::new(py, PyTensor { inner: inst.weight })?;
        let bias = Py::new(py, PyTensor { inner: inst.bias })?;
        Ok(Self {
            weight,
            bias,
            num_features,
            eps,
        })
    }

    /// Forward pass through InstanceNorm2d.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_features = self.num_features;
        let eps = self.eps;

        let inner = py.allow_threads(move || {
            let mut inst = coeus_nn::normalization::instancenorm::InstanceNorm2d::<
                f64,
                coeus_core::MoiraiBackend,
            >::new(num_features, eps);
            inst.weight = w_var;
            inst.bias = b_var;
            inst.forward(&input_var)
        });
        Ok(PyTensor { inner })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            self.bias.bind(py).borrow_mut().inner.tensor = b.clone();
        }
        Ok(())
    }
}
