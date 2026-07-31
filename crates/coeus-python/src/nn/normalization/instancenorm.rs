use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
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
    /// Number of channels being normalized per sample.
    #[pyo3(get)]
    pub num_features: usize,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyInstanceNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5))]
    /// Create an InstanceNorm1d layer for `num_features` input channels.
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
        inner.map(PyTensor::from_var).map_err(map_module_error)
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

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py), self.bias.clone_ref(py)]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        self.bias.bind(py).borrow().zero_grad();
    }
}

/// Python-exposed Instance Normalization 3D layer.
///
/// Input shape: `[N, C, D, H, W]`.
#[pyclass(name = "InstanceNorm3d")]
pub struct PyInstanceNorm3d {
    /// Trainable scale (gamma), shape `[num_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape `[num_features]`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    /// Number of channels being normalized per sample.
    #[pyo3(get)]
    pub num_features: usize,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyInstanceNorm3d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5))]
    /// Create an InstanceNorm3d layer for `num_features` input channels.
    pub fn new(py: Python<'_>, num_features: usize, eps: f64) -> PyResult<Self> {
        let inst = coeus_nn::normalization::instancenorm::InstanceNorm3d::<
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

    /// Forward pass through InstanceNorm3d.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_features = self.num_features;
        let eps = self.eps;

        let inner = py.allow_threads(move || {
            let mut inst = coeus_nn::normalization::instancenorm::InstanceNorm3d::<
                f64,
                coeus_core::MoiraiBackend,
            >::new(num_features, eps);
            inst.weight = w_var;
            inst.bias = b_var;
            inst.forward(&input_var)
        });
        inner.map(PyTensor::from_var).map_err(map_module_error)
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

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py), self.bias.clone_ref(py)]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        self.bias.bind(py).borrow().zero_grad();
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
    /// Number of channels being normalized per sample.
    #[pyo3(get)]
    pub num_features: usize,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyInstanceNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5))]
    /// Create an InstanceNorm2d layer for `num_features` input channels.
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
        inner.map(PyTensor::from_var).map_err(map_module_error)
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

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py), self.bias.clone_ref(py)]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        self.bias.bind(py).borrow().zero_grad();
    }
}
