use pyo3::prelude::*;
use crate::tensor::{PyTensor, PyStateDict};

/// Python-exposed Layer Normalization layer.
#[pyclass(name = "LayerNorm")]
pub struct PyLayerNorm {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    pub fn new(py: Python<'_>, normalized_shape: usize, eps: f64) -> PyResult<Self> {
        let ln = coeus_nn::normalization::layernorm::LayerNorm::<f64, coeus_core::MoiraiBackend>::new(normalized_shape, eps);
        let weight = Py::new(py, PyTensor { inner: ln.weight })?;
        let bias = Py::new(py, PyTensor { inner: ln.bias })?;
        Ok(Self { weight, bias, eps })
    }

    /// Forward pass through the LayerNorm layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let eps_val = self.eps;

        let inner = py.allow_threads(move || {
            let ln = coeus_nn::normalization::layernorm::LayerNorm::from_parts(
                w_var,
                b_var,
                eps_val,
            );
            ln.forward(&input_var)
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

/// Python-exposed RMS Normalization layer.
#[pyclass(name = "RMSNorm")]
pub struct PyRMSNorm {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-8))]
    pub fn new(py: Python<'_>, normalized_shape: usize, eps: f64) -> PyResult<Self> {
        let rms = coeus_nn::normalization::rmsnorm::RMSNorm::<f64, coeus_core::MoiraiBackend>::new(normalized_shape, eps);
        let weight = Py::new(py, PyTensor { inner: rms.weight })?;
        Ok(Self { weight, eps })
    }

    /// Forward pass through the RMSNorm layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let eps_val = self.eps;

        let inner = py.allow_threads(move || {
            let rms = coeus_nn::normalization::rmsnorm::RMSNorm::from_parts(
                w_var,
                eps_val,
            );
            rms.forward(&input_var)
        });
        Ok(PyTensor { inner })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        Ok(())
    }
}

/// Python-exposed 1D Batch Normalization layer.
#[pyclass(name = "BatchNorm1d")]
pub struct PyBatchNorm1d {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub running_mean: Py<PyTensor>,
    #[pyo3(get)]
    pub running_var: Py<PyTensor>,
    #[pyo3(get)]
    pub num_features: usize,
    #[pyo3(get)]
    pub eps: f64,
    #[pyo3(get)]
    pub momentum: f64,
}

#[pymethods]
impl PyBatchNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    pub fn new(
        py: Python<'_>,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> PyResult<Self> {
        let rust_bn = coeus_nn::normalization::BatchNorm1d::<f64, coeus_core::MoiraiBackend>::new(
            num_features,
            eps,
            momentum,
        );

        let weight = Py::new(py, PyTensor { inner: rust_bn.weight })?;
        let bias = Py::new(py, PyTensor { inner: rust_bn.bias })?;
        let running_mean = Py::new(py, PyTensor {
            inner: coeus_autograd::Var::new(rust_bn.running_mean.borrow().clone(), false),
        })?;
        let running_var = Py::new(py, PyTensor {
            inner: coeus_autograd::Var::new(rust_bn.running_var.borrow().clone(), false),
        })?;

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_features,
            eps,
            momentum,
        })
    }

    /// Forward pass through the BatchNorm1d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let rm_t = self.running_mean.bind(py).borrow().inner.tensor.clone();
        let rv_t = self.running_var.bind(py).borrow().inner.tensor.clone();
        let input_var = input.inner.clone();

        let (out_var, next_rm, next_rv) = py.allow_threads(move || {
            let bn = coeus_nn::normalization::BatchNorm1d::from_parts(
                self.num_features,
                w_var,
                b_var,
                self.eps,
                self.momentum,
                rm_t,
                rv_t,
            );
            let out = bn.forward(&input_var);
            let rm = bn.running_mean.into_inner();
            let rv = bn.running_var.into_inner();
            (out, rm, rv)
        });

        self.running_mean.bind(py).borrow_mut().inner.tensor = next_rm;
        self.running_var.bind(py).borrow_mut().inner.tensor = next_rv;

        Ok(PyTensor { inner: out_var })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        sd.insert("running_mean", self.running_mean.bind(py).borrow().inner.tensor.clone());
        sd.insert("running_var", self.running_var.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            self.bias.bind(py).borrow_mut().inner.tensor = b.clone();
        }
        if let Some(rm) = state_dict.inner.get("running_mean") {
            self.running_mean.bind(py).borrow_mut().inner.tensor = rm.clone();
        }
        if let Some(rv) = state_dict.inner.get("running_var") {
            self.running_var.bind(py).borrow_mut().inner.tensor = rv.clone();
        }
        Ok(())
    }
}

/// Python-exposed 2D Batch Normalization layer.
#[pyclass(name = "BatchNorm2d")]
pub struct PyBatchNorm2d {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub running_mean: Py<PyTensor>,
    #[pyo3(get)]
    pub running_var: Py<PyTensor>,
    #[pyo3(get)]
    pub num_features: usize,
    #[pyo3(get)]
    pub eps: f64,
    #[pyo3(get)]
    pub momentum: f64,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    pub fn new(
        py: Python<'_>,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> PyResult<Self> {
        let rust_bn = coeus_nn::normalization::BatchNorm2d::<f64, coeus_core::MoiraiBackend>::new(
            num_features,
            eps,
            momentum,
        );

        let weight = Py::new(py, PyTensor { inner: rust_bn.weight })?;
        let bias = Py::new(py, PyTensor { inner: rust_bn.bias })?;
        let running_mean = Py::new(py, PyTensor {
            inner: coeus_autograd::Var::new(rust_bn.running_mean.borrow().clone(), false),
        })?;
        let running_var = Py::new(py, PyTensor {
            inner: coeus_autograd::Var::new(rust_bn.running_var.borrow().clone(), false),
        })?;

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_features,
            eps,
            momentum,
        })
    }

    /// Forward pass through the BatchNorm2d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let rm_t = self.running_mean.bind(py).borrow().inner.tensor.clone();
        let rv_t = self.running_var.bind(py).borrow().inner.tensor.clone();
        let input_var = input.inner.clone();

        let (out_var, next_rm, next_rv) = py.allow_threads(move || {
            let bn = coeus_nn::normalization::BatchNorm2d::from_parts(
                self.num_features,
                w_var,
                b_var,
                self.eps,
                self.momentum,
                rm_t,
                rv_t,
            );
            let out = bn.forward(&input_var);
            let rm = bn.running_mean.into_inner();
            let rv = bn.running_var.into_inner();
            (out, rm, rv)
        });

        self.running_mean.bind(py).borrow_mut().inner.tensor = next_rm;
        self.running_var.bind(py).borrow_mut().inner.tensor = next_rv;

        Ok(PyTensor { inner: out_var })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        sd.insert("running_mean", self.running_mean.bind(py).borrow().inner.tensor.clone());
        sd.insert("running_var", self.running_var.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            self.bias.bind(py).borrow_mut().inner.tensor = b.clone();
        }
        if let Some(rm) = state_dict.inner.get("running_mean") {
            self.running_mean.bind(py).borrow_mut().inner.tensor = rm.clone();
        }
        if let Some(rv) = state_dict.inner.get("running_var") {
            self.running_var.bind(py).borrow_mut().inner.tensor = rv.clone();
        }
        Ok(())
    }
}

/// Python-exposed 3D Batch Normalization layer.
#[pyclass(name = "BatchNorm3d")]
pub struct PyBatchNorm3d {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub running_mean: Py<PyTensor>,
    #[pyo3(get)]
    pub running_var: Py<PyTensor>,
    #[pyo3(get)]
    pub num_features: usize,
    #[pyo3(get)]
    pub eps: f64,
    #[pyo3(get)]
    pub momentum: f64,
}

#[pymethods]
impl PyBatchNorm3d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    pub fn new(
        py: Python<'_>,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> PyResult<Self> {
        let rust_bn = coeus_nn::normalization::BatchNorm3d::<f64, coeus_core::MoiraiBackend>::new(
            num_features,
            eps,
            momentum,
        );

        let weight = Py::new(py, PyTensor { inner: rust_bn.weight })?;
        let bias = Py::new(py, PyTensor { inner: rust_bn.bias })?;
        let running_mean = Py::new(py, PyTensor {
            inner: coeus_autograd::Var::new(rust_bn.running_mean.borrow().clone(), false),
        })?;
        let running_var = Py::new(py, PyTensor {
            inner: coeus_autograd::Var::new(rust_bn.running_var.borrow().clone(), false),
        })?;

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_features,
            eps,
            momentum,
        })
    }

    /// Forward pass through the BatchNorm3d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let rm_t = self.running_mean.bind(py).borrow().inner.tensor.clone();
        let rv_t = self.running_var.bind(py).borrow().inner.tensor.clone();
        let input_var = input.inner.clone();

        let (out_var, next_rm, next_rv) = py.allow_threads(move || {
            let bn = coeus_nn::normalization::BatchNorm3d::from_parts(
                self.num_features,
                w_var,
                b_var,
                self.eps,
                self.momentum,
                rm_t,
                rv_t,
            );
            let out = bn.forward(&input_var);
            let rm = bn.running_mean.into_inner();
            let rv = bn.running_var.into_inner();
            (out, rm, rv)
        });

        self.running_mean.bind(py).borrow_mut().inner.tensor = next_rm;
        self.running_var.bind(py).borrow_mut().inner.tensor = next_rv;

        Ok(PyTensor { inner: out_var })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        sd.insert("running_mean", self.running_mean.bind(py).borrow().inner.tensor.clone());
        sd.insert("running_var", self.running_var.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            self.bias.bind(py).borrow_mut().inner.tensor = b.clone();
        }
        if let Some(rm) = state_dict.inner.get("running_mean") {
            self.running_mean.bind(py).borrow_mut().inner.tensor = rm.clone();
        }
        if let Some(rv) = state_dict.inner.get("running_var") {
            self.running_var.bind(py).borrow_mut().inner.tensor = rv.clone();
        }
        Ok(())
    }
}

/// Python-exposed Group Normalization layer.
///
/// Supported num_groups values at runtime: 1, 2, 4, 8, 16, 32, 64.
/// `num_features` must be divisible by `num_groups`.
#[pyclass(name = "GroupNorm")]
pub struct PyGroupNorm {
    /// Trainable scale (gamma), shape [num_features].
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape [num_features].
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub num_groups: usize,
    #[pyo3(get)]
    pub num_features: usize,
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyGroupNorm {
    #[new]
    #[pyo3(signature = (num_groups, num_features, eps = 1e-5))]
    pub fn new(
        py: Python<'_>,
        num_groups: usize,
        num_features: usize,
        eps: f64,
    ) -> PyResult<Self> {
        // Use G=1 to allocate canonical weight/bias tensors:
        // GroupNorm always initialises weight=ones([num_features]) and bias=zeros([num_features])
        // regardless of G; G=1 divides any positive num_features.
        let gn = coeus_nn::normalization::groupnorm::GroupNorm::<
            f64, coeus_core::MoiraiBackend, 1,
        >::new(num_features, eps);
        let weight = Py::new(py, PyTensor { inner: gn.weight })?;
        let bias   = Py::new(py, PyTensor { inner: gn.bias })?;
        Ok(Self { weight, bias, num_groups, num_features, eps })
    }

    /// Forward pass through GroupNorm.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var     = self.weight.bind(py).borrow().inner.clone();
        let b_var     = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_groups   = self.num_groups;
        let num_features = self.num_features;
        let eps          = self.eps;

        let inner = py.allow_threads(move || {
            // Dispatch to the monomorphized GroupNorm<f64, MoiraiBackend, G>.
            // Each arm constructs a fresh instance then overwrites the public weight/bias
            // fields with the stored parameters before calling forward().
            macro_rules! dispatch_gn {
                ($($g:literal),*) => {
                    match num_groups {
                        $($g => {
                            let mut gn = coeus_nn::normalization::groupnorm::GroupNorm::<
                                f64, coeus_core::MoiraiBackend, $g,
                            >::new(num_features, eps);
                            gn.weight = w_var;
                            gn.bias   = b_var;
                            gn.forward(&input_var)
                        },)*
                        _ => panic!(
                            "PyGroupNorm: unsupported num_groups={num_groups}; \
                             supported: 1,2,4,8,16,32,64"
                        ),
                    }
                }
            }
            dispatch_gn!(1, 2, 4, 8, 16, 32, 64)
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

/// Python-exposed Instance Normalization 1D layer.
///
/// Input shape: `[N, C]` or `[N, C, L]`.
#[pyclass(name = "InstanceNorm1d")]
pub struct PyInstanceNorm1d {
    /// Trainable scale (gamma), shape [num_features].
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape [num_features].
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
            f64, coeus_core::MoiraiBackend,
        >::new(num_features, eps);
        let weight = Py::new(py, PyTensor { inner: inst.weight })?;
        let bias   = Py::new(py, PyTensor { inner: inst.bias })?;
        Ok(Self { weight, bias, num_features, eps })
    }

    /// Forward pass through InstanceNorm1d.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var     = self.weight.bind(py).borrow().inner.clone();
        let b_var     = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_features = self.num_features;
        let eps          = self.eps;

        let inner = py.allow_threads(move || {
            let mut inst = coeus_nn::normalization::instancenorm::InstanceNorm1d::<
                f64, coeus_core::MoiraiBackend,
            >::new(num_features, eps);
            inst.weight = w_var;
            inst.bias   = b_var;
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
    /// Trainable scale (gamma), shape [num_features].
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape [num_features].
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
            f64, coeus_core::MoiraiBackend,
        >::new(num_features, eps);
        let weight = Py::new(py, PyTensor { inner: inst.weight })?;
        let bias   = Py::new(py, PyTensor { inner: inst.bias })?;
        Ok(Self { weight, bias, num_features, eps })
    }

    /// Forward pass through InstanceNorm2d.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var     = self.weight.bind(py).borrow().inner.clone();
        let b_var     = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_features = self.num_features;
        let eps          = self.eps;

        let inner = py.allow_threads(move || {
            let mut inst = coeus_nn::normalization::instancenorm::InstanceNorm2d::<
                f64, coeus_core::MoiraiBackend,
            >::new(num_features, eps);
            inst.weight = w_var;
            inst.bias   = b_var;
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
