use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

/// Python-exposed 1D Batch Normalization layer.
#[pyclass(name = "BatchNorm1d")]
pub struct PyBatchNorm1d {
    /// Learnable scale (gamma), shape `[num_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Learnable shift (beta), shape `[num_features]`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    /// Running mean tracked during training.
    #[pyo3(get)]
    pub running_mean: Py<PyTensor>,
    /// Running variance tracked during training.
    #[pyo3(get)]
    pub running_var: Py<PyTensor>,
    /// Number of features (channels) being normalized.
    #[pyo3(get)]
    pub num_features: usize,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
    /// Exponential moving-average factor for running statistics.
    #[pyo3(get)]
    pub momentum: f64,
    /// Training-mode flag: `forward` uses batch statistics and updates the
    /// running stats when true, and normalizes with the frozen running stats
    /// when false (PyTorch `Module.train`/`eval` contract).
    #[pyo3(get)]
    pub is_training: bool,
}

#[pymethods]
impl PyBatchNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    /// Create a BatchNorm1d layer for `num_features` input channels.
    pub fn new(py: Python<'_>, num_features: usize, eps: f64, momentum: f64) -> PyResult<Self> {
        let rust_bn = coeus_nn::normalization::BatchNorm1d::<f64, coeus_core::MoiraiBackend>::new(
            num_features,
            eps,
            momentum,
        );

        let weight = Py::new(
            py,
            PyTensor {
                inner: rust_bn.weight,
            },
        )?;
        let bias = Py::new(
            py,
            PyTensor {
                inner: rust_bn.bias,
            },
        )?;
        let running_mean = Py::new(
            py,
            PyTensor {
                inner: coeus_autograd::Var::new(rust_bn.running_mean.borrow().clone(), false),
            },
        )?;
        let running_var = Py::new(
            py,
            PyTensor {
                inner: coeus_autograd::Var::new(rust_bn.running_var.borrow().clone(), false),
            },
        )?;

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_features,
            eps,
            momentum,
            is_training: true,
        })
    }

    /// Switch to training mode (batch statistics + running-stat updates).
    #[pyo3(signature = (mode = true))]
    pub fn train(&mut self, mode: bool) {
        self.is_training = mode;
    }

    /// Switch to evaluation mode: `forward` uses the frozen running statistics
    /// and stops updating them (`torch.nn.Module.eval`).
    pub fn eval(&mut self) {
        self.is_training = false;
    }

    /// Forward pass through the BatchNorm1d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        // Eval mode normalizes with the frozen running statistics and does not
        // update them (torch semantics); training mode uses batch statistics.
        if !self.is_training {
            return self.eval_forward(input, py);
        }
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

        let out_var = out_var.map_err(map_module_error)?;
        self.running_mean.bind(py).borrow_mut().inner.tensor = next_rm;
        self.running_var.bind(py).borrow_mut().inner.tensor = next_rv;

        Ok(PyTensor { inner: out_var })
    }

    /// Eval-mode forward: normalizes using `running_mean`/`running_var`, does not update them.
    pub fn eval_forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let rm_t = self.running_mean.bind(py).borrow().inner.tensor.clone();
        let rv_t = self.running_var.bind(py).borrow().inner.tensor.clone();
        let input_var = input.inner.clone();
        let out_var = py.allow_threads(move || {
            let mut bn = coeus_nn::normalization::BatchNorm1d::from_parts(
                self.num_features,
                w_var,
                b_var,
                self.eps,
                self.momentum,
                rm_t,
                rv_t,
            );
            bn.is_training = false;
            bn.forward(&input_var)
        });
        out_var
            .map(|inner| PyTensor { inner })
            .map_err(map_module_error)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        sd.insert(
            "running_mean",
            self.running_mean.bind(py).borrow().inner.tensor.clone(),
        );
        sd.insert(
            "running_var",
            self.running_var.bind(py).borrow().inner.tensor.clone(),
        );
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
