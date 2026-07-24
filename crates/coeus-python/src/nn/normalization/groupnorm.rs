use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed Group Normalization layer.
///
/// Mirrors `torch.nn.GroupNorm(num_groups, num_channels, eps=1e-5)`.  The
/// constructor keyword is `num_channels` (PyTorch); the attribute exposed on
/// the bound object is `num_features` (internal Rust-core field name, kept for
/// backward compatibility with existing checkpoint/state_dict consumers).
/// Supported num_groups values at runtime: 1, 2, 4, 8, 16, 32, 64.
/// `num_features` must be divisible by `num_groups`.
#[pyclass(name = "GroupNorm")]
pub struct PyGroupNorm {
    /// Trainable scale (gamma), shape `[num_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Trainable shift (beta), shape `[num_features]`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    /// Number of groups to divide the channels into.
    #[pyo3(get)]
    pub num_groups: usize,
    /// Total number of channels (must be divisible by `num_groups`).
    #[pyo3(get)]
    pub num_features: usize,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyGroupNorm {
    #[new]
    /// Create a GroupNorm layer dividing `num_channels` channels into `num_groups` groups.
    ///
    /// Mirrors `torch.nn.GroupNorm(num_groups, num_channels, eps=1e-5)` keyword
    /// argument conventions.  (`num_features` is the internal Rust-core field name
    /// and the public attribute exposed by `pycoe3 GroupNorm`; the constructor
    /// keyword matches PyTorch's `num_channels`.)
    #[pyo3(signature = (num_groups, num_channels, eps = 1e-5))]
    pub fn new(py: Python<'_>, num_groups: usize, num_channels: usize, eps: f64) -> PyResult<Self> {
        // Use G=1 to allocate canonical weight/bias tensors:
        // GroupNorm always initialises weight=ones([num_features]) and bias=zeros([num_features])
        // regardless of G; G=1 divides any positive num_features.
        let gn =
            coeus_nn::normalization::groupnorm::GroupNorm::<f64, coeus_core::MoiraiBackend, 1>::new(
                num_channels,
                eps,
            );
        let weight = Py::new(py, PyTensor { inner: gn.weight })?;
        let bias = Py::new(py, PyTensor { inner: gn.bias })?;
        Ok(Self {
            weight,
            bias,
            num_groups,
            num_features: num_channels,
            eps,
        })
    }

    /// Forward pass through GroupNorm.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_groups = self.num_groups;
        let num_features = self.num_features;
        let eps = self.eps;

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
        Ok(PyTensor::from_var(inner))
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
