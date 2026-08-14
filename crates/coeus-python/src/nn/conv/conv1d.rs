use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

/// Python-exposed 1D Convolution layer.
#[pyclass(name = "Conv1d")]
pub struct PyConv1d {
    /// Learnable convolution weight, shape `[out_channels, in_channels, kernel_size]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Optional learnable bias, shape `[out_channels]`.
    #[pyo3(get)]
    pub bias: Option<Py<PyTensor>>,
    /// Number of input channels.
    #[pyo3(get)]
    pub in_channels: usize,
    /// Number of output channels.
    #[pyo3(get)]
    pub out_channels: usize,
    /// Kernel width.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Stride of the convolution.
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding applied to both sides.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyConv1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = true))]
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    /// Create a Conv1d layer with the specified parameters.
    pub fn new(
        py: Python<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> PyResult<Self> {
        let rust_conv = coeus_nn::conv::Conv1d::with_params(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
        );

        let weight = Py::new(
            py,
            PyTensor {
                inner: rust_conv.weight,
            },
        )?;
        let bias = if let Some(b) = rust_conv.bias {
            Some(Py::new(py, PyTensor { inner: b })?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Forward pass through the Conv1d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let input_var = input.inner.clone();

        let rust_conv = coeus_nn::conv::Conv1d::from_vars(
            w_var,
            b_var,
            coeus_nn::conv::ConvParams::new(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            ),
        );

        let inner = py.allow_threads(move || rust_conv.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.bias {
            sd.insert("bias", b.bind(py).borrow().inner.tensor.clone());
        }
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(ref b) = self.bias {
            if let Some(bias_tensor) = state_dict.inner.get("bias") {
                b.bind(py).borrow_mut().inner.tensor = bias_tensor.clone();
            }
        }
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut params = vec![self.weight.clone_ref(py)];
        if let Some(ref b) = self.bias {
            params.push(b.clone_ref(py));
        }
        params
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        if let Some(ref b) = self.bias {
            b.bind(py).borrow().zero_grad();
        }
    }
}
