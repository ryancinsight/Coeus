use crate::tensor::{PyStateDict, PyTensor};
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
    #[allow(clippy::too_many_arguments)]
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
        Ok(PyTensor::from_var(inner))
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

/// Python-exposed 2D Convolution layer.
#[pyclass(name = "Conv2d")]
pub struct PyConv2d {
    /// Learnable convolution weight, shape `[out_channels, in_channels, kH, kW]`.
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
    /// Square kernel side length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Stride of the convolution.
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding applied to all spatial sides.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = true))]
    #[allow(clippy::too_many_arguments)]
    /// Create a Conv2d layer with the specified parameters.
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
        let rust_conv = coeus_nn::conv::Conv2d::with_params(
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

    /// Forward pass through the Conv2d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let input_var = input.inner.clone();

        let rust_conv = coeus_nn::conv::Conv2d::from_vars(
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
        Ok(PyTensor::from_var(inner))
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

/// Python-exposed 3D Convolution layer.
#[pyclass(name = "Conv3d")]
pub struct PyConv3d {
    /// Learnable convolution weight, shape `[out_channels, in_channels, kD, kH, kW]`.
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
    /// Cubic kernel side length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Stride of the convolution.
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding applied to all spatial sides.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyConv3d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = true))]
    #[allow(clippy::too_many_arguments)]
    /// Create a Conv3d layer with the specified parameters.
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
        let rust_conv = coeus_nn::conv::Conv3d::with_params(
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

    /// Forward pass through the Conv3d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let input_var = input.inner.clone();

        let rust_conv = coeus_nn::conv::Conv3d::from_vars(
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
        Ok(PyTensor::from_var(inner))
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

// ── ConvTranspose1d ─────────────────────────────────────────────────────────

/// Python-exposed 1-D Transposed Convolution layer.
#[pyo3::pyclass(name = "ConvTranspose1d")]
pub struct PyConvTranspose1d {
    /// Learnable transposed-convolution weight.
    #[pyo3(get)]
    pub weight: pyo3::Py<PyTensor>,
    /// Optional learnable bias, shape `[out_channels]`.
    #[pyo3(get)]
    pub bias: Option<pyo3::Py<PyTensor>>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel width.
    pub kernel_size: usize,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Input-side padding removed from output.
    pub padding: usize,
    /// Additional output padding for shape disambiguation.
    pub output_padding: usize,
    /// Dilation factor.
    pub dilation: usize,
}

#[pyo3::pymethods]
impl PyConvTranspose1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=true))]
    #[allow(clippy::too_many_arguments)]
    /// Create a ConvTranspose1d layer with the specified parameters.
    pub fn new(
        py: pyo3::Python<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
    ) -> pyo3::PyResult<Self> {
        let rust = coeus_nn::conv::ConvTranspose1d::<f64, coeus_core::MoiraiBackend>::with_params(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            bias,
        );
        let weight = pyo3::Py::new(py, PyTensor { inner: rust.weight })?;
        let bias = if let Some(b) = rust.bias {
            Some(pyo3::Py::new(py, PyTensor { inner: b })?)
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
            output_padding,
            dilation,
        })
    }
    /// Forward pass through the ConvTranspose1d layer.
    pub fn forward(&self, input: &PyTensor, py: pyo3::Python<'_>) -> pyo3::PyResult<PyTensor> {
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x_var = input.inner.clone();
        let (s, p, op, d) = (
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        );
        let inner = py.allow_threads(move || {
            let bk = coeus_core::MoiraiBackend::new();
            let l = x_var.tensor.shape()[2];
            let l_out = coeus_ops::conv_transpose::conv_transpose1d_output_len(
                l,
                w_var.tensor.shape()[2],
                s,
                p,
                op,
                d,
            );
            let n = x_var.tensor.shape()[0];
            let c_out = w_var.tensor.shape()[1];
            let mut out_tensor = coeus_tensor::Tensor::zeros_on([n, c_out, l_out], &bk);
            let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
            use coeus_ops::ConvOps;
            bk.conv_transpose1d(
                x_var.tensor.storage(),
                x_var.tensor.layout(),
                w_var.tensor.storage(),
                w_var.tensor.layout(),
                b_var.as_ref().map(|b| b.tensor.storage()),
                s,
                p,
                op,
                d,
                out_storage,
                out_layout,
            );
            coeus_autograd::conv_transpose1d(&x_var, &w_var, &b_var, out_tensor, s, p, op, d)
        });
        Ok(PyTensor::from_var(inner))
    }
    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: pyo3::Python<'_>) -> Vec<pyo3::Py<PyTensor>> {
        let mut v = vec![self.weight.clone_ref(py)];
        if let Some(ref b) = self.bias {
            v.push(b.clone_ref(py));
        }
        v
    }
}

// ── ConvTranspose2d ─────────────────────────────────────────────────────────

/// Python-exposed 2-D Transposed Convolution layer.
#[pyo3::pyclass(name = "ConvTranspose2d")]
pub struct PyConvTranspose2d {
    /// Learnable transposed-convolution weight.
    #[pyo3(get)]
    pub weight: pyo3::Py<PyTensor>,
    /// Optional learnable bias, shape `[out_channels]`.
    #[pyo3(get)]
    pub bias: Option<pyo3::Py<PyTensor>>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Square kernel side length.
    pub kernel_size: usize,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Input-side padding removed from output.
    pub padding: usize,
    /// Additional output padding for shape disambiguation.
    pub output_padding: usize,
    /// Dilation factor.
    pub dilation: usize,
}

#[pyo3::pymethods]
impl PyConvTranspose2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=true))]
    #[allow(clippy::too_many_arguments)]
    /// Create a ConvTranspose2d layer with the specified parameters.
    pub fn new(
        py: pyo3::Python<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
    ) -> pyo3::PyResult<Self> {
        let rust = coeus_nn::conv::ConvTranspose2d::<f64, coeus_core::MoiraiBackend>::with_params(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            bias,
        );
        let weight = pyo3::Py::new(py, PyTensor { inner: rust.weight })?;
        let bias = if let Some(b) = rust.bias {
            Some(pyo3::Py::new(py, PyTensor { inner: b })?)
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
            output_padding,
            dilation,
        })
    }
    /// Forward pass through the ConvTranspose2d layer.
    pub fn forward(&self, input: &PyTensor, py: pyo3::Python<'_>) -> pyo3::PyResult<PyTensor> {
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x_var = input.inner.clone();
        let (s, p, op, d) = (
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        );
        let inner = py.allow_threads(move || {
            let bk = coeus_core::MoiraiBackend::new();
            let h = x_var.tensor.shape()[2];
            let w = x_var.tensor.shape()[3];
            let kh = w_var.tensor.shape()[2];
            let kw = w_var.tensor.shape()[3];
            let (h_out, w_out) =
                coeus_ops::conv_transpose::conv_transpose2d_output_dims(h, w, kh, kw, s, p, op, d);
            let n = x_var.tensor.shape()[0];
            let c_out = w_var.tensor.shape()[1];
            let mut out_tensor = coeus_tensor::Tensor::zeros_on([n, c_out, h_out, w_out], &bk);
            let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
            use coeus_ops::ConvOps;
            bk.conv_transpose2d(
                x_var.tensor.storage(),
                x_var.tensor.layout(),
                w_var.tensor.storage(),
                w_var.tensor.layout(),
                b_var.as_ref().map(|b| b.tensor.storage()),
                s,
                p,
                op,
                d,
                out_storage,
                out_layout,
            );
            coeus_autograd::conv_transpose2d(&x_var, &w_var, &b_var, out_tensor, s, p, op, d)
        });
        Ok(PyTensor::from_var(inner))
    }
    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: pyo3::Python<'_>) -> Vec<pyo3::Py<PyTensor>> {
        let mut v = vec![self.weight.clone_ref(py)];
        if let Some(ref b) = self.bias {
            v.push(b.clone_ref(py));
        }
        v
    }
}

// ── ConvTranspose3d ─────────────────────────────────────────────────────────

/// Python-exposed 3-D Transposed Convolution layer.
#[pyo3::pyclass(name = "ConvTranspose3d")]
pub struct PyConvTranspose3d {
    /// Learnable transposed-convolution weight.
    #[pyo3(get)]
    pub weight: pyo3::Py<PyTensor>,
    /// Optional learnable bias, shape `[out_channels]`.
    #[pyo3(get)]
    pub bias: Option<pyo3::Py<PyTensor>>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Cubic kernel side length.
    pub kernel_size: usize,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Input-side padding removed from output.
    pub padding: usize,
    /// Additional output padding for shape disambiguation.
    pub output_padding: usize,
    /// Dilation factor.
    pub dilation: usize,
}

#[pyo3::pymethods]
impl PyConvTranspose3d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=true))]
    #[allow(clippy::too_many_arguments)]
    /// Create a ConvTranspose3d layer with the specified parameters.
    pub fn new(
        py: pyo3::Python<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
    ) -> pyo3::PyResult<Self> {
        let rust = coeus_nn::conv::ConvTranspose3d::<f64, coeus_core::MoiraiBackend>::with_params(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            bias,
        );
        let weight = pyo3::Py::new(py, PyTensor { inner: rust.weight })?;
        let bias = if let Some(b) = rust.bias {
            Some(pyo3::Py::new(py, PyTensor { inner: b })?)
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
            output_padding,
            dilation,
        })
    }
    /// Forward pass through the ConvTranspose3d layer.
    pub fn forward(&self, input: &PyTensor, py: pyo3::Python<'_>) -> pyo3::PyResult<PyTensor> {
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x_var = input.inner.clone();
        let (s, p, op, d) = (
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        );
        let inner = py.allow_threads(move || {
            let bk = coeus_core::MoiraiBackend::new();
            let d_in = x_var.tensor.shape()[2];
            let h_in = x_var.tensor.shape()[3];
            let w_in = x_var.tensor.shape()[4];
            let kd = w_var.tensor.shape()[2];
            let kh = w_var.tensor.shape()[3];
            let kw = w_var.tensor.shape()[4];
            let (d_out, h_out, w_out) = coeus_ops::conv_transpose::conv_transpose3d_output_dims(
                d_in, h_in, w_in, kd, kh, kw, s, p, op, d,
            );
            let n = x_var.tensor.shape()[0];
            let c_out = w_var.tensor.shape()[1];
            let mut out_tensor =
                coeus_tensor::Tensor::zeros_on([n, c_out, d_out, h_out, w_out], &bk);
            let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
            use coeus_ops::backend_ops::ConvTranspose3dOps;
            bk.conv_transpose3d(
                x_var.tensor.storage(),
                x_var.tensor.layout(),
                w_var.tensor.storage(),
                w_var.tensor.layout(),
                b_var.as_ref().map(|b| b.tensor.storage()),
                s,
                p,
                op,
                d,
                out_storage,
                out_layout,
            );
            coeus_autograd::conv_transpose3d(&x_var, &w_var, &b_var, out_tensor, s, p, op, d)
        });
        Ok(PyTensor::from_var(inner))
    }
    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: pyo3::Python<'_>) -> Vec<pyo3::Py<PyTensor>> {
        let mut v = vec![self.weight.clone_ref(py)];
        if let Some(ref b) = self.bias {
            v.push(b.clone_ref(py));
        }
        v
    }
}
