use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Python-exposed Unfold2d layer (sliding-window extraction).
///
/// Equivalent to `torch.nn.Unfold(kernel_size, dilation, padding, stride)`.
/// Extracts `[N, C, H, W]` → `[N, C*kH*kW, H_out*W_out]`.
#[pyclass(name = "Unfold2d")]
pub struct PyUnfold2d {
    /// Square kernel (or height when using per-axis constructor).
    #[pyo3(get)]
    pub kernel_h: usize,
    /// Kernel width.
    #[pyo3(get)]
    pub kernel_w: usize,
    /// Vertical stride.
    #[pyo3(get)]
    pub stride_h: usize,
    /// Horizontal stride.
    #[pyo3(get)]
    pub stride_w: usize,
    /// Vertical padding.
    #[pyo3(get)]
    pub padding_h: usize,
    /// Horizontal padding.
    #[pyo3(get)]
    pub padding_w: usize,
    /// Vertical dilation.
    #[pyo3(get)]
    pub dilation_h: usize,
    /// Horizontal dilation.
    #[pyo3(get)]
    pub dilation_w: usize,
}

#[pymethods]
impl PyUnfold2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = 1, padding = 0, dilation = 1))]
    /// Create an `Unfold2d` with a square kernel and equal h/w hyperparameters.
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: stride,
            stride_w: stride,
            padding_h: padding,
            padding_w: padding,
            dilation_h: dilation,
            dilation_w: dilation,
        }
    }

    /// Forward pass: `[N, C, H, W]` → `[N, C*kH*kW, H_out*W_out]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let sh = self.stride_h;
        let sw = self.stride_w;
        let ph = self.padding_h;
        let pw = self.padding_w;
        let dh = self.dilation_h;
        let dw = self.dilation_w;
        let m = coeus_nn::Unfold2d::<f64, coeus_core::MoiraiBackend>::with_params(
            kh, kw, sh, sw, ph, pw, dh, dw,
        );
        let inner = py.allow_threads(move || m.forward(&input_var));
        Ok(PyTensor::from_var(inner))
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op — Unfold2d has no parameters).
    pub fn zero_grad(&self) {}
}

/// Python-exposed Fold2d layer (inverse of Unfold2d).
///
/// Equivalent to `torch.nn.Fold(output_size, kernel_size, ...)`.
/// Accumulates `[N, C*kH*kW, H_out*W_out]` → `[N, C, output_h, output_w]`.
#[pyclass(name = "Fold2d")]
pub struct PyFold2d {
    /// Target output height.
    #[pyo3(get)]
    pub output_h: usize,
    /// Target output width.
    #[pyo3(get)]
    pub output_w: usize,
    /// Kernel height.
    #[pyo3(get)]
    pub kernel_h: usize,
    /// Kernel width.
    #[pyo3(get)]
    pub kernel_w: usize,
    /// Vertical stride.
    #[pyo3(get)]
    pub stride_h: usize,
    /// Horizontal stride.
    #[pyo3(get)]
    pub stride_w: usize,
    /// Vertical padding.
    #[pyo3(get)]
    pub padding_h: usize,
    /// Horizontal padding.
    #[pyo3(get)]
    pub padding_w: usize,
    /// Vertical dilation.
    #[pyo3(get)]
    pub dilation_h: usize,
    /// Horizontal dilation.
    #[pyo3(get)]
    pub dilation_w: usize,
}

#[pymethods]
impl PyFold2d {
    #[new]
    #[pyo3(signature = (output_h, output_w, kernel_size, stride = 1, padding = 0, dilation = 1))]
    /// Create a `Fold2d` with a square kernel and equal h/w hyperparameters.
    pub fn new(
        output_h: usize,
        output_w: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            output_h,
            output_w,
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: stride,
            stride_w: stride,
            padding_h: padding,
            padding_w: padding,
            dilation_h: dilation,
            dilation_w: dilation,
        }
    }

    /// Forward pass: `[N, C*kH*kW, H_out*W_out]` → `[N, C, output_h, output_w]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let oh = self.output_h;
        let ow = self.output_w;
        let kh = self.kernel_h;
        let sh = self.stride_h;
        let ph = self.padding_h;
        let dh = self.dilation_h;
        let m = coeus_nn::Fold2d::<f64, coeus_core::MoiraiBackend>::new(oh, ow, kh, sh, ph, dh);
        let inner = py.allow_threads(move || m.forward(&input_var));
        Ok(PyTensor::from_var(inner))
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op — Fold2d has no parameters).
    pub fn zero_grad(&self) {}
}

// ── Unfold1d ──────────────────────────────────────────────────────────────────

/// Python-exposed Unfold1d layer (1D sliding-window extraction).
///
/// Extracts `[N, C, L]` → `[N, C*kernel_size, L_out]`.
/// The Rust equivalent does not have a direct PyTorch module API (PyTorch uses
/// `nn.Unfold` only in 2D), so parity is verified against the manual
/// `unfold` tensor method: `x.unfold(dim, size, step)`.
#[pyclass(name = "Unfold1d")]
pub struct PyUnfold1d {
    /// Sliding window length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Window stride.
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding on each side.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyUnfold1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = 1, padding = 0, dilation = 1))]
    /// Create an `Unfold1d` layer.
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass: `[N, C, L]` → `[N, C*kernel_size, L_out]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let d = self.dilation;
        let m = coeus_nn::Unfold1d::<f64, coeus_core::MoiraiBackend>::new(k, s, p, d);
        let inner = py.allow_threads(move || m.forward(&input_var));
        Ok(PyTensor::from_var(inner))
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op — Unfold1d has no parameters).
    pub fn zero_grad(&self) {}
}

/// Python-exposed Fold1d (col2im) layer — the 1D inverse of Unfold1d.
///
/// Accumulates `[N, C*kernel_size, L_out]` back into `[N, C, output_size]`,
/// summing overlapping contributions. Differentiable (backward is unfold1d).
#[pyclass(name = "Fold1d")]
pub struct PyFold1d {
    /// Target output length.
    #[pyo3(get)]
    pub output_size: usize,
    /// Sliding window length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Window stride.
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding on each side.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyFold1d {
    #[new]
    #[pyo3(signature = (output_size, kernel_size, stride = 1, padding = 0, dilation = 1))]
    /// Create a `Fold1d` layer.
    pub fn new(
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass: `[N, C*kernel_size, L_out]` → `[N, C, output_size]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let (os, k, s, p, d) = (
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
        let m = coeus_nn::Fold1d::<f64, coeus_core::MoiraiBackend>::new(os, k, s, p, d);
        let inner = py.allow_threads(move || m.forward(&input_var));
        Ok(PyTensor::from_var(inner))
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op — Fold1d has no parameters).
    pub fn zero_grad(&self) {}
}
