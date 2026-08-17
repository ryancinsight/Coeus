use crate::{init::map_initialization_error, nn::error::map_backend_error, tensor::PyTensor};

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
    #[expect(
        clippy::too_many_arguments,
        reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
    )]
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
        )
        .map_err(map_initialization_error)?;
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
        let inner = py
            .allow_threads(move || {
                let bk = coeus_core::MoiraiBackend::new();
                let h = x_var.tensor.shape()[2];
                let w = x_var.tensor.shape()[3];
                let kh = w_var.tensor.shape()[2];
                let kw = w_var.tensor.shape()[3];
                let (h_out, w_out) = coeus_ops::conv_transpose::conv_transpose2d_output_dims(
                    h, w, kh, kw, s, p, op, d,
                );
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
                )?;
                Ok(coeus_autograd::conv_transpose2d(
                    &x_var, &w_var, &b_var, out_tensor, s, p, op, d,
                ))
            })
            .map_err(map_backend_error)?;
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
