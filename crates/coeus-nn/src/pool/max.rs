use crate::module::Module;
use crate::pool::out_dim;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

// ── MaxPool2d ──

/// 2D max pooling layer.
#[derive(Clone)]
pub struct MaxPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Pooling window side length.
    pub kernel_size: usize,
    /// Stride along H and W dimensions.
    pub stride: usize,
    /// Zero-padding applied to all spatial sides.
    pub padding: usize,
    /// Spacing between pooling window elements.
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> MaxPool2d<T, B> {
    /// Create with stride equal to kernel_size, no padding, no dilation.
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    /// Create with explicit hyperparameters.
    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for MaxPool2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        let backend = B::default();

        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let h = input.tensor.shape()[2];
        let w = input.tensor.shape()[3];
        let h_out = out_dim(
            h,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        let w_out = out_dim(
            w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        assert!(
            h_out > 0 && w_out > 0,
            "MaxPool2d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input spatial dims [{h}x{w}]; output would be [{h_out}x{w_out}]",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::zeros_on([n, c, h_out, w_out], &backend)?;
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout()?;

        backend.max_pool2d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        )?;

        coeus_autograd::max_pool2d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

// ── MaxPool3d ──

/// 3D max pooling layer.
#[derive(Clone)]
pub struct MaxPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Cubic pooling window side length.
    pub kernel_size: usize,
    /// Stride along D, H, and W dimensions.
    pub stride: usize,
    /// Zero-padding applied to all spatial sides.
    pub padding: usize,
    /// Spacing between pooling window elements.
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> MaxPool3d<T, B> {
    /// Create with stride equal to kernel_size, no padding, no dilation.
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    /// Create with explicit hyperparameters.
    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for MaxPool3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        let backend = B::default();

        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let d = input.tensor.shape()[2];
        let h = input.tensor.shape()[3];
        let w = input.tensor.shape()[4];
        let d_out = out_dim(
            d,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        let h_out = out_dim(
            h,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        let w_out = out_dim(
            w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        );
        assert!(
            d_out > 0 && h_out > 0 && w_out > 0,
            "MaxPool3d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input spatial dims [{d}x{h}x{w}]",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::zeros_on([n, c, d_out, h_out, w_out], &backend)?;
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout()?;

        backend.max_pool3d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        )?;

        coeus_autograd::max_pool3d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
