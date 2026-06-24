use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

// ── Shared pooling helpers ──

#[inline]
fn k_eff(kernel_size: usize, dilation: usize) -> usize {
    dilation * (kernel_size - 1) + 1
}

#[inline]
fn out_dim(
    input_dim: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    let total = input_dim + 2 * padding;
    match total.checked_sub(k_eff(kernel_size, dilation)) {
        Some(numer) => numer / stride + 1,
        None => 0,
    }
}

// ── AvgPool2d ──

#[derive(Clone)]
pub struct AvgPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool2d<T, B> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
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
            "AvgPool2d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input spatial dims [{h}x{w}]; output would be [{h_out}x{w_out}]",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::zeros_on([n, c, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.avg_pool2d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::avg_pool2d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

// ── MaxPool2d ──

#[derive(Clone)]
pub struct MaxPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> MaxPool2d<T, B> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

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

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
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

        let mut out_tensor = Tensor::zeros_on([n, c, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.max_pool2d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

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

// ── AvgPool3d ──

#[derive(Clone)]
pub struct AvgPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool3d<T, B> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
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
            "AvgPool3d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input spatial dims [{d}x{h}x{w}]",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::zeros_on([n, c, d_out, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.avg_pool3d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::avg_pool3d(
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

#[derive(Clone)]
pub struct MaxPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> MaxPool3d<T, B> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

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

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
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

        let mut out_tensor = Tensor::zeros_on([n, c, d_out, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.max_pool3d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

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

// ── Global average pooling ─────────────────────────────────────────────────
//
// GlobalAvgPool{1,2,3}d reduces all spatial dimensions to 1, equivalent to
// `AvgPool{N}d(kernel_size=spatial_size, stride=1)`.  They are zero-parameter
// ZST modules (PhantomData only) so there is no allocation overhead.

/// Global average pooling for 3-D inputs `[N, C, L]` → `[N, C, 1]`.
///
/// Equivalent to `torch.nn.AdaptiveAvgPool1d(1)`.
#[derive(Clone, Default)]
pub struct GlobalAvgPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend>(
    PhantomData<(T, B)>,
);

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> GlobalAvgPool1d<T, B> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GlobalAvgPool1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        assert_eq!(input.tensor.ndim(), 3, "GlobalAvgPool1d expects [N,C,L]");
        coeus_autograd::mean_axis(input, 2)
    }
}

/// Global average pooling for 4-D inputs `[N, C, H, W]` → `[N, C, 1, 1]`.
///
/// Equivalent to `torch.nn.AdaptiveAvgPool2d(1)`.
#[derive(Clone, Default)]
pub struct GlobalAvgPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend>(
    PhantomData<(T, B)>,
);

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> GlobalAvgPool2d<T, B> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GlobalAvgPool2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        assert_eq!(input.tensor.ndim(), 4, "GlobalAvgPool2d expects [N,C,H,W]");
        let after_h = coeus_autograd::mean_axis(input, 2);
        coeus_autograd::mean_axis(&after_h, 3)
    }
}

/// Global average pooling for 5-D inputs `[N, C, D, H, W]` → `[N, C, 1, 1, 1]`.
///
/// Equivalent to `torch.nn.AdaptiveAvgPool3d(1)`.
#[derive(Clone, Default)]
pub struct GlobalAvgPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend>(
    PhantomData<(T, B)>,
);

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> GlobalAvgPool3d<T, B> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GlobalAvgPool3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        assert_eq!(
            input.tensor.ndim(),
            5,
            "GlobalAvgPool3d expects [N,C,D,H,W]"
        );
        let after_d = coeus_autograd::mean_axis(input, 2);
        let after_h = coeus_autograd::mean_axis(&after_d, 3);
        coeus_autograd::mean_axis(&after_h, 4)
    }
}

/// Global max pooling for 4-D inputs `[N, C, H, W]` → `[N, C, 1, 1]`.
///
/// Equivalent to `torch.nn.AdaptiveMaxPool2d(1)`.
#[derive(Clone, Default)]
pub struct GlobalMaxPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend>(
    PhantomData<(T, B)>,
);

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> GlobalMaxPool2d<T, B> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GlobalMaxPool2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        assert_eq!(input.tensor.ndim(), 4, "GlobalMaxPool2d expects [N,C,H,W]");
        let after_h = coeus_autograd::max_axis(input, 2);
        coeus_autograd::max_axis(&after_h, 3)
    }
}

/// Global max pooling for 5-D inputs `[N, C, D, H, W]` → `[N, C, 1, 1, 1]`.
#[derive(Clone, Default)]
pub struct GlobalMaxPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend>(
    PhantomData<(T, B)>,
);

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> GlobalMaxPool3d<T, B> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GlobalMaxPool3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        assert_eq!(
            input.tensor.ndim(),
            5,
            "GlobalMaxPool3d expects [N,C,D,H,W]"
        );
        let after_d = coeus_autograd::max_axis(input, 2);
        let after_h = coeus_autograd::max_axis(&after_d, 3);
        coeus_autograd::max_axis(&after_h, 4)
    }
}
