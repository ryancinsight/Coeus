use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use std::marker::PhantomData;

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
    /// Create a new `GlobalAvgPool1d` (zero-sized, no parameters).
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
    /// Create a new `GlobalAvgPool2d` (zero-sized, no parameters).
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
    /// Create a new `GlobalAvgPool3d` (zero-sized, no parameters).
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
    /// Create a new `GlobalMaxPool2d` (zero-sized, no parameters).
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
    /// Create a new `GlobalMaxPool3d` (zero-sized, no parameters).
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
