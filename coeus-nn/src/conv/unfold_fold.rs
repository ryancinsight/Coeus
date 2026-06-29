// ── Unfold and Fold NN modules ──
//
// Unfold1d / Fold1d / Unfold2d / Fold2d:
//   Thin stateless modules that delegate to `coeus_ops::unfold1d` /
//   `coeus_ops::fold1d` / `coeus_ops::unfold2d` / `coeus_ops::fold2d`.
//
// They match PyTorch `nn.Unfold` / `nn.Fold` semantics:
//   Unfold 1D: [N, C, L]      → [N, C*kernel, L_out]
//   Fold   1D: [N, C*kernel, L_out] → [N, C, output_size]
//   Unfold 2D: [N, C, H, W]   → [N, C*kH*kW, H_out*W_out]
//   Fold   2D: [N, C*kH*kW, L]   → [N, C, H, W]
//
// All hyperparameters are carried as struct fields rather than const generics
// so that runtime-configurable instances (e.g., loaded from configs) work
// naturally. ZST phantom markers are added for type-safety with `T` and `B`.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use std::marker::PhantomData;

// ── Unfold1d ──────────────────────────────────────────────────────────────────

/// Extracts sliding windows from `[N, C, L]` into `[N, C*kernel_size, L_out]`.
///
/// Stateless module with no learnable parameters. Matches PyTorch `nn.Unfold` in 1D.
///
/// # Shape
/// - Input:  `[N, C, L]`
/// - Output: `[N, C * kernel_size, L_out]`
///   where `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// # Examples
///
/// ```
/// use coeus_nn::{Unfold1d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let m = Unfold1d::<f32, SequentialBackend>::new(3, 1, 0, 1);
/// let x = Var::new(Tensor::<f32, SequentialBackend>::ones([1, 2, 5]), false);
/// let y = m.forward(&x);
/// assert_eq!(y.tensor.shape(), &[1, 6, 3]); // C*k=2*3=6, L_out=3
/// ```
#[derive(Clone, Debug)]
pub struct Unfold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Sliding window size.
    pub kernel_size: usize,
    /// Stride of the window.
    pub stride: usize,
    /// Zero-padding on each side of the input.
    pub padding: usize,
    /// Dilation (spacing between kernel elements).
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Unfold1d<T, B> {
    /// Create an Unfold1d with given hyperparameters.
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
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

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Unfold1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();
        let out_tensor = coeus_ops::unfold1d(
            &input.tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            &backend,
        );
        Var::new(out_tensor, false)
    }
}

// ── Fold1d ────────────────────────────────────────────────────────────────────

/// Accumulates `[N, C*kernel_size, L_out]` back into `[N, C, output_size]`.
///
/// Inverse (adjoint) of [`Unfold1d`]. Overlapping window contributions are summed.
/// Matches PyTorch `nn.Fold` in 1D.
///
/// # Shape
/// - Input:  `[N, C * kernel_size, L_out]`
/// - Output: `[N, C, output_size]`
#[derive(Clone, Debug)]
pub struct Fold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Target output length.
    pub output_size: usize,
    /// Kernel size that was used for unfolding.
    pub kernel_size: usize,
    /// Stride of the window.
    pub stride: usize,
    /// Zero-padding on each side.
    pub padding: usize,
    /// Dilation of the window.
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Fold1d<T, B> {
    /// Create a Fold1d with given hyperparameters.
    pub fn new(
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Fold1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();
        let out_tensor = coeus_ops::fold1d(
            &input.tensor,
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            &backend,
        );
        Var::new(out_tensor, false)
    }
}

// ── Unfold2d ──────────────────────────────────────────────────────────────────

/// Extracts sliding 2D windows from `[N, C, H, W]` into `[N, C*kH*kW, H_out*W_out]`.
///
/// Matches PyTorch `nn.Unfold`. Stateless; no learnable parameters.
///
/// # Shape
/// - Input:  `[N, C, H, W]`
/// - Output: `[N, C * kH * kW, H_out * W_out]`
#[derive(Clone, Debug)]
pub struct Unfold2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Vertical padding.
    pub padding_h: usize,
    /// Horizontal padding.
    pub padding_w: usize,
    /// Vertical dilation.
    pub dilation_h: usize,
    /// Horizontal dilation.
    pub dilation_w: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Unfold2d<T, B> {
    /// Create Unfold2d with a square kernel (equal h/w params).
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: stride,
            stride_w: stride,
            padding_h: padding,
            padding_w: padding,
            dilation_h: dilation,
            dilation_w: dilation,
            _marker: PhantomData,
        }
    }

    /// Create Unfold2d with per-axis hyperparameters.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> Self {
        assert!(
            stride_h >= 1 && stride_w >= 1 && dilation_h >= 1 && dilation_w >= 1,
            "strides and dilations must be >= 1"
        );
        Self {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Unfold2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();
        let out_tensor = coeus_ops::unfold2d(
            &input.tensor,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            &backend,
        );
        Var::new(out_tensor, false)
    }
}

// ── Fold2d ────────────────────────────────────────────────────────────────────

/// Accumulates `[N, C*kH*kW, H_out*W_out]` back into `[N, C, output_h, output_w]`.
///
/// Inverse (adjoint) of [`Unfold2d`]. Overlapping contributions are summed.
/// Matches PyTorch `nn.Fold`.
#[derive(Clone, Debug)]
pub struct Fold2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Target output height.
    pub output_h: usize,
    /// Target output width.
    pub output_w: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Vertical padding.
    pub padding_h: usize,
    /// Horizontal padding.
    pub padding_w: usize,
    /// Vertical dilation.
    pub dilation_h: usize,
    /// Horizontal dilation.
    pub dilation_w: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Fold2d<T, B> {
    /// Create Fold2d with given target output size and square kernel params.
    pub fn new(
        output_h: usize,
        output_w: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
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
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Fold2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();
        let out_tensor = coeus_ops::fold2d(
            &input.tensor,
            self.output_h,
            self.output_w,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            &backend,
        );
        Var::new(out_tensor, false)
    }
}
