// ── Pool1d ──
// MaxPool1d and AvgPool1d for 1D spatial inputs `[N, C, L]`.

use crate::module::Module;
use crate::pool::out_dim;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

// ── MaxPool1d ──

/// 1D max pooling layer for inputs shaped `[N, C, L]`.
///
/// Slides a window of `kernel_size` along the length dimension and takes the maximum.
///
/// # Examples
///
/// ```
/// use coeus_nn::{MaxPool1d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let pool = MaxPool1d::<f32, SequentialBackend>::new(2);
/// let x = Var::new(Tensor::from_slice([1, 1, 4], &[1.0_f32, 3.0, 2.0, 4.0]), false);
/// let y = pool.forward(&x);
/// assert_eq!(y.tensor.shape(), &[1, 1, 2]);
/// ```
#[derive(Clone)]
pub struct MaxPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Pooling window length.
    pub kernel_size: usize,
    /// Stride along the L dimension.
    pub stride: usize,
    /// Zero-padding applied to both sides.
    pub padding: usize,
    /// Spacing between pooling window elements.
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> MaxPool1d<T, B> {
    /// Create with stride equal to kernel_size, no padding, no dilation.
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    /// Create with explicit hyperparameters.
    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(stride >= 1 && dilation >= 1, "stride and dilation must be >= 1");
        Self { kernel_size, stride, padding, dilation, _marker: PhantomData }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for MaxPool1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();

        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let l = input.tensor.shape()[2];
        let l_out = out_dim(l, self.kernel_size, self.padding, self.stride, self.dilation);

        assert!(
            l_out > 0,
            "MaxPool1d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input length {l}; output would be {l_out}",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::alloc_on([n, c, l_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.max_pool1d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::max_pool1d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

// ── AvgPool1d ──

/// 1D average pooling layer for inputs shaped `[N, C, L]`.
///
/// Slides a window of `kernel_size` along the length dimension and takes the average.
///
/// # Examples
///
/// ```
/// use coeus_nn::{AvgPool1d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let pool = AvgPool1d::<f32, SequentialBackend>::new(2);
/// let x = Var::new(Tensor::from_slice([1, 1, 4], &[1.0_f32, 3.0, 2.0, 4.0]), false);
/// let y = pool.forward(&x);
/// assert_eq!(y.tensor.shape(), &[1, 1, 2]);
/// ```
#[derive(Clone)]
pub struct AvgPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Pooling window length.
    pub kernel_size: usize,
    /// Stride along the L dimension.
    pub stride: usize,
    /// Zero-padding applied to both sides.
    pub padding: usize,
    /// Spacing between pooling window elements.
    pub dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool1d<T, B> {
    /// Create with stride equal to kernel_size, no padding, no dilation.
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    /// Create with explicit hyperparameters.
    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        assert!(stride >= 1 && dilation >= 1, "stride and dilation must be >= 1");
        Self { kernel_size, stride, padding, dilation, _marker: PhantomData }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();

        let n = input.tensor.shape()[0];
        let c = input.tensor.shape()[1];
        let l = input.tensor.shape()[2];
        let l_out = out_dim(l, self.kernel_size, self.padding, self.stride, self.dilation);

        assert!(
            l_out > 0,
            "AvgPool1d: kernel ({}) with dilation ({}) and padding ({}) \
             does not fit input length {l}; output would be {l_out}",
            self.kernel_size,
            self.dilation,
            self.padding,
        );

        let mut out_tensor = Tensor::alloc_on([n, c, l_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend.avg_pool1d(
            input.tensor.storage(),
            input.tensor.layout(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            out_storage,
            out_layout,
        );

        coeus_autograd::avg_pool1d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
