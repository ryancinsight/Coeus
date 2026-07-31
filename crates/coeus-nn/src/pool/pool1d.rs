// ── Pool1d ──
// MaxPool1d and AvgPool1d for 1D spatial inputs `[N, C, L]`.

use crate::module::{Module, ModuleError};
use crate::pool::checked_out_dim;
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
/// let y = pool.forward(&x).expect("valid MaxPool1d input");
/// assert_eq!(y.tensor.shape(), &[1, 1, 2]);
/// ```
#[derive(Clone)]
pub struct MaxPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Pooling window length.
    kernel_size: usize,
    /// Stride along the L dimension.
    stride: usize,
    /// Zero-padding applied to both sides.
    padding: usize,
    /// Spacing between pooling window elements.
    dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> MaxPool1d<T, B> {
    /// Create with stride equal to kernel_size, no padding, no dilation.
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    /// Create with explicit hyperparameters.
    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for MaxPool1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let backend = B::default();
        let shape = input.tensor.shape();
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module: "MaxPool1d",
                expected: "3",
                actual: shape.len(),
            });
        }
        let [n, c, l] = [shape[0], shape[1], shape[2]];
        let l_out = checked_out_dim(
            "MaxPool1d",
            l,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;

        let mut out_tensor = Tensor::alloc_on([n, c, l_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend
            .max_pool1d(
                input.tensor.storage(),
                input.tensor.layout(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                out_storage,
                out_layout,
            )
            .map_err(|source| ModuleError::Backend {
                module: "MaxPool1d",
                source,
            })?;

        Ok(coeus_autograd::max_pool1d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        ))
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
/// let y = pool.forward(&x).expect("valid AvgPool1d input");
/// assert_eq!(y.tensor.shape(), &[1, 1, 2]);
/// ```
#[derive(Clone)]
pub struct AvgPool1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Pooling window length.
    kernel_size: usize,
    /// Stride along the L dimension.
    stride: usize,
    /// Zero-padding applied to both sides.
    padding: usize,
    /// Spacing between pooling window elements.
    dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool1d<T, B> {
    /// Create with stride equal to kernel_size, no padding, no dilation.
    pub fn new(kernel_size: usize) -> Self {
        Self::with_params(kernel_size, kernel_size, 0, 1)
    }

    /// Create with explicit hyperparameters.
    pub fn with_params(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let backend = B::default();
        let shape = input.tensor.shape();
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module: "AvgPool1d",
                expected: "3",
                actual: shape.len(),
            });
        }
        let [n, c, l] = [shape[0], shape[1], shape[2]];
        let l_out = checked_out_dim(
            "AvgPool1d",
            l,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;

        let mut out_tensor = Tensor::alloc_on([n, c, l_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend
            .avg_pool1d(
                input.tensor.storage(),
                input.tensor.layout(),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                out_storage,
                out_layout,
            )
            .map_err(|source| ModuleError::Backend {
                module: "AvgPool1d",
                source,
            })?;

        Ok(coeus_autograd::avg_pool1d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        ))
    }
}
