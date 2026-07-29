use crate::module::{Module, ModuleError};
use crate::pool::checked_out_dim;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

// ── AvgPool2d ──

/// 2D average pooling layer.
#[derive(Clone)]
pub struct AvgPool2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Pooling window side length.
    kernel_size: usize,
    /// Stride along H and W dimensions.
    stride: usize,
    /// Zero-padding applied to all spatial sides.
    padding: usize,
    /// Spacing between pooling window elements.
    dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool2d<T, B> {
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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let backend = B::default();
        let shape = input.tensor.shape();
        if shape.len() != 4 {
            return Err(ModuleError::InvalidRank {
                module: "AvgPool2d",
                expected: "4",
                actual: shape.len(),
            });
        }
        let [n, c, h, w] = [shape[0], shape[1], shape[2], shape[3]];
        let h_out = checked_out_dim(
            "AvgPool2d",
            h,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;
        let w_out = checked_out_dim(
            "AvgPool2d",
            w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;

        let mut out_tensor = Tensor::zeros_on([n, c, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend
            .avg_pool2d(
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
                module: "AvgPool2d",
                source,
            })?;

        Ok(coeus_autograd::avg_pool2d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        ))
    }
}

// ── AvgPool3d ──

/// 3D average pooling layer.
#[derive(Clone)]
pub struct AvgPool3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Cubic pooling window side length.
    kernel_size: usize,
    /// Stride along D, H, and W dimensions.
    stride: usize,
    /// Zero-padding applied to all spatial sides.
    padding: usize,
    /// Spacing between pooling window elements.
    dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> AvgPool3d<T, B> {
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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AvgPool3d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let backend = B::default();
        let shape = input.tensor.shape();
        if shape.len() != 5 {
            return Err(ModuleError::InvalidRank {
                module: "AvgPool3d",
                expected: "5",
                actual: shape.len(),
            });
        }
        let [n, c, d, h, w] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
        let d_out = checked_out_dim(
            "AvgPool3d",
            d,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;
        let h_out = checked_out_dim(
            "AvgPool3d",
            h,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;
        let w_out = checked_out_dim(
            "AvgPool3d",
            w,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )?;

        let mut out_tensor = Tensor::zeros_on([n, c, d_out, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();

        backend
            .avg_pool3d(
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
                module: "AvgPool3d",
                source,
            })?;

        Ok(coeus_autograd::avg_pool3d(
            input,
            out_tensor,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        ))
    }
}
