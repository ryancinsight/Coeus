//! Pooling sub-trait.
//!
//! [`PoolOps`] is the interface-segregated sub-trait for all pooling
//! kernel dispatch (max/avg pool 1D/2D/3D forward and backward).

use coeus_core::{ComputeBackend, Layout, Scalar};

/// Pooling operations.
///
/// This sub-trait is one of seven concerns that compose
/// [`BackendOps`].  Backends implement `PoolOps` directly; the
/// blanket impl provides `BackendOps` automatically.
///
/// # Safety
///
/// On `Ok(())`, each forward method (`max_pool1d`, `avg_pool1d`,
/// `max_pool2d`, `avg_pool2d`, `max_pool3d`, and `avg_pool3d`) must write every
/// logical output element described by its output layout without reading
/// prior output contents. Backward methods require initialized gradient
/// destinations. An error may leave destinations partially written; callers
/// must discard uninitialized destinations on error.
///
/// [`BackendOps`]: super::super::BackendOps
pub unsafe trait PoolOps<T: Scalar>: ComputeBackend {
    /// 1D Max Pooling over `[N, C, L]` input.
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn max_pool1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 1D Max Pooling Backward.
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn max_pool1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 1D Average Pooling over `[N, C, L]` input.
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn avg_pool1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 1D Average Pooling Backward.
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn avg_pool1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 2D Max Pooling.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn max_pool2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 2D Max Pooling Backward.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn max_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 2D Average Pooling.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn avg_pool2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 2D Average Pooling Backward.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn avg_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 3D Max Pooling.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn max_pool3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 3D Max Pooling Backward.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn max_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 3D Average Pooling.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn avg_pool3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// 3D Average Pooling Backward.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when the backend cannot validate
    /// or dispatch the requested kernel.
    fn avg_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
