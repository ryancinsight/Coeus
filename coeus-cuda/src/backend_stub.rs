use crate::storage::CudaStorage;
use coeus_core::{Backend, ComputeBackend, Layout, Scalar, Storage, StorageMut};

/// Trait mapping CPU scalar types to their CUDA type representation.
///
/// # Examples
///
/// ```
/// use coeus_cuda::CudaScalar;
///
/// assert_eq!(f32::CUDA_TYPE, "float");
/// assert_eq!(f64::CUDA_TYPE, "double");
/// assert_eq!(i32::CUDA_TYPE, "int");
/// ```
pub trait CudaScalar: Scalar + leto_ops::Scalar {
    /// CUDA type name string used in NVRTC kernel compilation.
    const CUDA_TYPE: &'static str;
}

impl CudaScalar for f32 {
    const CUDA_TYPE: &'static str = "float";
}

impl CudaScalar for f64 {
    const CUDA_TYPE: &'static str = "double";
}

impl CudaScalar for half::f16 {
    const CUDA_TYPE: &'static str = "__half";
}

impl CudaScalar for half::bf16 {
    const CUDA_TYPE: &'static str = "__nv_bfloat16";
}

impl CudaScalar for i32 {
    const CUDA_TYPE: &'static str = "int";
}

/// CUDA backend compiled without CUDA provider support.
///
/// Without the `cuda` feature, `CudaBackend` delegates to the CPU
/// sequential backend so the workspace builds on machines without a
/// CUDA toolkit.
///
/// # Examples
///
/// ```
/// use coeus_cuda::CudaBackend;
/// use coeus_core::ComputeBackend;
///
/// let backend = CudaBackend::new();
/// assert_eq!(backend.name(), "cuda-cpu-fallback");
/// assert_eq!(backend.num_threads(), 1);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBackend;

impl coeus_core::backend::private::Sealed for CudaBackend {}

impl CudaBackend {
    /// Create a new CUDA backend instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_cuda::CudaBackend;
    ///
    /// let backend = CudaBackend::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CudaBackend {
    type DeviceBuffer<T: Scalar> = CudaStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "cuda-cpu-fallback"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        CudaStorage::new(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        if let Some(slice) = dst.try_as_mut_slice() {
            slice.fill(val);
        }
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        let dst_slice = dst
            .try_as_mut_slice()
            .expect("invariant: no-CUDA CudaStorage is CPU-addressable");
        dst_slice.copy_from_slice(src);
    }

    #[inline]
    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        let src_slice = src
            .try_as_slice()
            .expect("invariant: no-CUDA CudaStorage is CPU-addressable");
        dst.copy_from_slice(src_slice);
    }
}

impl Backend for CudaBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        coeus_core::SequentialBackend::new().parallel_for(start, end, f);
    }
}

impl<T: CudaScalar> coeus_ops::BackendOps<T> for CudaBackend {
    #[inline]
    fn elementwise_binary(
        &self,
        op: coeus_ops::BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        self.fallback_binary(op, a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: coeus_ops::UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        self.fallback_unary(op, a, a_layout, c, c_layout);
    }

    #[inline]
    fn matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        self.fallback_matmul(a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn reduce(
        &self,
        op: coeus_ops::ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        self.fallback_reduce(op, a, a_layout, axis, c, c_layout);
    }

    #[inline]
    fn conv1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        self.fallback_conv1d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        self.fallback_conv1d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[inline]
    fn conv2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        self.fallback_conv2d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        self.fallback_conv2d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[inline]
    fn conv3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        self.fallback_conv3d(
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
    fn conv3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        self.fallback_conv3d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            weight,
            weight_layout,
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        );
    }

    #[inline]
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
    ) {
        self.fallback_max_pool2d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_max_pool2d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_avg_pool2d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_avg_pool2d_backward(
            grad_out,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_max_pool3d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_max_pool3d_backward(
            grad_out,
            grad_out_layout,
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_avg_pool3d(
            input,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            output,
            output_layout,
        );
    }

    #[inline]
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
    ) {
        self.fallback_avg_pool3d_backward(
            grad_out,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            grad_input,
            grad_input_layout,
        );
    }

    #[inline]
    fn sdp_attention(
        &self,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&Self::DeviceBuffer<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
        attn_weights: &mut Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_sdp_attention(
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            key_padding_mask,
            key_padding_mask_layout,
            is_causal,
            scale,
            output,
            output_layout,
            attn_weights,
            attn_weights_layout,
        );
    }

    #[inline]
    fn sdp_attention_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        attn_weights: &Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
        scale: T,
        grad_q: Option<&mut Self::DeviceBuffer<T>>,
        grad_k: Option<&mut Self::DeviceBuffer<T>>,
        grad_v: Option<&mut Self::DeviceBuffer<T>>,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_sdp_attention_backward(
            grad_out,
            grad_out_layout,
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            attn_weights,
            attn_weights_layout,
            scale,
            grad_q,
            grad_k,
            grad_v,
        );
    }

    #[inline]
    fn sgd_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<T>,
        velocity_layout: &Layout,
        lr: T,
        momentum: T,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_sgd_step(
            param,
            param_layout,
            grad,
            grad_layout,
            velocity,
            velocity_layout,
            lr,
            momentum,
        );
    }

    #[inline]
    fn adam_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_adam_step(
            param,
            param_layout,
            grad,
            grad_layout,
            m,
            m_layout,
            v,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            t,
        );
    }

    #[inline]
    fn rmsprop_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        alpha: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_rmsprop_step(
            param,
            param_layout,
            grad,
            grad_layout,
            v,
            v_layout,
            lr,
            alpha,
            eps,
        );
    }

    #[inline]
    fn adamw_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_adamw_step(
            param,
            param_layout,
            grad,
            grad_layout,
            m,
            m_layout,
            v,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        );
    }

    #[inline]
    fn adagrad_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        history: &mut Self::DeviceBuffer<T>,
        history_layout: &Layout,
        lr: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        self.fallback_adagrad_step(
            param,
            param_layout,
            grad,
            grad_layout,
            history,
            history_layout,
            lr,
            eps,
        );
    }
}
