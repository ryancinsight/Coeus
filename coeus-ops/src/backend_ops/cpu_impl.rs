use super::traits::{
    AttentionOps, ConvOps, ElementwiseOps, MatmulOps, OptimizerOps, PoolOps, ReductionOps,
};
use super::{BinaryOp, ReductionOp, UnaryOp};
use coeus_core::{Backend, CpuAddressableStorageMut, Layout, Scalar};

mod attention;
mod conv;
mod elementwise;
mod matmul;
mod optim;
mod pool;
mod reduction;

mod sealed {
    pub trait Sealed {}
    impl Sealed for coeus_core::SequentialBackend {}
    impl Sealed for coeus_core::MoiraiBackend {}
}

/// CPU execution backend marker trait.
///
/// Implemented by [`MoiraiBackend`] and [`SequentialBackend`] from `coeus_core`.
/// The trait is sealed: external crates cannot add new implementations.
///
/// [`MoiraiBackend`]: coeus_core::MoiraiBackend
/// [`SequentialBackend`]: coeus_core::SequentialBackend
///
/// # Examples
///
/// ```
/// use coeus_ops::CpuBackend;
/// use coeus_core::SequentialBackend;
///
/// fn accept_cpu<B: CpuBackend>(_: &B) {}
///
/// let backend = SequentialBackend::new();
/// accept_cpu(&backend);
/// ```
pub trait CpuBackend: Backend + sealed::Sealed {
    /// Borrow an `i64` device buffer as a mutable slice.
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64];
}

impl CpuBackend for coeus_core::SequentialBackend {
    #[inline]
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64] {
        use coeus_core::CpuAddressableStorageMut;
        buf.as_mut_slice()
    }
}

impl CpuBackend for coeus_core::MoiraiBackend {
    #[inline]
    fn as_mut_slice_i64<'a>(&self, buf: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64] {
        use coeus_core::CpuAddressableStorageMut;
        buf.as_mut_slice()
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> ElementwiseOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn elementwise_binary(
        &self,
        op: BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        elementwise::elementwise_binary(self, op, a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        elementwise::elementwise_unary(self, op, a, a_layout, c, c_layout);
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> MatmulOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
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
        matmul::matmul(self, a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn batched_matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        matmul::batched_matmul(self, a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn matmul_accumulate(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        matmul::matmul_accumulate(self, a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn batched_matmul_accumulate(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        matmul::batched_matmul_accumulate(self, a, a_layout, b, b_layout, c, c_layout);
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> ReductionOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn reduce(
        &self,
        op: ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        reduction::reduce(self, op, a, a_layout, axis, c, c_layout);
    }

    #[inline]
    fn argmax(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<i64>,
        c_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
    {
        reduction::argmax(self, a, a_layout, axis, c, c_layout);
    }

    #[inline]
    fn argmin(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<i64>,
        c_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
    {
        reduction::argmin(self, a, a_layout, axis, c, c_layout);
    }

    #[inline]
    fn topk(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        k: usize,
        axis: usize,
        largest: bool,
        values: &mut Self::DeviceBuffer<T>,
        values_layout: &Layout,
        indices: &mut Self::DeviceBuffer<i64>,
        indices_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
    {
        reduction::topk(
            self,
            a,
            a_layout,
            k,
            axis,
            largest,
            values,
            values_layout,
            indices,
            indices_layout,
        );
    }

    #[inline]
    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
    {
        reduction::cumsum(self, a, a_layout, axis, c, c_layout);
    }

    #[inline]
    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
    {
        reduction::suffix_sum(self, a, a_layout, axis, c, c_layout);
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> ConvOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
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
        conv::conv1d(
            self,
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
        conv::conv1d_backward(
            self,
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
        conv::conv2d(
            self,
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
        conv::conv2d_backward(
            self,
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
        conv::conv3d(
            self,
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
        conv::conv3d_backward(
            self,
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
}

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> PoolOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
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
        pool::max_pool2d(
            self,
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
        pool::max_pool2d_backward(
            self,
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
        pool::avg_pool2d(
            self,
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
        pool::avg_pool2d_backward(
            self,
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
        pool::max_pool3d(
            self,
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
        pool::max_pool3d_backward(
            self,
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
        pool::avg_pool3d(
            self,
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
        pool::avg_pool3d_backward(
            self,
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

    // ── Pool 1D ──────────────────────────────────────────────────────────────

    #[inline]
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
    ) {
        pool::max_pool1d(
            self,
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
    ) {
        pool::max_pool1d_backward(
            self,
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
    ) {
        pool::avg_pool1d(
            self,
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
    ) {
        pool::avg_pool1d_backward(
            self,
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
}

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> OptimizerOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
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
    ) {
        optim::sgd_step(
            self,
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
        optim::adam_step(
            self,
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
    ) {
        optim::rmsprop_step(
            self,
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
        optim::adamw_step(
            self,
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
        optim::adagrad_step(
            self,
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

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> AttentionOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
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
        attention::sdp_attention(
            self,
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
        attention::sdp_attention_backward(
            self,
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
}
