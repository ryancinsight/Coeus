use coeus_core::{Layout, Scalar, Shape, Strides};

use super::ops::{BinaryOp, ReductionOp, UnaryOp};

/// Dynamic operations supported by execution hardware backends.
///
/// `BackendOps<T>` is the single dispatch surface that routes all tensor kernels
/// (elementwise, matmul, conv, pooling, attention, optimizer steps) to the
/// underlying device.  The CPU path is provided by a blanket `impl BackendOps<T>
/// for CpuBackend` in `backend_ops::cpu`; other devices add a new `impl` without
/// touching the algorithm bodies.
///
/// # Examples
/// ```ignore
/// // Obtain the default CPU backend and run an elementwise add.
/// let backend = coeus_ops::CpuBackend::default();
/// backend.elementwise_binary(BinaryOp::Add, &a_buf, &a_layout, &b_buf, &b_layout, &mut c_buf, &c_layout);
/// ```
pub trait BackendOps<T: Scalar>: coeus_core::ComputeBackend {
    /// Element-wise binary operations.
    fn elementwise_binary(
        &self,
        op: BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    );

    /// Element-wise unary operations.
    fn elementwise_unary(
        &self,
        op: UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    );

    /// Matrix multiplication.
    fn matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    );

    /// Matrix multiplication with accumulation: `c += a * b`.
    fn matmul_accumulate(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        let temp_len = c_layout.shape().iter().product();
        let mut temp = self.allocate::<T>(temp_len);
        let temp_layout =
            Layout::from_shape_strides(c_layout.shape_cloned(), c_layout.strides_cloned(), 0);
        self.fill(&mut temp, T::zero());
        self.matmul(a, a_layout, b, b_layout, &mut temp, &temp_layout);
        let c_ptr = c as *mut Self::DeviceBuffer<T>;
        unsafe {
            self.elementwise_binary(
                BinaryOp::Add,
                &*c_ptr,
                c_layout,
                &temp,
                &temp_layout,
                &mut *c_ptr,
                c_layout,
            );
        }
    }

    /// Rank-3 batched matrix multiplication.
    ///
    /// The default implementation keeps backend compatibility by slicing each
    /// batch into rank-2 layouts and dispatching through [`Self::matmul`]. CPU
    /// backends override this with the `coeus-leto` batched kernel.
    fn batched_matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        assert_eq!(a_layout.ndim(), 3, "batched_matmul: lhs must be rank 3");
        assert_eq!(b_layout.ndim(), 3, "batched_matmul: rhs must be rank 3");
        assert_eq!(c_layout.ndim(), 3, "batched_matmul: out must be rank 3");

        let [lhs_batch, m, lhs_k] = shape3(a_layout.shape(), "lhs");
        let [rhs_batch, rhs_k, n] = shape3(b_layout.shape(), "rhs");
        let [out_batch, out_m, out_n] = shape3(c_layout.shape(), "out");
        assert!(
            (lhs_batch == out_batch || lhs_batch == 1)
                && (rhs_batch == out_batch || rhs_batch == 1)
                && lhs_k == rhs_k
                && m == out_m
                && n == out_n,
            "batched_matmul: incompatible shapes {:?}, {:?}, {:?}",
            a_layout.shape(),
            b_layout.shape(),
            c_layout.shape(),
        );

        let lhs_batch_stride = if lhs_batch == 1 {
            0
        } else {
            a_layout.strides()[0]
        };
        let rhs_batch_stride = if rhs_batch == 1 {
            0
        } else {
            b_layout.strides()[0]
        };
        let out_batch_stride = c_layout.strides()[0];

        let lhs_shape = Shape::from([m, lhs_k].as_slice());
        let rhs_shape = Shape::from([rhs_k, n].as_slice());
        let out_shape = Shape::from([out_m, out_n].as_slice());
        let lhs_strides = Strides::from([a_layout.strides()[1], a_layout.strides()[2]].as_slice());
        let rhs_strides = Strides::from([b_layout.strides()[1], b_layout.strides()[2]].as_slice());
        let out_strides = Strides::from([c_layout.strides()[1], c_layout.strides()[2]].as_slice());

        for batch in 0..out_batch {
            let lhs_layout = Layout::from_shape_strides(
                lhs_shape.clone(),
                lhs_strides.clone(),
                a_layout.offset() + batch * lhs_batch_stride,
            );
            let rhs_layout = Layout::from_shape_strides(
                rhs_shape.clone(),
                rhs_strides.clone(),
                b_layout.offset() + batch * rhs_batch_stride,
            );
            let out_layout = Layout::from_shape_strides(
                out_shape.clone(),
                out_strides.clone(),
                c_layout.offset() + batch * out_batch_stride,
            );
            self.matmul(a, &lhs_layout, b, &rhs_layout, c, &out_layout);
        }
    }

    /// Rank-3 batched matrix multiplication with accumulation: `c += a * b`.
    fn batched_matmul_accumulate(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        let temp_len = c_layout.shape().iter().product();
        let mut temp = self.allocate::<T>(temp_len);
        let temp_layout =
            Layout::from_shape_strides(c_layout.shape_cloned(), c_layout.strides_cloned(), 0);
        self.fill(&mut temp, T::zero());
        self.batched_matmul(a, a_layout, b, b_layout, &mut temp, &temp_layout);
        let c_ptr = c as *mut Self::DeviceBuffer<T>;
        unsafe {
            self.elementwise_binary(
                BinaryOp::Add,
                &*c_ptr,
                c_layout,
                &temp,
                &temp_layout,
                &mut *c_ptr,
                c_layout,
            );
        }
    }

    /// Reduction operations along an axis.
    fn reduce(
        &self,
        op: ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    );

    /// Compute the indices of the maximum values along `axis`.
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
        let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
        self.copy_to_host(a, &mut host_a);

        let mut host_c = vec![0i64; c_layout.shape().iter().product()];
        coeus_leto::argmax_into(a_layout, &host_a, axis, c_layout, &mut host_c)
            .expect("argmax default impl failed");

        self.copy_to_device(&host_c, c);
    }

    /// Compute the indices of the minimum values along `axis`.
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
        let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
        self.copy_to_host(a, &mut host_a);

        let mut host_c = vec![0i64; c_layout.shape().iter().product()];
        coeus_leto::argmin_into(a_layout, &host_a, axis, c_layout, &mut host_c)
            .expect("argmin default impl failed");

        self.copy_to_device(&host_c, c);
    }

    /// Return the `k` largest (or smallest) values and their indices along an axis.
    #[allow(clippy::too_many_arguments)]
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
        let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
        self.copy_to_host(a, &mut host_a);

        let mut host_values = vec![T::zero(); values_layout.shape().iter().product()];
        let mut host_indices = vec![0i64; indices_layout.shape().iter().product()];

        crate::reduction::topk::topk_impl(
            &host_a,
            a_layout.shape(),
            k,
            axis,
            largest,
            &mut host_values,
            &mut host_indices,
        );

        self.copy_to_device(&host_values, values);
        self.copy_to_device(&host_indices, indices);
    }

    /// Inclusive cumulative sum along an axis.
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
        let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
        self.copy_to_host(a, &mut host_a);

        let mut host_c = vec![T::zero(); c_layout.shape().iter().product()];
        coeus_leto::cumsum_into(a_layout, &host_a, axis, c_layout, &mut host_c)
            .expect("cumsum default impl failed");

        self.copy_to_device(&host_c, c);
    }

    /// Inclusive cumulative suffix sum (reverse cumulative sum) along an axis.
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
        let mut host_a = vec![T::zero(); a_layout.shape().iter().product()];
        self.copy_to_host(a, &mut host_a);

        let mut host_c = vec![T::zero(); c_layout.shape().iter().product()];
        coeus_leto::suffix_sum_into(a_layout, &host_a, axis, c_layout, &mut host_c)
            .expect("suffix_sum default impl failed");

        self.copy_to_device(&host_c, c);
    }

    /// 1D Convolution
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
    );

    /// 1D Convolution Backward
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
    );

    /// 2D Convolution
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
    );

    /// 2D Convolution Backward
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
    );

    /// 3D Convolution
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
    );

    /// 3D Convolution Backward
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
    );

    /// 2D Max Pooling
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
    );

    /// 2D Max Pooling Backward
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
    );

    /// 2D Average Pooling
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
    );

    /// 2D Average Pooling Backward
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
    );

    /// 3D Max Pooling
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
    );

    /// 3D Max Pooling Backward
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
    );

    /// 3D Average Pooling
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
    );

    /// 3D Average Pooling Backward
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
    );

    /// Scaled dot-product attention forward.
    ///
    /// # Shapes
    /// - `query`:        `[batch, seq_q, d_k]`
    /// - `key`:          `[batch, seq_k, d_k]`
    /// - `value`:        `[batch, seq_k, d_v]`
    /// - `output`:       `[batch, seq_q, d_v]`  (pre-allocated, overwritten)
    /// - `attn_weights`: `[batch, seq_q, seq_k]` (pre-allocated, overwritten; stored for backward)
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
        T: coeus_core::Float;

    /// Scaled dot-product attention backward.
    ///
    /// Accumulates gradients into `grad_q`, `grad_k`, `grad_v` (if Some).
    #[allow(clippy::too_many_arguments)]
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
        T: coeus_core::Float;

    /// Fused SGD step update.
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
        T: coeus_core::Float;

    /// Fused Adam step update.
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
        T: coeus_core::Float;

    /// Fused RMSProp step update.
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
        T: coeus_core::Float;

    /// Fused AdamW step update (decoupled weight decay).
    ///
    /// Applies weight decay directly to the parameter before the Adam update:
    /// `p = p * (1 - lr * weight_decay)`, then applies the standard Adam correction.
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
        T: coeus_core::Float;

    /// Fused AdaGrad step update.
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
        T: coeus_core::Float;
}

fn shape3(shape: &[usize], name: &str) -> [usize; 3] {
    assert_eq!(
        shape.len(),
        3,
        "batched_matmul: {name} shape must have rank 3"
    );
    [shape[0], shape[1], shape[2]]
}
