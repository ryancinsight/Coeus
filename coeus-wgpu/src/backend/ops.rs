use coeus_core::{Layout, Storage};
use crate::backend::{WgpuBackend, WgpuScalar};
use crate::kernels;

impl<T: WgpuScalar> coeus_ops::BackendOps<T> for WgpuBackend {
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
        if a.len() == c.len()
            && b.len() == c.len()
            && a_layout.is_contiguous()
            && a_layout.offset() == 0
            && b_layout.is_contiguous()
            && b_layout.offset() == 0
            && c_layout.is_contiguous()
            && c_layout.offset() == 0
        {
            kernels::dispatch_contiguous_binary::<T>(op, &a.buffer, &b.buffer, &c.buffer, c.len());
        } else {
            kernels::dispatch_binary::<T>(op, &a.buffer, a_layout, &b.buffer, b_layout, &c.buffer, c_layout, c.len());
        }
    }

    fn elementwise_unary(
        &self,
        op: coeus_ops::UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        if a.len() == c.len()
            && a_layout.is_contiguous()
            && a_layout.offset() == 0
            && c_layout.is_contiguous()
            && c_layout.offset() == 0
        {
            kernels::dispatch_contiguous_unary::<T>(op, &a.buffer, &c.buffer, c.len());
        } else {
            kernels::dispatch_unary::<T>(op, &a.buffer, a_layout, &c.buffer, c_layout, c.len());
        }
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
        kernels::dispatch_matmul::<T>(&a.buffer, a_layout, &b.buffer, b_layout, &c.buffer, c_layout);
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
        kernels::dispatch_reduce::<T>(op, &a.buffer, a_layout, axis, &c.buffer, c_layout);
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
        let out_numel = output_layout.shape().iter().product::<usize>();
        kernels::dispatch_conv1d::<T>(
            &input.buffer,
            &weight.buffer,
            bias.map(|b| b.buffer.as_ref()),
            &output.buffer,
            input_layout,
            weight_layout,
            output_layout,
            stride,
            padding,
            dilation,
            out_numel,
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
        kernels::dispatch_conv1d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            &weight.buffer,
            weight_layout,
            grad_input.map(|gi| gi.buffer.as_ref()),
            grad_input_layout,
            grad_weight.map(|gw| gw.buffer.as_ref()),
            grad_weight_layout,
            grad_bias.map(|gb| gb.buffer.as_ref()),
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
        let out_numel = output_layout.shape().iter().product::<usize>();
        kernels::dispatch_conv2d::<T>(
            &input.buffer,
            &weight.buffer,
            bias.map(|b| b.buffer.as_ref()),
            &output.buffer,
            input_layout,
            weight_layout,
            output_layout,
            stride,
            padding,
            dilation,
            out_numel,
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
        kernels::dispatch_conv2d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            &weight.buffer,
            weight_layout,
            grad_input.map(|gi| gi.buffer.as_ref()),
            grad_input_layout,
            grad_weight.map(|gw| gw.buffer.as_ref()),
            grad_weight_layout,
            grad_bias.map(|gb| gb.buffer.as_ref()),
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
        kernels::dispatch_max_pool2d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
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
        kernels::dispatch_max_pool2d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
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
        kernels::dispatch_avg_pool2d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
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
        kernels::dispatch_avg_pool2d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
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
    ) where T: coeus_core::Float {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_sgd_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &velocity.buffer,
            velocity_layout,
            lr,
            momentum,
            len,
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
    ) where T: coeus_core::Float {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_adam_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &m.buffer,
            m_layout,
            &v.buffer,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            t,
            len,
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
    ) where T: coeus_core::Float {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_adamw_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &m.buffer,
            m_layout,
            &v.buffer,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
            len,
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
    ) where T: coeus_core::Float {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_rmsprop_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &v.buffer,
            v_layout,
            lr,
            alpha,
            eps,
            len,
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
    ) where T: coeus_core::Float {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_adagrad_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &history.buffer,
            history_layout,
            lr,
            eps,
            len,
        );
    }

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
        kernels::dispatch_conv3d::<T>(
            &input.buffer,
            &weight.buffer,
            bias.map(|b| b.buffer.as_ref()),
            &output.buffer,
            input_layout,
            weight_layout,
            output_layout,
            stride,
            padding,
            dilation,
            output.len(),
        );
    }

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
        kernels::dispatch_conv3d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            &weight.buffer,
            weight_layout,
            grad_input.map(|gi| gi.buffer.as_ref()),
            grad_input_layout,
            grad_weight.map(|gw| gw.buffer.as_ref()),
            grad_weight_layout,
            grad_bias.map(|gb| gb.buffer.as_ref()),
            stride,
            padding,
            dilation,
        );
    }

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
        kernels::dispatch_max_pool3d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
        );
    }

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
        kernels::dispatch_max_pool3d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
        );
    }

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
        kernels::dispatch_avg_pool3d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
        );
    }

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
        kernels::dispatch_avg_pool3d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
        );
    }

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
    ) where T: coeus_core::Float {
        use coeus_core::{SequentialBackend, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut};
        // Stage: GPU → CPU
        let seq = SequentialBackend::new();
        let q_len = query.len();
        let k_len = key.len();
        let v_len = value.len();
        let out_len = output.len();
        let aw_len = attn_weights.len();
        let mut q_cpu = seq.allocate::<T>(q_len);
        let mut k_cpu = seq.allocate::<T>(k_len);
        let mut v_cpu = seq.allocate::<T>(v_len);
        let mut out_cpu = seq.allocate::<T>(out_len);
        let mut aw_cpu = seq.allocate::<T>(aw_len);
        self.copy_to_host(query, q_cpu.as_mut_slice());
        self.copy_to_host(key, k_cpu.as_mut_slice());
        self.copy_to_host(value, v_cpu.as_mut_slice());

        let mask_cpu = key_padding_mask.map(|mask| {
            let mut buf = seq.allocate::<T>(mask.len());
            self.copy_to_host(mask, buf.as_mut_slice());
            buf
        });

        // Execute on CPU
        seq.sdp_attention(
            &q_cpu,
            query_layout,
            &k_cpu,
            key_layout,
            &v_cpu,
            value_layout,
            mask_cpu.as_ref(),
            key_padding_mask_layout,
            is_causal,
            scale,
            &mut out_cpu,
            output_layout,
            &mut aw_cpu,
            attn_weights_layout,
        );
        // Stage back: CPU → GPU
        self.copy_to_device(out_cpu.as_slice(), output);
        self.copy_to_device(aw_cpu.as_slice(), attn_weights);
    }

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
    ) where T: coeus_core::Float {
        use coeus_core::{SequentialBackend, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut};
        let seq = SequentialBackend::new();
        // Stage inputs to CPU
        let go_len = grad_out.len();
        let q_len = query.len();
        let k_len = key.len();
        let v_len = value.len();
        let aw_len = attn_weights.len();
        let mut go_cpu = seq.allocate::<T>(go_len);
        let mut q_cpu  = seq.allocate::<T>(q_len);
        let mut k_cpu  = seq.allocate::<T>(k_len);
        let mut v_cpu  = seq.allocate::<T>(v_len);
        let mut aw_cpu = seq.allocate::<T>(aw_len);
        self.copy_to_host(grad_out, go_cpu.as_mut_slice());
        self.copy_to_host(query, q_cpu.as_mut_slice());
        self.copy_to_host(key, k_cpu.as_mut_slice());
        self.copy_to_host(value, v_cpu.as_mut_slice());
        self.copy_to_host(attn_weights, aw_cpu.as_mut_slice());
        // Prepare CPU gradient buffers
        let mut gq_cpu = grad_q.as_ref().map(|g| { let mut s = seq.allocate::<T>(g.len()); self.copy_to_host(*g, s.as_mut_slice()); s });
        let mut gk_cpu = grad_k.as_ref().map(|g| { let mut s = seq.allocate::<T>(g.len()); self.copy_to_host(*g, s.as_mut_slice()); s });
        let mut gv_cpu = grad_v.as_ref().map(|g| { let mut s = seq.allocate::<T>(g.len()); self.copy_to_host(*g, s.as_mut_slice()); s });
        seq.sdp_attention_backward(&go_cpu, grad_out_layout, &q_cpu, query_layout, &k_cpu, key_layout, &v_cpu, value_layout, &aw_cpu, attn_weights_layout, scale, gq_cpu.as_mut(), gk_cpu.as_mut(), gv_cpu.as_mut());
        // Stage back
        if let (Some(gq_gpu), Some(ref gq_c)) = (grad_q, &gq_cpu) {
            self.copy_to_device(gq_c.as_slice(), gq_gpu);
        }
        if let (Some(gk_gpu), Some(ref gk_c)) = (grad_k, &gk_cpu) {
            self.copy_to_device(gk_c.as_slice(), gk_gpu);
        }
        if let (Some(gv_gpu), Some(ref gv_c)) = (grad_v, &gv_cpu) {
            self.copy_to_device(gv_c.as_slice(), gv_gpu);
        }
    }
}


