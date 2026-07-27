use super::utils::scatter_accumulate_into;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for 1-D transposed convolution.
pub struct ConvTranspose1dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved weight tensor for backward.
    pub w_clone: Tensor<T, B>,
    /// Saved input tensor for backward.
    pub inp_clone: Tensor<T, B>,
    /// Whether a bias term was applied.
    pub has_bias: bool,
    /// Convolution stride.
    pub stride: usize,
    /// Convolution padding.
    pub padding: usize,
    /// Convolution dilation.
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for ConvTranspose1dNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "conv_transpose1d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let needs_grad_input = input_grads.get(0).and_then(|g| g.as_ref()).is_some();
        let needs_grad_weight = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        let needs_grad_bias =
            self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some();

        if !needs_grad_input && !needs_grad_weight && !needs_grad_bias {
            return;
        }

        let backend = B::default();
        let n = self.inp_clone.shape()[0];
        let c_in = self.inp_clone.shape()[1];
        let l_in = self.inp_clone.shape()[2];
        let c_out = self.w_clone.shape()[1];
        let k = self.w_clone.shape()[2];
        let l_out = grad_out.shape()[2];

        let mut input_host = vec![T::zero(); self.inp_clone.numel()];
        let mut weight_host = vec![T::zero(); self.w_clone.numel()];
        let mut grad_out_host = vec![T::zero(); grad_out.numel()];
        backend.copy_to_host(self.inp_clone.storage(), &mut input_host);
        backend.copy_to_host(self.w_clone.storage(), &mut weight_host);
        backend.copy_to_host(grad_out.storage(), &mut grad_out_host);

        if needs_grad_input {
            let mut grad_input = vec![T::zero(); self.inp_clone.numel()];
            for batch in 0..n {
                for input_channel in 0..c_in {
                    for input_index in 0..l_in {
                        let mut acc = T::zero();
                        for output_channel in 0..c_out {
                            for kernel_index in 0..k {
                                let expanded =
                                    input_index * self.stride + kernel_index * self.dilation;
                                if expanded < self.padding {
                                    continue;
                                }
                                let output_index = expanded - self.padding;
                                if output_index >= l_out {
                                    continue;
                                }
                                let grad_out_value = grad_out_host
                                    [batch * c_out * l_out + output_channel * l_out + output_index];
                                let weight_value = weight_host
                                    [input_channel * c_out * k + output_channel * k + kernel_index];
                                acc += grad_out_value * weight_value;
                            }
                        }
                        grad_input[batch * c_in * l_in + input_channel * l_in + input_index] = acc;
                    }
                }
            }
            // Fused: directly accumulate into the GradBuffer without an intermediate Tensor.
            scatter_accumulate_into(
                input_grads[0].as_ref().unwrap().write(),
                &grad_input,
                &backend,
            );
        }

        if needs_grad_weight {
            let mut grad_weight = vec![T::zero(); self.w_clone.numel()];
            for batch in 0..n {
                for input_channel in 0..c_in {
                    for input_index in 0..l_in {
                        let input_value =
                            input_host[batch * c_in * l_in + input_channel * l_in + input_index];
                        for output_channel in 0..c_out {
                            for kernel_index in 0..k {
                                let expanded =
                                    input_index * self.stride + kernel_index * self.dilation;
                                if expanded < self.padding {
                                    continue;
                                }
                                let output_index = expanded - self.padding;
                                if output_index >= l_out {
                                    continue;
                                }
                                let grad_out_value = grad_out_host
                                    [batch * c_out * l_out + output_channel * l_out + output_index];
                                let weight_offset =
                                    input_channel * c_out * k + output_channel * k + kernel_index;
                                grad_weight[weight_offset] += input_value * grad_out_value;
                            }
                        }
                    }
                }
            }
            scatter_accumulate_into(
                input_grads[1].as_ref().unwrap().write(),
                &grad_weight,
                &backend,
            );
        }

        if needs_grad_bias {
            let mut grad_bias = vec![T::zero(); c_out];
            for batch in 0..n {
                for output_channel in 0..c_out {
                    for output_index in 0..l_out {
                        grad_bias[output_channel] += grad_out_host
                            [batch * c_out * l_out + output_channel * l_out + output_index];
                    }
                }
            }
            scatter_accumulate_into(
                input_grads[2].as_ref().unwrap().write(),
                &grad_bias,
                &backend,
            );
        }
    }
}

/// Tracked 1-D transposed convolution.
#[allow(clippy::too_many_arguments)] // Mirrors the canonical convolution kernel contract.
pub fn conv_transpose1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input)
        || crate::grad_mode::should_track_var(weight)
        || bias
            .as_ref()
            .map(crate::grad_mode::should_track_var)
            .unwrap_or(false);

    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = {
            let mut values = vec![input.clone(), weight.clone()];
            if let Some(bias_var) = bias {
                values.push(bias_var.clone());
            }
            values
        };
        let node = ConvTranspose1dNode {
            output_grad,
            inputs,
            w_clone: weight.tensor.clone(),
            inp_clone: input.tensor.clone(),
            has_bias: bias.is_some(),
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── ConvTranspose2d backward node ─────────────────────────────────────────────

/// Autograd node for 2-D transposed convolution.
///
/// Backward math (stride `s`, dilation `d`, padding `p`):
///
/// ```text
/// grad_input[n, cin, hin, win] =
///     Σ_{cout, kh, kw}  grad_out[n, cout, hin*s + kh*d - p, win*s + kw*d - p]
///                        × weight[cin, cout, kh, kw]
///
/// grad_weight[cin, cout, kh, kw] +=
///     Σ_{n, hin, win}   input[n, cin, hin, win]
///                        × grad_out[n, cout, hin*s + kh*d - p, win*s + kw*d - p]
///
/// grad_bias[cout] = Σ_{n, hout, wout} grad_out[n, cout, hout, wout]
/// ```
pub struct ConvTranspose2dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved weight tensor for backward computation.
    pub w_clone: Tensor<T, B>,
    /// Saved input tensor for backward computation.
    pub inp_clone: Tensor<T, B>,
    /// Whether the convolution includes a bias term.
    pub has_bias: bool,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Padding applied to the transposed convolution.
    pub padding: usize,
    /// Dilation of the transposed convolution.
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for ConvTranspose2dNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "conv_transpose2d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let needs_grad_input = input_grads.get(0).and_then(|g| g.as_ref()).is_some();
        let needs_grad_weight = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        let needs_grad_bias =
            self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some();

        if !needs_grad_input && !needs_grad_weight && !needs_grad_bias {
            return;
        }

        let backend = B::default();
        let n = self.inp_clone.shape()[0];
        let c_in = self.inp_clone.shape()[1];
        let h_in = self.inp_clone.shape()[2];
        let w_in = self.inp_clone.shape()[3];
        let c_out = self.w_clone.shape()[1];
        let kh = self.w_clone.shape()[2];
        let kw = self.w_clone.shape()[3];
        let h_out = grad_out.shape()[2];
        let w_out = grad_out.shape()[3];
        let stride = self.stride;
        let padding = self.padding;
        let dilation = self.dilation;

        let mut inp_host = vec![T::zero(); self.inp_clone.numel()];
        let mut w_host = vec![T::zero(); self.w_clone.numel()];
        let mut go_host = vec![T::zero(); grad_out.numel()];
        backend.copy_to_host(self.inp_clone.storage(), &mut inp_host);
        backend.copy_to_host(self.w_clone.storage(), &mut w_host);
        backend.copy_to_host(grad_out.storage(), &mut go_host);

        // Helper index closures.
        let go_idx = |ni: usize, co: usize, ho: usize, wo: usize| {
            ni * c_out * h_out * w_out + co * h_out * w_out + ho * w_out + wo
        };
        let w_idx = |ci: usize, co: usize, ki: usize, kj: usize| {
            ci * c_out * kh * kw + co * kh * kw + ki * kw + kj
        };

        if needs_grad_input {
            let mut gi = vec![T::zero(); n * c_in * h_in * w_in];
            for ni in 0..n {
                for ci in 0..c_in {
                    for hi in 0..h_in {
                        for wi in 0..w_in {
                            let mut acc = T::zero();
                            for co in 0..c_out {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let raw_ho = hi * stride + ki * dilation;
                                        let raw_wo = wi * stride + kj * dilation;
                                        if raw_ho < padding || raw_wo < padding {
                                            continue;
                                        }
                                        let ho = raw_ho - padding;
                                        let wo = raw_wo - padding;
                                        if ho >= h_out || wo >= w_out {
                                            continue;
                                        }
                                        acc += go_host[go_idx(ni, co, ho, wo)]
                                            * w_host[w_idx(ci, co, ki, kj)];
                                    }
                                }
                            }
                            gi[ni * c_in * h_in * w_in + ci * h_in * w_in + hi * w_in + wi] = acc;
                        }
                    }
                }
            }
            scatter_accumulate_into(input_grads[0].as_ref().unwrap().write(), &gi, &backend);
        }

        if needs_grad_weight {
            let mut gw = vec![T::zero(); c_in * c_out * kh * kw];
            for ni in 0..n {
                for ci in 0..c_in {
                    for hi in 0..h_in {
                        for wi in 0..w_in {
                            let iv = inp_host
                                [ni * c_in * h_in * w_in + ci * h_in * w_in + hi * w_in + wi];
                            for co in 0..c_out {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let raw_ho = hi * stride + ki * dilation;
                                        let raw_wo = wi * stride + kj * dilation;
                                        if raw_ho < padding || raw_wo < padding {
                                            continue;
                                        }
                                        let ho = raw_ho - padding;
                                        let wo = raw_wo - padding;
                                        if ho >= h_out || wo >= w_out {
                                            continue;
                                        }
                                        let widx = w_idx(ci, co, ki, kj);
                                        gw[widx] += iv * go_host[go_idx(ni, co, ho, wo)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            scatter_accumulate_into(input_grads[1].as_ref().unwrap().write(), &gw, &backend);
        }

        if needs_grad_bias {
            let mut gb = vec![T::zero(); c_out];
            for ni in 0..n {
                for co in 0..c_out {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            gb[co] += go_host[go_idx(ni, co, ho, wo)];
                        }
                    }
                }
            }
            scatter_accumulate_into(input_grads[2].as_ref().unwrap().write(), &gb, &backend);
        }
    }
}

/// Tracked 2-D transposed convolution.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input)
        || crate::grad_mode::should_track_var(weight)
        || bias
            .as_ref()
            .map(crate::grad_mode::should_track_var)
            .unwrap_or(false);

    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = {
            let mut values = vec![input.clone(), weight.clone()];
            if let Some(bias_var) = bias {
                values.push(bias_var.clone());
            }
            values
        };
        let node = ConvTranspose2dNode {
            output_grad,
            inputs,
            w_clone: weight.tensor.clone(),
            inp_clone: input.tensor.clone(),
            has_bias: bias.is_some(),
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

// ── ConvTranspose3d backward node ─────────────────────────────────────────────

/// Autograd node for 3-D transposed convolution.
///
/// Backward math (stride `s`, dilation `d`, padding `p`) lifts the 2-D
/// reductions to five dimensions:
///
/// ```text
/// grad_input[n, cin, din, hin, win] =
///     Σ_{cout, kd, kh, kw}
///         grad_out[n, cout, din*s + kd*d - p, hin*s + kh*d - p, win*s + kw*d - p]
///         × weight[cin, cout, kd, kh, kw]
///
/// grad_weight[cin, cout, kd, kh, kw] +=
///     Σ_{n, din, hin, win}
///         input[n, cin, din, hin, win]
///         × grad_out[n, cout, din*s + kd*d - p, hin*s + kh*d - p, win*s + kw*d - p]
///
/// grad_bias[cout] = Σ_{n, dout, hout, wout} grad_out[n, cout, dout, hout, wout]
/// ```
pub struct ConvTranspose3dNode<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::CpuBackend + Default,
> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved weight tensor for backward computation.
    pub w_clone: Tensor<T, B>,
    /// Saved input tensor for backward computation.
    pub inp_clone: Tensor<T, B>,
    /// Whether the convolution includes a bias term.
    pub has_bias: bool,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Padding applied to the transposed convolution.
    pub padding: usize,
    /// Dilation of the transposed convolution.
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::CpuBackend + Default> BackwardNode<T, B>
    for ConvTranspose3dNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "conv_transpose3d"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let needs_grad_input = input_grads.get(0).and_then(|g| g.as_ref()).is_some();
        let needs_grad_weight = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        let needs_grad_bias =
            self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some();

        if !needs_grad_input && !needs_grad_weight && !needs_grad_bias {
            return;
        }

        let backend = B::default();
        let n = self.inp_clone.shape()[0];
        let c_in = self.inp_clone.shape()[1];
        let d_in = self.inp_clone.shape()[2];
        let h_in = self.inp_clone.shape()[3];
        let w_in = self.inp_clone.shape()[4];
        let c_out = self.w_clone.shape()[1];
        let kd = self.w_clone.shape()[2];
        let kh = self.w_clone.shape()[3];
        let kw = self.w_clone.shape()[4];
        let d_out = grad_out.shape()[2];
        let h_out = grad_out.shape()[3];
        let w_out = grad_out.shape()[4];
        let stride = self.stride;
        let padding = self.padding;
        let dilation = self.dilation;

        let mut inp_host = vec![T::zero(); self.inp_clone.numel()];
        let mut w_host = vec![T::zero(); self.w_clone.numel()];
        let mut go_host = vec![T::zero(); grad_out.numel()];
        backend.copy_to_host(self.inp_clone.storage(), &mut inp_host);
        backend.copy_to_host(self.w_clone.storage(), &mut w_host);
        backend.copy_to_host(grad_out.storage(), &mut go_host);

        let go_idx = |ni: usize, co: usize, do_: usize, ho: usize, wo: usize| {
            ni * c_out * d_out * h_out * w_out
                + co * d_out * h_out * w_out
                + do_ * h_out * w_out
                + ho * w_out
                + wo
        };
        let w_idx = |ci: usize, co: usize, ki: usize, kj: usize, kk: usize| {
            ci * c_out * kd * kh * kw + co * kd * kh * kw + ki * kh * kw + kj * kw + kk
        };

        if needs_grad_input {
            let mut gi = vec![T::zero(); n * c_in * d_in * h_in * w_in];
            for ni in 0..n {
                for ci in 0..c_in {
                    for di in 0..d_in {
                        for hi in 0..h_in {
                            for wi in 0..w_in {
                                let mut acc = T::zero();
                                for co in 0..c_out {
                                    for ki in 0..kd {
                                        for kj in 0..kh {
                                            for kk in 0..kw {
                                                let raw_do = di * stride + ki * dilation;
                                                let raw_ho = hi * stride + kj * dilation;
                                                let raw_wo = wi * stride + kk * dilation;
                                                if raw_do < padding
                                                    || raw_ho < padding
                                                    || raw_wo < padding
                                                {
                                                    continue;
                                                }
                                                let do_ = raw_do - padding;
                                                let ho = raw_ho - padding;
                                                let wo = raw_wo - padding;
                                                if do_ >= d_out || ho >= h_out || wo >= w_out {
                                                    continue;
                                                }
                                                acc += go_host[go_idx(ni, co, do_, ho, wo)]
                                                    * w_host[w_idx(ci, co, ki, kj, kk)];
                                            }
                                        }
                                    }
                                }
                                gi[ni * c_in * d_in * h_in * w_in
                                    + ci * d_in * h_in * w_in
                                    + di * h_in * w_in
                                    + hi * w_in
                                    + wi] = acc;
                            }
                        }
                    }
                }
            }
            scatter_accumulate_into(input_grads[0].as_ref().unwrap().write(), &gi, &backend);
        }

        if needs_grad_weight {
            let mut gw = vec![T::zero(); c_in * c_out * kd * kh * kw];
            for ni in 0..n {
                for ci in 0..c_in {
                    for di in 0..d_in {
                        for hi in 0..h_in {
                            for wi in 0..w_in {
                                let iv = inp_host[ni * c_in * d_in * h_in * w_in
                                    + ci * d_in * h_in * w_in
                                    + di * h_in * w_in
                                    + hi * w_in
                                    + wi];
                                for co in 0..c_out {
                                    for ki in 0..kd {
                                        for kj in 0..kh {
                                            for kk in 0..kw {
                                                let raw_do = di * stride + ki * dilation;
                                                let raw_ho = hi * stride + kj * dilation;
                                                let raw_wo = wi * stride + kk * dilation;
                                                if raw_do < padding
                                                    || raw_ho < padding
                                                    || raw_wo < padding
                                                {
                                                    continue;
                                                }
                                                let do_ = raw_do - padding;
                                                let ho = raw_ho - padding;
                                                let wo = raw_wo - padding;
                                                if do_ >= d_out || ho >= h_out || wo >= w_out {
                                                    continue;
                                                }
                                                let widx = w_idx(ci, co, ki, kj, kk);
                                                gw[widx] +=
                                                    iv * go_host[go_idx(ni, co, do_, ho, wo)];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            scatter_accumulate_into(input_grads[1].as_ref().unwrap().write(), &gw, &backend);
        }

        if needs_grad_bias {
            let mut gb = vec![T::zero(); c_out];
            for ni in 0..n {
                for co in 0..c_out {
                    for do_ in 0..d_out {
                        for ho in 0..h_out {
                            for wo in 0..w_out {
                                gb[co] += go_host[go_idx(ni, co, do_, ho, wo)];
                            }
                        }
                    }
                }
            }
            scatter_accumulate_into(input_grads[2].as_ref().unwrap().write(), &gb, &backend);
        }
    }
}

/// Tracked CPU 3-D transposed convolution.
///
/// The backward implementation uses the canonical CPU scatter loops and
/// therefore does not accept accelerator-only backends without a provider
/// implementation for the complete forward and backward operation family.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d<T: Float, B: coeus_ops::BackendOps<T> + coeus_ops::CpuBackend + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input)
        || crate::grad_mode::should_track_var(weight)
        || bias
            .as_ref()
            .map(crate::grad_mode::should_track_var)
            .unwrap_or(false);

    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = {
            let mut values = vec![input.clone(), weight.clone()];
            if let Some(bias_var) = bias {
                values.push(bias_var.clone());
            }
            values
        };
        let node = ConvTranspose3dNode {
            output_grad,
            inputs,
            w_clone: weight.tensor.clone(),
            inp_clone: input.tensor.clone(),
            has_bias: bias.is_some(),
            stride,
            padding,
            dilation,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
