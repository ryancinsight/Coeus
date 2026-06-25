use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

struct ConvBackwardDispatch<'a, T: Float, B: coeus_ops::BackendOps<T> + Default> {
    backend: &'a B,
    grad_out_storage: &'a B::DeviceBuffer<T>,
    grad_out_layout: &'a coeus_core::Layout,
    inp_storage: &'a B::DeviceBuffer<T>,
    inp_layout: &'a coeus_core::Layout,
    w_storage: &'a B::DeviceBuffer<T>,
    w_layout: &'a coeus_core::Layout,
    gi_storage: Option<&'a mut B::DeviceBuffer<T>>,
    gi_layout: &'a coeus_core::Layout,
    gw_storage: Option<&'a mut B::DeviceBuffer<T>>,
    gw_layout: &'a coeus_core::Layout,
    grad_bias: Option<&'a mut B::DeviceBuffer<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

/// Backend conv*_backward dispatch helper.
#[inline]
fn dispatch_conv_backward<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    request: ConvBackwardDispatch<'_, T, B>,
) {
    let ConvBackwardDispatch {
        backend,
        grad_out_storage,
        grad_out_layout,
        inp_storage,
        inp_layout,
        w_storage,
        w_layout,
        gi_storage,
        gi_layout,
        gw_storage,
        gw_layout,
        grad_bias,
        stride,
        padding,
        dilation,
    } = request;

    match DIM {
        1 => backend.conv1d_backward(
            grad_out_storage,
            grad_out_layout,
            inp_storage,
            inp_layout,
            w_storage,
            w_layout,
            gi_storage,
            gi_layout,
            gw_storage,
            gw_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        ),
        2 => backend.conv2d_backward(
            grad_out_storage,
            grad_out_layout,
            inp_storage,
            inp_layout,
            w_storage,
            w_layout,
            gi_storage,
            gi_layout,
            gw_storage,
            gw_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        ),
        3 => backend.conv3d_backward(
            grad_out_storage,
            grad_out_layout,
            inp_storage,
            inp_layout,
            w_storage,
            w_layout,
            gi_storage,
            gi_layout,
            gw_storage,
            gw_layout,
            grad_bias,
            stride,
            padding,
            dilation,
        ),
        _ => panic!("conv_backward: unsupported dimension {DIM}"),
    }
}

pub struct ConvNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub w_clone: Tensor<T, B>,
    pub inp_clone: Tensor<T, B>,
    pub has_bias: bool,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BackwardNode<T, B>
    for ConvNode<T, B, DIM>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        match DIM {
            1 => "conv1d",
            2 => "conv2d",
            3 => "conv3d",
            _ => "convNd",
        }
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
        let backend = B::default();

        let mut grad_input = if input_grads.get(0).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.inp_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_weight = if input_grads.get(1).and_then(|g| g.as_ref()).is_some() {
            Some(Tensor::zeros_on(self.w_clone.shape_cloned(), &backend))
        } else {
            None
        };

        let mut grad_bias =
            if self.has_bias && input_grads.get(2).and_then(|g| g.as_ref()).is_some() {
                Some(Tensor::zeros_on([self.w_clone.shape()[0]], &backend))
            } else {
                None
            };

        let dummy_layout = grad_out.layout();

        let mut gi_storage = None;
        let mut gi_layout_val = None;
        if let Some(ref mut gi) = grad_input {
            let (store, lay) = gi.storage_mut_and_layout();
            gi_storage = Some(store);
            gi_layout_val = Some(lay);
        }
        let gi_layout_ref = gi_layout_val.unwrap_or(dummy_layout);

        let mut gw_storage = None;
        let mut gw_layout_val = None;
        if let Some(ref mut gw) = grad_weight {
            let (store, lay) = gw.storage_mut_and_layout();
            gw_storage = Some(store);
            gw_layout_val = Some(lay);
        }
        let gw_layout_ref = gw_layout_val.unwrap_or(dummy_layout);

        dispatch_conv_backward::<T, B, DIM>(ConvBackwardDispatch {
            backend: &backend,
            grad_out_storage: grad_out.storage(),
            grad_out_layout: grad_out.layout(),
            inp_storage: self.inp_clone.storage(),
            inp_layout: self.inp_clone.layout(),
            w_storage: self.w_clone.storage(),
            w_layout: self.w_clone.layout(),
            gi_storage,
            gi_layout: gi_layout_ref,
            gw_storage,
            gw_layout: gw_layout_ref,
            grad_bias: grad_bias.as_mut().map(|gb| gb.storage_mut()),
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
        });

        if let Some(gi) = grad_input {
            let gl = input_grads[0].as_ref().unwrap().write();
            coeus_ops::add_assign(gl, &gi, &backend);
        }
        if let Some(gw) = grad_weight {
            let gl = input_grads[1].as_ref().unwrap().write();
            coeus_ops::add_assign(gl, &gw, &backend);
        }
        if let Some(gb) = grad_bias {
            let gl = input_grads[2].as_ref().unwrap().write();
            coeus_ops::add_assign(gl, &gb, &backend);
        }
    }
}

fn conv_nd_inner<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
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
            let mut v = vec![input.clone(), weight.clone()];
            if let Some(ref b) = bias {
                v.push(b.clone());
            }
            v
        };
        let w_clone = weight.tensor.clone();
        let inp_clone = input.tensor.clone();
        let has_bias = bias.is_some();

        let node = ConvNode::<T, B, DIM> {
            output_grad,
            inputs,
            w_clone,
            inp_clone,
            has_bias,
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

/// Tracked 1D Convolution.
pub fn conv1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    conv_nd_inner::<T, B, 1>(input, weight, bias, out_tensor, stride, padding, dilation)
}

/// Tracked 2D Convolution.
pub fn conv2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    conv_nd_inner::<T, B, 2>(input, weight, bias, out_tensor, stride, padding, dilation)
}

/// Tracked 3D Convolution.
pub fn conv3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Var<T, B> {
    conv_nd_inner::<T, B, 3>(input, weight, bias, out_tensor, stride, padding, dilation)
}

pub struct ConvTranspose1dNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub w_clone: Tensor<T, B>,
    pub inp_clone: Tensor<T, B>,
    pub has_bias: bool,
    pub stride: usize,
    pub padding: usize,
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
                                acc = acc + grad_out_value * weight_value;
                            }
                        }
                        grad_input[batch * c_in * l_in + input_channel * l_in + input_index] = acc;
                    }
                }
            }
            let grad_tensor = Tensor::from_slice(self.inp_clone.shape().to_vec(), &grad_input);
            let target = input_grads[0].as_ref().unwrap().write();
            coeus_ops::add_assign(target, &grad_tensor, &backend);
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
                                grad_weight[weight_offset] =
                                    grad_weight[weight_offset] + input_value * grad_out_value;
                            }
                        }
                    }
                }
            }
            let grad_tensor = Tensor::from_slice(self.w_clone.shape().to_vec(), &grad_weight);
            let target = input_grads[1].as_ref().unwrap().write();
            coeus_ops::add_assign(target, &grad_tensor, &backend);
        }

        if needs_grad_bias {
            let mut grad_bias = vec![T::zero(); c_out];
            for batch in 0..n {
                for output_channel in 0..c_out {
                    for output_index in 0..l_out {
                        grad_bias[output_channel] = grad_bias[output_channel]
                            + grad_out_host
                                [batch * c_out * l_out + output_channel * l_out + output_index];
                    }
                }
            }
            let grad_tensor = Tensor::from_slice(vec![c_out], &grad_bias);
            let target = input_grads[2].as_ref().unwrap().write();
            coeus_ops::add_assign(target, &grad_tensor, &backend);
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
