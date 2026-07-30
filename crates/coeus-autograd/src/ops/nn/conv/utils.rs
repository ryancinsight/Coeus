use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub(super) struct ConvBackwardDispatch<'a, T: Float, B: coeus_ops::BackendOps<T> + Default> {
    pub backend: &'a B,
    pub grad_out_storage: &'a B::DeviceBuffer<T>,
    pub grad_out_layout: &'a coeus_core::Layout,
    pub inp_storage: &'a B::DeviceBuffer<T>,
    pub inp_layout: &'a coeus_core::Layout,
    pub w_storage: &'a B::DeviceBuffer<T>,
    pub w_layout: &'a coeus_core::Layout,
    pub gi_storage: Option<&'a mut B::DeviceBuffer<T>>,
    pub gi_layout: &'a coeus_core::Layout,
    pub gw_storage: Option<&'a mut B::DeviceBuffer<T>>,
    pub gw_layout: &'a coeus_core::Layout,
    pub grad_bias: Option<&'a mut B::DeviceBuffer<T>>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

/// Backend conv*_backward dispatch helper.
#[inline]
pub(super) fn dispatch_conv_backward<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    const DIM: usize,
>(
    request: ConvBackwardDispatch<'_, T, B>,
) -> Result<(), B::Error> {
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
        _ => unreachable!("invariant: convolution spatial rank is one through three"),
    }
}

/// Autograd node for N-dimensional convolution.
pub struct ConvNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
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
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
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
        })?;

        if let Some(gi) = grad_input {
            let gl = input_grads[0].as_ref().unwrap().write();
            coeus_ops::add_assign(gl, &gi, &backend)?;
        }
        if let Some(gw) = grad_weight {
            let gl = input_grads[1].as_ref().unwrap().write();
            coeus_ops::add_assign(gl, &gw, &backend)?;
        }
        if let Some(gb) = grad_bias {
            let gl = input_grads[2].as_ref().unwrap().write();
            coeus_ops::add_assign(gl, &gb, &backend)?;
        }

        Ok(())
    }
}

/// Inner generic implementation for N-dimensional convolution.
pub(super) fn conv_nd_inner<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
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
