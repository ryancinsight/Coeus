use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Layout, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

use super::conv_node::accumulate_gradient;

/// Autograd node for an `DIM`-dimensional transposed convolution.
///
/// Backward execution is delegated once at the operation boundary. The
/// selected backend therefore owns both arithmetic and memory placement.
pub struct ConvTransposeNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved weight tensor for backward computation.
    pub weight: Tensor<T, B>,
    /// Saved input tensor for backward computation.
    pub input: Tensor<T, B>,
    /// Whether the convolution includes a bias term.
    pub has_bias: bool,
    /// Stride of the transposed convolution.
    pub stride: usize,
    /// Padding applied to the transposed convolution.
    pub padding: usize,
    /// Additional output extent selected by the caller.
    pub output_padding: usize,
    /// Dilation of the transposed convolution.
    pub dilation: usize,
}

/// Autograd node for one-dimensional transposed convolution.
pub type ConvTranspose1dNode<T, B> = ConvTransposeNode<T, B, 1>;
/// Autograd node for two-dimensional transposed convolution.
pub type ConvTranspose2dNode<T, B> = ConvTransposeNode<T, B, 2>;
/// Autograd node for three-dimensional transposed convolution.
pub type ConvTranspose3dNode<T, B> = ConvTransposeNode<T, B, 3>;

struct GradientOutputs<'a, T: Scalar, B: coeus_ops::BackendOps<T>> {
    input: Option<&'a mut B::DeviceBuffer<T>>,
    input_layout: &'a Layout,
    weight: Option<&'a mut B::DeviceBuffer<T>>,
    weight_layout: &'a Layout,
    bias: Option<&'a mut B::DeviceBuffer<T>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> BackwardNode<T, B>
    for ConvTransposeNode<T, B, DIM>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        match DIM {
            1 => "conv_transpose1d",
            2 => "conv_transpose2d",
            3 => "conv_transpose3d",
            _ => "conv_transpose",
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

    fn backward(
        &self,
        grad_output: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let needs_input = input_grads.first().and_then(Option::as_ref).is_some();
        let needs_weight = input_grads.get(1).and_then(Option::as_ref).is_some();
        let needs_bias = self.has_bias && input_grads.get(2).and_then(Option::as_ref).is_some();

        if !needs_input && !needs_weight && !needs_bias {
            return Ok(());
        }

        let mut grad_input =
            needs_input.then(|| Tensor::zeros_on(self.input.shape_cloned(), &backend));
        let mut grad_weight =
            needs_weight.then(|| Tensor::zeros_on(self.weight.shape_cloned(), &backend));
        let mut grad_bias =
            needs_bias.then(|| Tensor::zeros_on([self.weight.shape()[1]], &backend));

        let reference_layout = grad_output.layout();
        let (input_storage, input_layout) = optional_gradient(&mut grad_input, reference_layout);
        let (weight_storage, weight_layout) = optional_gradient(&mut grad_weight, reference_layout);
        let outputs = GradientOutputs {
            input: input_storage,
            input_layout,
            weight: weight_storage,
            weight_layout,
            bias: grad_bias.as_mut().map(Tensor::storage_mut),
        };

        dispatch_backward::<T, B, DIM>(
            &backend,
            grad_output,
            &self.input,
            &self.weight,
            outputs,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )?;

        accumulate_gradient(input_grads.first(), grad_input, &backend)?;
        accumulate_gradient(input_grads.get(1), grad_weight, &backend)?;
        accumulate_gradient(input_grads.get(2), grad_bias, &backend)
    }
}

fn optional_gradient<'a, T: Scalar, B: coeus_ops::BackendOps<T>>(
    tensor: &'a mut Option<Tensor<T, B>>,
    reference_layout: &'a Layout,
) -> (Option<&'a mut B::DeviceBuffer<T>>, &'a Layout) {
    match tensor {
        Some(tensor) => {
            let (storage, layout) = tensor.storage_mut_and_layout();
            (Some(storage), layout)
        }
        None => (None, reference_layout),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
fn dispatch_backward<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    backend: &B,
    grad_output: &Tensor<T, B>,
    input: &Tensor<T, B>,
    weight: &Tensor<T, B>,
    outputs: GradientOutputs<'_, T, B>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Result<(), B::Error> {
    let GradientOutputs {
        input: grad_input,
        input_layout: grad_input_layout,
        weight: grad_weight,
        weight_layout: grad_weight_layout,
        bias: grad_bias,
    } = outputs;

    match DIM {
        1 => backend.conv_transpose1d_backward(
            grad_output.storage(),
            grad_output.layout(),
            input.storage(),
            input.layout(),
            weight.storage(),
            weight.layout(),
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            output_padding,
            dilation,
        ),
        2 => backend.conv_transpose2d_backward(
            grad_output.storage(),
            grad_output.layout(),
            input.storage(),
            input.layout(),
            weight.storage(),
            weight.layout(),
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            output_padding,
            dilation,
        ),
        3 => backend.conv_transpose3d_backward(
            grad_output.storage(),
            grad_output.layout(),
            input.storage(),
            input.layout(),
            weight.storage(),
            weight.layout(),
            grad_input,
            grad_input_layout,
            grad_weight,
            grad_weight_layout,
            grad_bias,
            stride,
            padding,
            output_padding,
            dilation,
        ),
        _ => unreachable!("invariant: transposed convolution rank is one through three"),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
fn conv_transpose<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    output: Tensor<T, B>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(input)
        || crate::grad_mode::should_track_var(weight)
        || bias
            .as_ref()
            .is_some_and(crate::grad_mode::should_track_var);

    let grad = requires_grad.then(|| {
        Arc::new(GradBuffer::new(Tensor::zeros_on(
            output.shape_cloned(),
            &backend,
        )))
    });

    let creator = grad.as_ref().map(|output_grad| {
        let mut inputs = Vec::with_capacity(2 + usize::from(bias.is_some()));
        inputs.push(input.clone());
        inputs.push(weight.clone());
        inputs.extend(bias.iter().cloned());
        Arc::new(ConvTransposeNode::<T, B, DIM> {
            output_grad: Arc::clone(output_grad),
            inputs,
            weight: weight.tensor.clone(),
            input: input.tensor.clone(),
            has_bias: bias.is_some(),
            stride,
            padding,
            output_padding,
            dilation,
        }) as Arc<dyn BackwardNode<T, B>>
    });

    Var {
        tensor: output,
        grad,
        creator,
    }
}

/// Track a one-dimensional transposed convolution.
#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
pub fn conv_transpose1d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    output: Tensor<T, B>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    conv_transpose::<T, B, 1>(
        input,
        weight,
        bias,
        output,
        stride,
        padding,
        output_padding,
        dilation,
    )
}

/// Track a two-dimensional transposed convolution.
#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
pub fn conv_transpose2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    output: Tensor<T, B>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    conv_transpose::<T, B, 2>(
        input,
        weight,
        bias,
        output,
        stride,
        padding,
        output_padding,
        dilation,
    )
}

/// Track a three-dimensional transposed convolution.
#[expect(
    clippy::too_many_arguments,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
pub fn conv_transpose3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    output: Tensor<T, B>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Var<T, B> {
    conv_transpose::<T, B, 3>(
        input,
        weight,
        bias,
        output,
        stride,
        padding,
        output_padding,
        dilation,
    )
}
