use super::conv_node::conv_nd_inner;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

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
