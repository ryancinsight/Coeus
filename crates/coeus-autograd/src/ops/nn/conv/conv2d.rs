use super::utils::conv_nd_inner;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Tracked 2D Convolution.
pub fn conv2d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
    bias: &Option<Var<T, B>>,
    out_tensor: Tensor<T, B>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Var<T, B>, B::Error> {
    conv_nd_inner::<T, B, 2>(input, weight, bias, out_tensor, stride, padding, dilation)
}
