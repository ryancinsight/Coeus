use crate::Tensor;
use backend::Backend;
use storage::DenseStorage;
use dtype::DataType;
use anyhow::Result;

/// Applies a bilinear transformation to the incoming data:
/// y = x1 @ A @ x2.T + b
///
/// Shape:
/// - input1: (N, *, H1)
/// - input2: (N, *, H2)
/// - weight: (Out, H1, H2)
/// - bias: (Out)
/// - output: (N, *, Out)
pub fn bilinear<B, T>(
    input1: &Tensor<B, DenseStorage<T>, T>,
    input2: &Tensor<B, DenseStorage<T>, T>,
    weight: &Tensor<B, DenseStorage<T>, T>,
    bias: Option<&Tensor<B, DenseStorage<T>, T>>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    T: DataType + num_traits::Float + Clone + 'static,
{
    // input1: [N, H1] (simplified)
    // input2: [N, H2]
    // weight: [Out, H1, H2]
    
    let shape1 = input1.shape().dims();
    let shape2 = input2.shape().dims();
    let weight_shape = weight.shape().dims();
    
    let n = shape1[0] as isize;
    let h1 = shape1[1] as isize;
    let h2 = shape2[1] as isize;
    let out_features = weight_shape[0] as isize;
    
    // Steps:
    // 1. W_t = weight.permute(1, 0, 2) -> (H1, Out, H2)
    // 2. W_flat = W_t.reshape(h1, out_features * h2)
    // 3. temp = input1.matmul(W_flat) -> (N, Out * H2)
    // 4. temp = temp.reshape(n, out_features, h2)
    // 5. res = temp * input2.unsqueeze(1) -> (N, Out, H2) * (N, 1, H2) -> (N, Out, H2)
    // 6. res = res.sum(-1) -> (N, Out)
    // 7. Add bias if present.

    let w_t = crate::ops::shape::permute(weight, &[1, 0, 2])?; // (H1, Out, H2)
    let w_flat = w_t.reshape(&[h1, out_features * h2])?;
    
    let temp = super::matmul(input1, &w_flat)?; // (N, Out * H2)
    let temp = temp.reshape(&[n, out_features, h2])?;
    
    let input2_unsqueezed = crate::ops::shape::unsqueeze(input2, 1)?;
    let mid_res = crate::ops::arithmetic::mul(&temp, &input2_unsqueezed)?; // (N, Out, H2)
    
    let mut res = crate::ops::reduction::sum(&mid_res, Some(&[2]), false)?; // (N, Out), keepdim=false
    
    if let Some(b) = bias {
        res = crate::ops::arithmetic::add(&res, b)?;
    }
    
    Ok(res)
}
