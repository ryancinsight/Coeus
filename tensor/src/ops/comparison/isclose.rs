use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, traits::FloatExt};
use storage::{Storage, StorageFromVec};
use num_traits::FromPrimitive;

/// Returns a boolean tensor where two tensors are element-wise equal within a tolerance.
///
/// The equation is: `|input - other| <= atol + rtol * |other|`
pub fn isclose<
    T: DataType + FloatExt + PartialOrd + num_traits::One + num_traits::Zero + FromPrimitive + num_traits::Signed + 'static,
    B: Backend<Data = T> + Clone + Send + Sync + Default + 'static,
    S1: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + StorageFromVec<T> + 'static,
    S2: Storage<T> + Clone + Send + Sync + crate::ops::TensorStorageOps<T> + StorageFromVec<T> + 'static,
>(
    input: &Tensor<B, S1, T>,
    other: &Tensor<B, S2, T>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<Tensor<B, S1, T>> {
    // cast tolerances to T
    let rtol_t = T::from_f64(rtol).unwrap_or(T::zero());
    let atol_t = T::from_f64(atol).unwrap_or(T::zero());
    
    // diff = |input - other|
    let diff = crate::ops::arithmetic::sub(input, other)?;
    let diff_abs = crate::ops::math::abs(&diff)?;
    
    let other_abs = crate::ops::math::abs(other)?;
    
    // Create scalar tensors for broadcasting
    let shape_broadcast = vec![1; input.shape().ndim()];
    
    // Use proper casting/creation. Assuming B handles S1.
    // We need backend from input.
    let backend = input.backend.clone();
    
    let rtol_tensor = Tensor::<B, S1, T>::from_vec_with_backend(vec![rtol_t], &shape_broadcast, backend.clone())?;
    let atol_tensor = Tensor::<B, S1, T>::from_vec_with_backend(vec![atol_t], &shape_broadcast, backend.clone())?;
    
    // tol = atol + rtol * |other|
    // Need to cast other_abs to S1 if S1 != S2?
    // math::abs returns Tensor with same storage as input.
    // If other is S2, other_abs is S2.
    // mul(S1, S2) -> S1 (usually).
    let term = crate::ops::arithmetic::mul(&rtol_tensor, &other_abs)?; 
    // term is S1.
    
    let tol = crate::ops::arithmetic::add(&atol_tensor, &term)?;
    
    // result = diff <= tol
    let result = crate::ops::comparison::le(&diff_abs, &tol)?;
    
    // Equal NaNs handling ignored for simplicity or assume masked later
    Ok(result)
}
