use super::binary::{AddOp, DivOp, MulOp, SubOp};
use super::traits::binary_op;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Tracked element-wise multiply by a scalar.
#[must_use]
#[inline]
pub fn scalar_mul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend)?;
    let scalar_var = Var::new(scalar_tensor, false)?;
    binary_op::<T, B, MulOp>(x, &scalar_var)
}

/// Tracked element-wise add by a scalar.
#[must_use]
#[inline]
pub fn scalar_add<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend)?;
    let scalar_var = Var::new(scalar_tensor, false)?;
    binary_op::<T, B, AddOp>(x, &scalar_var)
}

/// Tracked element-wise subtraction by a scalar (x - s).
#[must_use]
#[inline]
pub fn scalar_sub<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend)?;
    let scalar_var = Var::new(scalar_tensor, false)?;
    binary_op::<T, B, SubOp>(x, &scalar_var)
}

/// Tracked element-wise division by a scalar (x / s).
#[must_use]
#[inline]
pub fn scalar_div<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    s: T,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let scalar_tensor = Tensor::full_on([1], s, &backend)?;
    let scalar_var = Var::new(scalar_tensor, false)?;
    binary_op::<T, B, DivOp>(x, &scalar_var)
}
