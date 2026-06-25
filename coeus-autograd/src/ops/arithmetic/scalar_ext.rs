use super::scalar::{scalar_add as free_scalar_add, scalar_div as free_scalar_div, scalar_mul as free_scalar_mul, scalar_sub as free_scalar_sub};
use crate::var::Var;
use coeus_core::Scalar;

/// Extension trait for generic-backend scalar arithmetic on [`Var<T, B>`].
///
/// The `std::ops::{Mul,Add,Sub,Div}<T>` traits cannot be implemented generically
/// over `B` due to the orphan rule (both `T` and the implementing type `Var<T,B>`
/// are parameterized by the same `T`).  This extension trait provides the same
/// operations via method syntax for any backend:
///
/// ```rust
/// use coeus_autograd::{Var, VarScalarExt};
/// # let v: Var<f64, _> = unimplemented!();
/// let scaled = v.scalar_mul(2.0);
/// ```
///
/// For the default `MoiraiBackend`, the `*`/`+`/`-`/`/` operators also work
/// directly via the concrete impls in `var_ops`.
pub trait VarScalarExt<T: Scalar>: Sized {
    type Output;

    /// `self * scalar` (element-wise multiply by scalar).
    fn scalar_mul(self, s: T) -> Self::Output;

    /// `self + scalar` (element-wise add by scalar).
    fn scalar_add(self, s: T) -> Self::Output;

    /// `self - scalar` (element-wise subtract by scalar).
    fn scalar_sub(self, s: T) -> Self::Output;

    /// `self / scalar` (element-wise divide by scalar).
    fn scalar_div(self, s: T) -> Self::Output;
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> VarScalarExt<T> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn scalar_mul(self, s: T) -> Var<T, B> {
        free_scalar_mul(&self, s)
    }

    #[inline]
    fn scalar_add(self, s: T) -> Var<T, B> {
        free_scalar_add(&self, s)
    }

    #[inline]
    fn scalar_sub(self, s: T) -> Var<T, B> {
        free_scalar_sub(&self, s)
    }

    #[inline]
    fn scalar_div(self, s: T) -> Var<T, B> {
        free_scalar_div(&self, s)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> VarScalarExt<T> for &Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn scalar_mul(self, s: T) -> Var<T, B> {
        free_scalar_mul(self, s)
    }

    #[inline]
    fn scalar_add(self, s: T) -> Var<T, B> {
        free_scalar_add(self, s)
    }

    #[inline]
    fn scalar_sub(self, s: T) -> Var<T, B> {
        free_scalar_sub(self, s)
    }

    #[inline]
    fn scalar_div(self, s: T) -> Var<T, B> {
        free_scalar_div(self, s)
    }
}
