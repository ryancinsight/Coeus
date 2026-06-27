//! ZST tag types for binary operations in the fused expression DAG.

use coeus_core::Scalar;

/// Tag trait for binary operations in the fused expression DAG.
pub trait BinaryOpTag: 'static + Send + Sync + Copy + Clone {
    /// WGSL symbol for the operation (e.g. `"+"`, `"*"`).
    const WGSL_SYMBOL: &'static str;
    /// Apply the binary operation to two scalar values.
    fn apply<T: Scalar>(x: T, y: T) -> T;
}

#[derive(Clone, Copy)]
/// Addition operation tag.
pub struct Add;
impl BinaryOpTag for Add {
    const WGSL_SYMBOL: &'static str = "+";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x + y
    }
}

#[derive(Clone, Copy)]
/// Subtraction operation tag.
pub struct Sub;
impl BinaryOpTag for Sub {
    const WGSL_SYMBOL: &'static str = "-";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x - y
    }
}

#[derive(Clone, Copy)]
/// Multiplication operation tag.
pub struct Mul;
impl BinaryOpTag for Mul {
    const WGSL_SYMBOL: &'static str = "*";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x * y
    }
}

#[derive(Clone, Copy)]
/// Division operation tag.
pub struct Div;
impl BinaryOpTag for Div {
    const WGSL_SYMBOL: &'static str = "/";
    #[inline(always)]
    fn apply<T: Scalar>(x: T, y: T) -> T {
        x / y
    }
}
