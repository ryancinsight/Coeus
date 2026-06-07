use coeus_core::Scalar;

/// A ZST tag representing a reduction operation used in collectives.
pub trait ReduceOpTag: 'static + Copy + Clone + Send + Sync {
    /// Apply the reduction binary operation to compute the reduced scalar value.
    fn apply<T: Scalar>(a: T, b: T) -> T;
}

/// Sum reduction tag.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sum;
impl ReduceOpTag for Sum {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        a + b
    }
}

/// Max reduction tag.
#[derive(Debug, Clone, Copy, Default)]
pub struct Max;
impl ReduceOpTag for Max {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        if a > b { a } else { b }
    }
}

/// Min reduction tag.
#[derive(Debug, Clone, Copy, Default)]
pub struct Min;
impl ReduceOpTag for Min {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        if a < b { a } else { b }
    }
}

/// Product reduction tag.
#[derive(Debug, Clone, Copy, Default)]
pub struct Product;
impl ReduceOpTag for Product {
    #[inline(always)]
    fn apply<T: Scalar>(a: T, b: T) -> T {
        a * b
    }
}
