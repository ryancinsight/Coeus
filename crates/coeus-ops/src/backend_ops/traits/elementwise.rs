//! Element-wise binary and unary operation sub-trait.
//!
//! [`ElementwiseOps`] is the interface-segregated sub-trait for all
//! element-wise kernel dispatch.  It is a super-trait of [`crate::backend_ops::BackendOps`];
//! backends implement this trait directly and receive `BackendOps` via the
//! blanket impl.

use coeus_core::{ComputeBackend, Layout, Scalar};

use super::super::ops::{BinaryOp, UnaryOp};

/// Element-wise binary and unary operations.
///
/// This sub-trait is one of seven concerns that compose
/// [`BackendOps`].  Backends implement `ElementwiseOps` directly; the
/// blanket impl provides `BackendOps` automatically.
///
/// [`BackendOps`]: super::super::BackendOps
pub trait ElementwiseOps<T: Scalar>: ComputeBackend {
    /// Element-wise binary operations.
    fn elementwise_binary(
        &self,
        op: BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Element-wise unary operations.
    fn elementwise_unary(
        &self,
        op: UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
