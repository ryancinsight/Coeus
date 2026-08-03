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

    /// Apply a binary operation to `a` and `b`, replacing `a` with the result.
    ///
    /// The default uses a distinct output allocation because accelerator APIs
    /// may prohibit binding one buffer for simultaneous read and write. CPU
    /// backends override this with provider-owned in-place traversal.
    fn elementwise_binary_assign(
        &self,
        op: BinaryOp,
        a: &mut Self::DeviceBuffer<T>,
        a_layout: &mut Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let output_layout = Layout::new(a_layout.shape_cloned());
        let mut output = self.allocate(output_layout.numel());
        self.elementwise_binary(op, a, a_layout, b, b_layout, &mut output, &output_layout)?;
        *a = output;
        *a_layout = output_layout;
        Ok(())
    }

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
