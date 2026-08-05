//! Element-wise binary and unary operation sub-trait.
//!
//! [`ElementwiseOps`] is the interface-segregated sub-trait for all
//! element-wise kernel dispatch.  It is a super-trait of [`crate::backend_ops::BackendOps`];
//! backends implement this trait directly and receive `BackendOps` via the
//! blanket impl.

use coeus_core::{ComputeBackend, Float, Layout, Scalar, StorageMut};

use super::super::ops::{BinaryOp, UnaryOp};

/// Element-wise binary and unary operations.
///
/// This sub-trait is one of the concerns that compose
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

    /// Update one layout inside `destination` while preserving all elements
    /// outside that layout.
    ///
    /// The default snapshots the original storage handle, detaches a candidate
    /// through device-local COW, then dispatches between distinct allocations.
    /// The candidate replaces the destination only after successful dispatch.
    /// CPU backends override this with direct Leto mutation.
    fn elementwise_binary_update(
        &self,
        op: BinaryOp,
        destination: &mut Self::DeviceBuffer<T>,
        destination_layout: &Layout,
        rhs: &Self::DeviceBuffer<T>,
        rhs_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let source = destination.clone();
        let mut candidate = source.clone();
        candidate.make_unique();
        self.elementwise_binary(
            op,
            &source,
            destination_layout,
            rhs,
            rhs_layout,
            &mut candidate,
            destination_layout,
        )?;
        *destination = candidate;
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

    /// Apply a unary operation to `input`, replacing it with the result.
    ///
    /// A distinct output avoids simultaneous shared and mutable references and
    /// accelerator read/write binding conflicts. The compact output allocation
    /// is fully initialized by [`Self::elementwise_unary`] before installation.
    fn elementwise_unary_assign(
        &self,
        op: UnaryOp,
        input: &mut Self::DeviceBuffer<T>,
        input_layout: &mut Layout,
    ) -> Result<(), Self::Error> {
        let output_layout = Layout::new(input_layout.shape_cloned());
        let mut output = self.allocate(output_layout.numel());
        self.elementwise_unary(op, input, input_layout, &mut output, &output_layout)?;
        *input = output;
        *input_layout = output_layout;
        Ok(())
    }
}

/// Provider-owned scalar exponentiation.
pub trait ScalarPowerOps<T: Float>: ComputeBackend {
    /// Compute `output = input.powf(exponent)` over the input layout.
    fn elementwise_pow_scalar(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        exponent: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
