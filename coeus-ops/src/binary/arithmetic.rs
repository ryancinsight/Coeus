// ── Binary arithmetic ops ──

use coeus_core::Scalar;
use coeus_tensor::Tensor;
use crate::backend_ops::{BackendOps, BinaryOp};
use super::kernel::elementwise_binary;

macro_rules! binary_op {
    ($name:ident, $op:expr, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name<T: Scalar, B: BackendOps<T>>(
            a: &Tensor<T, B>,
            b: &Tensor<T, B>,
            backend: &B,
        ) -> Tensor<T, B> {
            elementwise_binary(a, b, backend, $op)
        }
    };
}

binary_op!(add, BinaryOp::Add, "Element-wise addition.");
binary_op!(sub, BinaryOp::Sub, "Element-wise subtraction.");
binary_op!(mul, BinaryOp::Mul, "Element-wise multiplication.");
binary_op!(div, BinaryOp::Div, "Element-wise division.");

macro_rules! binary_assign_op {
    ($name:ident, $op:expr, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name<T: Scalar, B: BackendOps<T>>(
            a: &mut Tensor<T, B>,
            b: &Tensor<T, B>,
            backend: &B,
        ) {
            use coeus_tensor::broadcast::broadcast_shapes;
            if a.shape() != b.shape() {
                let out_shape = broadcast_shapes(a.shape(), b.shape())
                    .expect("Incompatible shapes for in-place operation");
                assert_eq!(&out_shape[..], a.shape(), "In-place operation cannot expand the shape of the target tensor");
            }
            let (a_dest, a_layout) = a.storage_mut_and_layout();
            // SAFETY: We cast the mutable reference `a_dest` to an immutable reference `a_src`
            // to pass as the source buffer. This is safe because:
            // 1. `a_dest` has been made unique (Arc count is 1) via `storage_mut()`.
            // 2. The backend supports in-place / overlapping reads and writes to the same device buffer.
            // 3. We avoid cloning the device buffer (Arc clone), preventing copy-on-write reallocation.
            let a_src: &B::DeviceBuffer<T> = unsafe { &*(a_dest as *const B::DeviceBuffer<T>) };
            backend.elementwise_binary(
                $op,
                a_src,
                a_layout,
                b.storage(),
                b.layout(),
                a_dest,
                a_layout,
            );
        }
    };
}

binary_assign_op!(add_assign, BinaryOp::Add, "In-place element-wise addition.");
binary_assign_op!(sub_assign, BinaryOp::Sub, "In-place element-wise subtraction.");
binary_assign_op!(mul_assign, BinaryOp::Mul, "In-place element-wise multiplication.");
binary_assign_op!(div_assign, BinaryOp::Div, "In-place element-wise division.");
