// ── Binary arithmetic ops ──

use super::kernel::elementwise_binary;
use crate::backend_ops::{BackendOps, BinaryOp};
use coeus_core::{BackendError, Scalar};
use coeus_tensor::Tensor;

/// Element-wise addition.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::add;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2], &[1.0, 2.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([2], &[3.0, 4.0]);
/// let c = add(&a, &b, &backend);
/// assert_eq!(c.as_slice(), &[4.0, 6.0]);
/// ```
#[inline]
pub fn add<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Add).expect("add: incompatible shapes")
}

/// Element-wise subtraction.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::sub;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([4], &[5.0, 6.0, 7.0, 8.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
/// let c = sub(&a, &b, &backend);
/// assert_eq!(c.as_slice(), &[4.0, 4.0, 4.0, 4.0]);
/// ```
#[inline]
pub fn sub<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Sub).expect("sub: incompatible shapes")
}

/// Element-wise multiplication.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::mul;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([4], &[5.0, 6.0, 7.0, 8.0]);
/// let c = mul(&a, &b, &backend);
/// assert_eq!(c.as_slice(), &[5.0, 12.0, 21.0, 32.0]);
/// ```
#[inline]
pub fn mul<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Mul).expect("mul: incompatible shapes")
}

/// Element-wise division.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::div;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([4], &[6.0, 8.0, 10.0, 12.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([4], &[2.0, 4.0, 5.0, 6.0]);
/// let c = div(&a, &b, &backend);
/// let s = c.as_slice();
/// assert!((s[0] - 3.0).abs() < 1e-5);
/// assert!((s[1] - 2.0).abs() < 1e-5);
/// assert!((s[2] - 2.0).abs() < 1e-5);
/// assert!((s[3] - 2.0).abs() < 1e-5);
/// ```
#[inline]
pub fn div<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Div).expect("div: incompatible shapes")
}

macro_rules! binary_assign_op {
    ($name:ident, $op:expr, $doc:expr) => {
        #[doc = $doc]
        #[inline]
        pub fn $name<T: Scalar, B: BackendOps<T>>(
            a: &mut Tensor<T, B>,
            b: &Tensor<T, B>,
            backend: &B,
        ) -> Result<(), B::Error> {
            use coeus_tensor::broadcast::broadcast_shapes;
            if a.shape() != b.shape() {
                let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
                    B::Error::from(BackendError::IncompatibleBroadcast {
                        operation: "elementwise_binary_assign",
                        from: b.shape().to_vec(),
                        to: a.shape().to_vec(),
                    })
                })?;
                if &out_shape[..] != a.shape() {
                    return Err(B::Error::from(BackendError::IncompatibleBroadcast {
                        operation: "elementwise_binary_assign",
                        from: b.shape().to_vec(),
                        to: a.shape().to_vec(),
                    }));
                }
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
            )?;
            Ok(())
        }
    };
}

binary_assign_op!(add_assign, BinaryOp::Add, "In-place element-wise addition.");
binary_assign_op!(
    sub_assign,
    BinaryOp::Sub,
    "In-place element-wise subtraction."
);
binary_assign_op!(
    mul_assign,
    BinaryOp::Mul,
    "In-place element-wise multiplication."
);
binary_assign_op!(div_assign, BinaryOp::Div, "In-place element-wise division.");

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn incompatible_broadcast_panics() {
        let backend = SequentialBackend::new();
        let lhs = Tensor::from_slice([2], &[1.0_f32, 2.0]);
        let rhs = Tensor::from_slice([3], &[3.0_f32, 4.0, 5.0]);

        let result = std::panic::catch_unwind(|| add(&lhs, &rhs, &backend));
        assert!(result.is_err(), "incompatible shapes must panic");
    }
}
