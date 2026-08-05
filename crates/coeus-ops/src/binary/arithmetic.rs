// ── Binary arithmetic ops ──

use super::kernel::elementwise_binary;
use crate::backend_ops::{BinaryOp, ElementwiseOps};
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
pub fn add<T: Scalar, B: ElementwiseOps<T>>(
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
pub fn sub<T: Scalar, B: ElementwiseOps<T>>(
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
pub fn mul<T: Scalar, B: ElementwiseOps<T>>(
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
pub fn div<T: Scalar, B: ElementwiseOps<T>>(
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
        pub fn $name<T: Scalar, B: ElementwiseOps<T>>(
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
            let (a_dest, a_layout) = a.storage_and_layout_mut();
            backend.elementwise_binary_assign($op, a_dest, a_layout, b.storage(), b.layout())?;
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

    #[test]
    fn cpu_assign_broadcasts_without_replacing_storage() {
        let backend = SequentialBackend::new();
        let mut lhs = Tensor::from_slice([2, 2], &[1.0_f32, 2.0, 3.0, 4.0]);
        let rhs = Tensor::from_slice([1, 2], &[10.0_f32, 20.0]);
        let allocation = lhs.as_slice().as_ptr();

        add_assign(&mut lhs, &rhs, &backend).expect("valid row broadcast");

        assert_eq!(lhs.as_slice(), &[11.0, 22.0, 13.0, 24.0]);
        assert_eq!(lhs.as_slice().as_ptr(), allocation);
    }
}
