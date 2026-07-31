use super::traits::{binary_op, BinaryAutogradOp};
use crate::backward::reduce_broadcast;
use crate::grad_buffer::GradBuffer;
use crate::var::Var;
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// ZST tag for element-wise addition autograd.
pub struct AddOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for AddOp {
    const OP_NAME: &'static str = "add";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::add(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _a: &Tensor<T, B>,
        _b: &Tensor<T, B>,
        a_shape: &Shape,
        b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let gl = g.write();
            if grad_out.shape() == &a_shape[..] {
                coeus_ops::add_assign(gl, grad_out, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), a_shape);
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let gl = g.write();
            if grad_out.shape() == &b_shape[..] {
                coeus_ops::add_assign(gl, grad_out, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), b_shape);
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        Ok(())
    }
}

/// ZST tag for element-wise subtraction autograd.
pub struct SubOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for SubOp {
    const OP_NAME: &'static str = "sub";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sub(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _a: &Tensor<T, B>,
        _b: &Tensor<T, B>,
        a_shape: &Shape,
        b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let gl = g.write();
            if grad_out.shape() == &a_shape[..] {
                coeus_ops::add_assign(gl, grad_out, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), a_shape);
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let gl = g.write();
            if grad_out.shape() == &b_shape[..] {
                coeus_ops::sub_assign(gl, grad_out, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), b_shape);
                coeus_ops::sub_assign(gl, &reduced, backend)?;
            }
        }
        Ok(())
    }
}

/// ZST tag for element-wise multiplication autograd.
pub struct MulOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for MulOp {
    const OP_NAME: &'static str = "mul";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mul(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        _a_shape: &Shape,
        _b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let prod = coeus_ops::mul(grad_out, b, backend);
            let gl = g.write();
            if prod.shape() == a.shape() {
                coeus_ops::add_assign(gl, &prod, backend)?;
            } else {
                let reduced = reduce_broadcast(prod, a.shape());
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let prod = coeus_ops::mul(grad_out, a, backend);
            let gl = g.write();
            if prod.shape() == b.shape() {
                coeus_ops::add_assign(gl, &prod, backend)?;
            } else {
                let reduced = reduce_broadcast(prod, b.shape());
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        Ok(())
    }
}

/// ZST tag for element-wise division autograd.
pub struct DivOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for DivOp {
    const OP_NAME: &'static str = "div";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::div(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        _a_shape: &Shape,
        _b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        if let Some(Some(ref g)) = input_grads.get(0) {
            let grad_a = coeus_ops::div(grad_out, b, backend);
            let gl = g.write();
            if grad_a.shape() == a.shape() {
                coeus_ops::add_assign(gl, &grad_a, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_a, a.shape());
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let b_sq = coeus_ops::mul(b, b, backend);
            let grad_b_pos = coeus_ops::div(&coeus_ops::mul(grad_out, a, backend), &b_sq, backend);
            let gl = g.write();
            if grad_b_pos.shape() == b.shape() {
                coeus_ops::sub_assign(gl, &grad_b_pos, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_b_pos, b.shape());
                coeus_ops::sub_assign(gl, &reduced, backend)?;
            }
        }
        Ok(())
    }
}

/// ZST tag for element-wise remainder (Python/torch modulo) autograd.
///
/// `remainder(a, b) = a - floor(a / b) * b` (result carries the sign of the
/// divisor `b`, matching `torch.remainder` and NumPy `remainder`, in contrast
/// to the C-style `fmod` which carries the sign of the dividend).
///
/// The quotient `q = floor(a / b)` is piecewise-constant in both operands, so
/// its own derivative is zero almost everywhere and the gradient reduces to
/// that of `a - q * b` with `q` held constant:
///   ∂/∂a = 1        (identity)
///   ∂/∂b = −q       (matches PyTorch `-grad * self.div(other, floor)`)
pub struct RemainderOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for RemainderOp {
    const OP_NAME: &'static str = "remainder";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let q = coeus_ops::floor(&coeus_ops::div(a, b, backend), backend);
        coeus_ops::sub(a, &coeus_ops::mul(&q, b, backend), backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        _a_shape: &Shape,
        _b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        // ∂/∂a = 1: identity passthrough (broadcast-reduced to a's shape).
        if let Some(Some(ref g)) = input_grads.get(0) {
            let gl = g.write();
            if grad_out.shape() == a.shape() {
                coeus_ops::add_assign(gl, grad_out, backend)?;
            } else {
                let reduced = reduce_broadcast(grad_out.clone(), a.shape());
                coeus_ops::add_assign(gl, &reduced, backend)?;
            }
        }
        // ∂/∂b = −floor(a / b): subtract grad_out · q (broadcast-reduced to b).
        if let Some(Some(ref g)) = input_grads.get(1) {
            let q = coeus_ops::floor(&coeus_ops::div(a, b, backend), backend);
            let prod = coeus_ops::mul(grad_out, &q, backend);
            let gl = g.write();
            if prod.shape() == b.shape() {
                coeus_ops::sub_assign(gl, &prod, backend)?;
            } else {
                let reduced = reduce_broadcast(prod, b.shape());
                coeus_ops::sub_assign(gl, &reduced, backend)?;
            }
        }
        Ok(())
    }
}

/// Tracked element-wise addition.
///
/// # Examples
///
/// `y = a + b`; the gradient of the scalar sum flows through unchanged, so
/// `da = db = [1, 1, 1]`.
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let a = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
/// let b = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[4.0, 5.0, 6.0]), true);
/// let y = coeus_autograd::add(&a, &b);
/// assert!((y.tensor.as_slice()[0] - 5.0).abs() < 1e-5);
/// let loss = coeus_autograd::sum(&y);
/// loss.backward().expect("invariant: valid autograd fixture completes backward");
/// let ga = a.grad().unwrap();
/// assert!((ga.as_slice()[0] - 1.0).abs() < 1e-5);
/// assert!((ga.as_slice()[1] - 1.0).abs() < 1e-5);
/// assert!((ga.as_slice()[2] - 1.0).abs() < 1e-5);
/// ```
#[must_use]
#[inline]
pub fn add<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    binary_op::<T, B, AddOp>(a, b)
}

/// Tracked element-wise subtraction.
#[must_use]
#[inline]
pub fn sub<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    binary_op::<T, B, SubOp>(a, b)
}

/// Tracked element-wise multiplication.
///
/// # Examples
///
/// `y = a * b`; for the scalar sum, `da = b` and `db = a`.
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let a = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[1.0, 2.0, 3.0]), true);
/// let b = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([3], &[4.0, 5.0, 6.0]), true);
/// let y = coeus_autograd::mul(&a, &b);
/// assert!((y.tensor.as_slice()[0] - 4.0).abs() < 1e-5);
/// let loss = coeus_autograd::sum(&y);
/// loss.backward().expect("invariant: valid autograd fixture completes backward");
/// let ga = a.grad().unwrap();
/// assert!((ga.as_slice()[0] - 4.0).abs() < 1e-5); // da = b
/// assert!((ga.as_slice()[1] - 5.0).abs() < 1e-5);
/// let gb = b.grad().unwrap();
/// assert!((gb.as_slice()[0] - 1.0).abs() < 1e-5); // db = a
/// assert!((gb.as_slice()[2] - 3.0).abs() < 1e-5);
/// ```
#[must_use]
#[inline]
pub fn mul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    binary_op::<T, B, MulOp>(a, b)
}

/// Tracked element-wise division.
#[must_use]
#[inline]
pub fn div<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    binary_op::<T, B, DivOp>(a, b)
}

/// Non-differentiable element-wise equality comparison mask.
#[must_use]
#[inline]
pub fn eq<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    Var::new(coeus_ops::eq(&a.tensor, &b.tensor, &backend), false)
}

/// Non-differentiable element-wise inequality comparison mask.
#[must_use]
#[inline]
pub fn ne<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    Var::new(coeus_ops::ne(&a.tensor, &b.tensor, &backend), false)
}

/// Non-differentiable element-wise less-than comparison mask.
#[must_use]
#[inline]
pub fn lt<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    Var::new(coeus_ops::lt(&a.tensor, &b.tensor, &backend), false)
}

/// Non-differentiable element-wise greater-than comparison mask.
#[must_use]
#[inline]
pub fn gt<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    Var::new(coeus_ops::gt(&a.tensor, &b.tensor, &backend), false)
}

/// Non-differentiable element-wise less-than-or-equal comparison mask.
#[must_use]
#[inline]
pub fn le<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    Var::new(coeus_ops::le(&a.tensor, &b.tensor, &backend), false)
}

/// Non-differentiable element-wise greater-than-or-equal comparison mask.
#[must_use]
#[inline]
pub fn ge<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    Var::new(coeus_ops::ge(&a.tensor, &b.tensor, &backend), false)
}

/// Tracked element-wise remainder (`torch.remainder` / NumPy `remainder`):
/// `a - floor(a / b) * b`, carrying the sign of the divisor `b`.
///
/// Gradient flows to `a` as the identity and to `b` as `-floor(a / b)`
/// (the floor quotient is held constant, matching PyTorch).
///
/// # Examples
///
/// `7 % -3 = -2` (sign of the divisor); `d/da = 1`, `d/db = -floor(7/-3) = 3`.
///
/// ```
/// use coeus_autograd::{remainder, sum, Var};
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let a = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([1], &[7.0]), true);
/// let b = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([1], &[-3.0]), true);
/// let r = remainder(&a, &b);
/// assert!((r.tensor.as_slice()[0] - (-2.0)).abs() < 1e-12);
/// sum(&r).backward().expect("invariant: valid autograd fixture completes backward");
/// assert!((a.grad().unwrap().as_slice()[0] - 1.0).abs() < 1e-12);
/// assert!((b.grad().unwrap().as_slice()[0] - 3.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn remainder<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    binary_op::<T, B, RemainderOp>(a, b)
}
