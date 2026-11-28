// -- Tracked binary transcendental compositions --
//
// `hypot`, `atan2`, element-wise `pow_tensor` and numerically stable
// `logsumexp` as tracked `Var` wrappers. Each composes existing tracked or
// untracked tensor ops so gradients flow through the DAG with no bespoke
// node and no new opcode.
//
// Backward identities:
// - `hypot(a,b) = sqrt(a^2+b^2)`: `da = g*a/y`, `db = g*b/y`.
// - `atan2(y,x)`: `dy = g*x/(x^2+y^2)`, `dx = -g*y/(x^2+y^2)`.
// - `pow_tensor(b,e) = exp(e*ln b)`: `db = g*e*b^(e-1)`,
//   `de = g*ln(b)*b^e`.
// - `logsumexp(x)`: `dx_i = g*softmax(x)_i` (max-shifted, stable).

use crate::var::Var;
use coeus_core::{Float, FloatElement};
use coeus_tensor::Tensor;

/// Tracked element-wise hypotenuse.
#[inline]
pub fn hypot<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    let y_squared = crate::add(&crate::mul(a, a)?, &crate::mul(b, b)?)?;
    crate::sqrt(&y_squared)
}

/// Tracked element-wise `atan2(y, x)`.
///
/// Backward: `dy = g*x/(x^2+y^2)`, `dx = -g*y/(x^2+y^2)`.
pub fn atan2<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default>(
    y: &Var<T, B>,
    x: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    use super::traits::binary_op;
    binary_op::<T, B, Atan2Forward>(y, x)
}

/// Forward tag for tracked `atan2`.
pub struct Atan2Forward;

impl<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default>
    super::traits::BinaryAutogradOp<T, B> for Atan2Forward
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    const OP_NAME: &'static str = "atan2";

    #[inline(always)]
    fn forward(y: &Tensor<T, B>, x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::atan2(y, x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        y: &Tensor<T, B>,
        x: &Tensor<T, B>,
        y_shape: &coeus_core::Shape,
        x_shape: &coeus_core::Shape,
        input_grads: &[Option<std::sync::Arc<crate::grad_buffer::GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        let denom = coeus_ops::add(
            &coeus_ops::mul(x, x, backend)?,
            &coeus_ops::mul(y, y, backend)?,
            backend,
        )?;
        if let Some(Some(ref g)) = input_grads.first() {
            let dy = coeus_ops::div(&coeus_ops::mul(grad_out, x, backend)?, &denom, backend)?;
            let gl = g.write();
            let reduced = crate::backward::reduce_broadcast(dy, y_shape);
            coeus_ops::add_assign(gl, &reduced, backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let neg_y = coeus_ops::neg(y, backend)?;
            let dx = coeus_ops::div(&coeus_ops::mul(grad_out, &neg_y, backend)?, &denom, backend)?;
            let gl = g.write();
            let reduced = crate::backward::reduce_broadcast(dx, x_shape);
            coeus_ops::add_assign(gl, &reduced, backend)?;
        }
        Ok(())
    }
}

/// Tracked element-wise power with a tracked exponent: `base ^ exponent`.
///
/// Backward: `d_base = g*e*b^(e-1)`, `d_exp = g*ln(b)*b^e`.
pub fn pow_tensor<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default>(
    base: &Var<T, B>,
    exponent: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    use super::traits::binary_op;
    binary_op::<T, B, PowTensorForward>(base, exponent)
}

/// Forward tag for tracked tensor-exponent power.
pub struct PowTensorForward;

impl<T: Float + FloatElement, B: coeus_ops::BackendOps<T> + Default>
    super::traits::BinaryAutogradOp<T, B> for PowTensorForward
{
    const OP_NAME: &'static str = "pow_tensor";

    #[inline(always)]
    fn forward(
        base: &Tensor<T, B>,
        exponent: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::pow_tensor(base, exponent, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        base: &Tensor<T, B>,
        exponent: &Tensor<T, B>,
        base_shape: &coeus_core::Shape,
        exp_shape: &coeus_core::Shape,
        input_grads: &[Option<std::sync::Arc<crate::grad_buffer::GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        let y = coeus_ops::pow_tensor(base, exponent, backend)?;
        if let Some(Some(ref g)) = input_grads.first() {
            let one = Tensor::full_on(exponent.shape(), T::ONE, backend)?;
            let e_minus_one = coeus_ops::sub(exponent, &one, backend)?;
            let b_pow = coeus_ops::pow_tensor(base, &e_minus_one, backend)?;
            let d_base = coeus_ops::mul(
                &coeus_ops::mul(grad_out, exponent, backend)?,
                &b_pow,
                backend,
            )?;
            let gl = g.write();
            let reduced = crate::backward::reduce_broadcast(d_base, base_shape);
            coeus_ops::add_assign(gl, &reduced, backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let ln_b = coeus_ops::log(base, backend)?;
            let d_exp = coeus_ops::mul(&coeus_ops::mul(grad_out, &ln_b, backend)?, &y, backend)?;
            let gl = g.write();
            let reduced = crate::backward::reduce_broadcast(d_exp, exp_shape);
            coeus_ops::add_assign(gl, &reduced, backend)?;
        }
        Ok(())
    }
}

/// Tracked numerically stable log-sum-exp over all elements.
#[inline]
pub fn logsumexp_tracked<
    T: Float + FloatElement + leto_ops::Scalar + leto_ops::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    x: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    let m = crate::max_axis(x, 0usize)?;
    let shifted = crate::sub(x, &m)?;
    let exp_shifted = crate::exp(&shifted)?;
    let summed = crate::sum(&exp_shifted)?;
    let log_summed = crate::log(&summed)?;
    crate::add(&log_summed, &m)
}
