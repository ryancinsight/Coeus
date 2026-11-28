//! Hand-written tracked special-function nodes.
//!
//! `sinc`, `bessel_j0`/`bessel_j1` and `lgamma` compose host-side
//! `coeus_ops` entry points (and, for `sinc`/`bessel_j1`, host-side
//! singularity patches), which need host-addressable storage -- a bound
//! the `unary_autograd!` macro does not thread. Hand-written, like `pow`
//! and `clamp`, for the same reason.

use super::super::{unary_op, UnaryAutogradOp};
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

// â”€â”€ Sinc â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// `d/dx sinc(x) = cos(x)/x âˆ’ sin(x)/xÂ²`, with the removable singularity
// at `x = 0` patched to `0` (the Taylor expansion `sinc(x) = 1 âˆ’ xÂ²/6 + â€¦`
// has zero slope at the origin).
//
// Hand-written (not `unary_autograd!`) because both forward and backward
// compose the host-side `coeus_ops::sinc` and a `where_cond` patch, which
// need host-addressable storage â€” a bound the macro does not thread.

/// ZST tag for the tracked `sinc` op.
pub struct SincOp;

impl<T, B> UnaryAutogradOp<T, B> for SincOp
where
    T: Float + coeus_core::FloatElement,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    const OP_NAME: &'static str = "sinc";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::sinc(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        // Host-side derivative with the `x = 0 â†’ 0` patch applied before any
        // tensor traffic: the `where_cond` composition cannot patch this
        // singularity because `deriv` itself is NaN at `x = 0`
        // (`cos(0)/0 âˆ’ sin(0)/0`), and `NaN * 0 = NaN` poisons the masked
        // sum. The Taylor slope at the origin is exactly zero.
        let x_host = x.to_contiguous_on(backend)?;
        let g_host = grad_out.to_contiguous_on(backend)?;
        let xs = x_host.as_slice();
        let gs = g_host.as_slice();
        assert_eq!(
            xs.len(),
            gs.len(),
            "sinc backward: grad/tensor len mismatch"
        );
        let mut out: Vec<T> = Vec::with_capacity(xs.len());
        for (&xv, &gv) in xs.iter().zip(gs.iter()) {
            let xf = xv.to_f64();
            let d = if xf == 0.0 {
                0.0
            } else {
                xf.cos() / xf - xf.sin() / (xf * xf)
            };
            out.push(gv * <T as coeus_core::FloatElement>::from_f64(d));
        }
        Tensor::from_slice_on(x.shape().to_vec(), &out, backend)
    }
}

/// Tracked element-wise unnormalized sinc.
#[inline]
pub fn sinc<T: Float + coeus_core::FloatElement, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    unary_op::<T, B, SincOp>(a)
}

// â”€â”€ Bessel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Hand-written (not `unary_autograd!`) because forward and backward compose
// the host-side `coeus_ops::bessel_j0`/`bessel_j1`, which need
// host-addressable storage â€” a bound the macro does not thread.
//
// Identities: `d/dx Jâ‚€(x) = âˆ’Jâ‚(x)`;
// `d/dx Jâ‚(x) = Jâ‚€(x) âˆ’ Jâ‚(x)/x` (quotient form of the recurrence
// `Jâ‚€ + Jâ‚‚ = 2Jâ‚/x`, avoiding a `Jâ‚‚` evaluation). At `x = 0`, `Jâ‚`'s slope
// is exactly `1/2` (series `Jâ‚(x) = x/2 âˆ’ â€¦`); the backward patches that
// lane on the host because `Jâ‚(x)/x` is `0/0 = NaN` there and the
// `where_cond` composition would poison the masked sum with `NaN * 0`.

/// ZST tag for the tracked `bessel_j0` op.
pub struct BesselJ0Op;

impl<T, B> UnaryAutogradOp<T, B> for BesselJ0Op
where
    T: Float + coeus_core::FloatElement + eunomia::RealField,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    const OP_NAME: &'static str = "bessel_j0";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::bessel_j0(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let j1 = coeus_ops::bessel_j1(x, backend)?;
        let neg_j1 = coeus_ops::neg(&j1, backend)?;
        coeus_ops::mul(grad_out, &neg_j1, backend)
    }
}

/// Tracked element-wise Bessel `Jâ‚€`.
#[inline]
pub fn bessel_j0<
    T: Float + coeus_core::FloatElement + eunomia::RealField,
    B: coeus_ops::BackendOps<T> + Default,
>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    unary_op::<T, B, BesselJ0Op>(a)
}

/// ZST tag for the tracked `bessel_j1` op.
pub struct BesselJ1Op;

impl<T, B> UnaryAutogradOp<T, B> for BesselJ1Op
where
    T: Float + coeus_core::FloatElement + eunomia::RealField,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    const OP_NAME: &'static str = "bessel_j1";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::bessel_j1(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let x_host = x.to_contiguous_on(backend)?;
        let g_host = grad_out.to_contiguous_on(backend)?;
        let xs = x_host.as_slice();
        let gs = g_host.as_slice();
        assert_eq!(
            xs.len(),
            gs.len(),
            "bessel_j1 backward: grad/tensor len mismatch"
        );
        let j0 = coeus_ops::bessel_j0(x, backend)?;
        let j1 = coeus_ops::bessel_j1(x, backend)?;
        let j0_host = j0.to_contiguous_on(backend)?;
        let j1_host = j1.to_contiguous_on(backend)?;
        let j0s = j0_host.as_slice();
        let j1s = j1_host.as_slice();
        let mut out: Vec<T> = Vec::with_capacity(xs.len());
        for (((&xv, &gv), &j0v), &j1v) in xs.iter().zip(gs.iter()).zip(j0s.iter()).zip(j1s.iter()) {
            let d = if xv.to_f64() == 0.0 {
                0.5
            } else {
                (j0v - j1v / xv).to_f64()
            };
            out.push(gv * <T as coeus_core::FloatElement>::from_f64(d));
        }
        Tensor::from_slice_on(x.shape().to_vec(), &out, backend)
    }
}

/// Tracked element-wise Bessel `Jâ‚`.
#[inline]
pub fn bessel_j1<
    T: Float + coeus_core::FloatElement + eunomia::RealField,
    B: coeus_ops::BackendOps<T> + Default,
>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    unary_op::<T, B, BesselJ1Op>(a)
}

// â”€â”€ Forward-only (legacy) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// `lgamma_forward` is the legacy forward-only entry point retained for the
// Python binding. New code uses the differentiable `lgamma` below, whose
// backward composes `digamma(x) * grad_out` through `coeus_ops::digamma`
// (Eunomia `FloatElement::digamma`, now available in the pinned provider).

/// Forward-only natural logarithm of the absolute gamma function.
///
/// Retained for the Python binding; prefer the differentiable [`lgamma`].
#[inline]
pub fn lgamma_forward<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Tensor<T, B>, B::Error> {
    let backend = B::default();
    coeus_ops::lgamma(&a.tensor, &backend)
}

/// Tracked natural logarithm of the absolute gamma function.
///
/// `d/dx lgamma(x) = digamma(x)`; backward is `grad_out * digamma(x)`.
/// Domain inherits `digamma`: NaN at negative integers, `âˆ“inf` at `Â±0`.
///
/// The forward-only [`lgamma_forward`] is retained for the Python binding.
///
/// Hand-written (not `unary_autograd!`) because the backward composes
/// `coeus_ops::digamma`, which needs host-addressable storage â€” a bound
/// the macro's generated signature does not thread through.
pub struct LgammaOp;

impl<T, B> UnaryAutogradOp<T, B> for LgammaOp
where
    T: Float + coeus_core::FloatElement,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    const OP_NAME: &'static str = "lgamma";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::lgamma(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let psi = coeus_ops::digamma(x, backend)?;
        coeus_ops::mul(grad_out, &psi, backend)
    }
}

/// Tracked natural logarithm of the absolute gamma function.
#[inline]
pub fn lgamma<T: Float + coeus_core::FloatElement, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    unary_op::<T, B, LgammaOp>(a)
}
