// ── Operator overloads for Var<T, B> ──
//
// Implements `std::ops::{Add, Sub, Mul, Div, Neg}` so callers can write
// natural arithmetic expressions:
//
//     let z = &x + &y;          // instead of coeus_autograd::add(&x, &y)
//     let w = &x * T::from(2.0); // instead of coeus_autograd::scalar_mul(&x, 2.0)
//     let n = -&x;               // instead of coeus_autograd::neg(&x)
//
// All implementations are `#[inline]` delegations to the existing tracked
// free functions.  The compiler monomorphizes each to the same code as calling
// the free function directly — zero runtime overhead.
//
// Implemented for reference operands (`&Var`) to avoid consuming the variable
// and to match the most common call pattern in forward-pass code.  Owned `Var`
// operand forms can be added as extension traits if needed; they desugar to
// the same underlying calls.
//
// Scalar `Rhs = T` variants via `scalar_mul` / `scalar_add` cover the common
// case of scaling a tensor by a constant without allocating a full broadcast
// tensor.

use std::ops::{Add, Sub, Mul, Div, Neg};
use coeus_core::{Scalar, MoiraiBackend};
use super::arithmetic::{add, sub, mul, div, scalar_mul, scalar_add, scalar_sub, scalar_div};
use super::activation::neg as tracked_neg;
use crate::var::Var;

// ── &Var op &Var ──────────────────────────────────────────────────────────

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Add<&Var<T, B>> for &Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn add(self, rhs: &Var<T, B>) -> Var<T, B> {
        add(self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Sub<&Var<T, B>> for &Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn sub(self, rhs: &Var<T, B>) -> Var<T, B> {
        sub(self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Mul<&Var<T, B>> for &Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn mul(self, rhs: &Var<T, B>) -> Var<T, B> {
        mul(self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Div<&Var<T, B>> for &Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn div(self, rhs: &Var<T, B>) -> Var<T, B> {
        div(self, rhs)
    }
}

// ── Neg for &Var ──────────────────────────────────────────────────────────

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Neg for &Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn neg(self) -> Var<T, B> {
        tracked_neg(self)
    }
}

// ── &Var op T (scalar rhs) ────────────────────────────────────────────────
//
// Only implemented for the default `MoiraiBackend` to avoid the orphan rule:
// the impl `Mul<T> for &Var<T, B>` would require `T: Scalar` but `T` is also
// the right-hand side type, creating a conflict if `B` is generic.  The
// cleanest resolution without a newtype is to restrict to the concrete default
// backend.  Users on custom backends can call `scalar_mul` / `scalar_add`
// directly.
//
// If a generic-backend scalar form is required, the extension-trait pattern
// (a sealed trait re-exporting these impls) is the correct path.

impl<T: Scalar> Mul<T> for &Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn mul(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_mul(self, rhs)
    }
}

impl<T: Scalar> Add<T> for &Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn add(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_add(self, rhs)
    }
}

impl<T: Scalar> Sub<T> for &Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn sub(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_sub(self, rhs)
    }
}

impl<T: Scalar> Div<T> for &Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn div(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_div(self, rhs)
    }
}

// ── Owned Var op &Var ─────────────────────────────────────────────────────
//
// Convenience impls for `Var op &Var` (owned LHS).  Internally borrows the
// owned value to call the reference form.  The clone inside `add`/`mul` etc.
// is a shallow `Arc` clone of the tensor's storage block — O(1) overhead.

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Add<&Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn add(self, rhs: &Var<T, B>) -> Var<T, B> {
        add(&self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Sub<&Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn sub(self, rhs: &Var<T, B>) -> Var<T, B> {
        sub(&self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Mul<&Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn mul(self, rhs: &Var<T, B>) -> Var<T, B> {
        mul(&self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Div<&Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn div(self, rhs: &Var<T, B>) -> Var<T, B> {
        div(&self, rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Neg for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn neg(self) -> Var<T, B> {
        tracked_neg(&self)
    }
}

// ── Owned Var op Owned Var ────────────────────────────────────────────────

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Add<Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn add(self, rhs: Var<T, B>) -> Var<T, B> {
        add(&self, &rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Sub<Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn sub(self, rhs: Var<T, B>) -> Var<T, B> {
        sub(&self, &rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Mul<Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn mul(self, rhs: Var<T, B>) -> Var<T, B> {
        mul(&self, &rhs)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Div<Var<T, B>> for Var<T, B> {
    type Output = Var<T, B>;

    #[inline]
    fn div(self, rhs: Var<T, B>) -> Var<T, B> {
        div(&self, &rhs)
    }
}

// ── Scalar rhs for owned Var (MoiraiBackend) ─────────────────────────────

impl<T: Scalar> Mul<T> for Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn mul(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_mul(&self, rhs)
    }
}

impl<T: Scalar> Add<T> for Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn add(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_add(&self, rhs)
    }
}

impl<T: Scalar> Sub<T> for Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn sub(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_sub(&self, rhs)
    }
}

impl<T: Scalar> Div<T> for Var<T, MoiraiBackend> {
    type Output = Var<T, MoiraiBackend>;

    #[inline]
    fn div(self, rhs: T) -> Var<T, MoiraiBackend> {
        scalar_div(&self, rhs)
    }
}

// Suppress unused-import warning for Float in case it is not directly used
// in this file but pulled in for completeness.
const _: fn() = || {
    let _: Option<f64> = None::<f64>;
};
