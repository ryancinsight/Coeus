use super::CpuUnaryDispatch;
use bytemuck::Pod;
use eunomia::{NumericElement, Pod as EunomiaPod};
use std::ops::Rem;

/// Base numeric trait for all tensor element types.
///
/// # Safety / Design
/// - `Pod` enables zero-copy byte transmutation (bytemuck).
/// - `EunomiaPod` is the canonical device-buffer layout contract consumed by
///   Hephaestus; keeping it on `Scalar` prevents backend seams from accepting
///   a host-only numeric type that cannot cross a device boundary.
/// - `NumericElement` is the eunomia SSOT element vocabulary (constants
///   `ZERO`/`ONE`/etc., `abs`/`sqrt`/`is_finite`/`is_nan`/`to_f64`/
///   `try_from_count`, plus `Add`/`Sub`/`Mul`/`Div`/`Assigns`/`Copy`/`Send`/`Sync`/
///   `'static`/`Debug`/`PartialOrd`); `from_f64` lives one level down on
///   `FloatElement`, and integer `f64` conversion on `IntegerTarget`.
///   Backend `Scalar` traits must extend—not redeclare—this vocabulary.
/// - `Rem<Output=Self>` is per-Scalar (not on `NumericElement`).
/// - Sealed via `eunomia::private::Sealed`.
///
/// # Examples
///
/// Scalar operations on contiguous slices (the SIMD seam):
///
/// ```
/// use coeus_core::Scalar;
///
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [4.0_f32, 5.0, 6.0];
/// let mut out = [0.0_f32; 3];
/// f32::add_slice(&a, &b, &mut out);
/// assert_eq!(out, [5.0, 7.0, 9.0]);
///
/// let dot = f32::dot_slice(&a, &b);
/// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
///
/// let mut acc = [10.0_f32, 10.0, 10.0];
/// f32::axpy_slice(2.0, &a, &mut acc);
/// assert_eq!(acc, [12.0, 14.0, 16.0]); // 10 + 2*[1,2,3]
/// ```
pub trait Scalar:
    NumericElement + CpuUnaryDispatch + Pod + EunomiaPod + Rem<Output = Self> + Clone
{
    /// Return whether every byte in this value's representation is zero.
    ///
    /// Accelerator fills use this representation-level predicate to select a
    /// byte-clear operation without changing signed-zero or NaN payload bits.
    #[inline]
    fn has_zero_bit_pattern(&self) -> bool {
        bytemuck::bytes_of(self).iter().all(|&byte| byte == 0)
    }

    /// Scalar square root.
    fn sqrt_val(self) -> Self;

    /// Scalar absolute value.
    fn abs_val(self) -> Self;

    /// Addition defined for every input: integers wrap modulo 2^bits on
    /// overflow (two's-complement `wrapping_add`); floats follow IEEE 754
    /// (overflow saturates to ±infinity, matching native `+`). Named after
    /// `total_cmp`'s "the well-defined variant of the usual operator" sense,
    /// not because it changes float behavior — float `+` is already total.
    ///
    /// Every `Scalar` implementor states this explicitly (no default): the
    /// reduction operations (`coeus_dist::Sum`/`Product`) fold peer-supplied
    /// values through this method rather than `+`/`*` directly, so an
    /// adversarial or merely large peer value can never panic a debug or
    /// overflow-checked build. Integer wrapping matches the two's-complement
    /// behavior release builds already give unchecked `+`, and the reduction
    /// semantics of MPI/NCCL integer sums.
    fn total_add(self, rhs: Self) -> Self;

    /// Multiplication defined for every input. See [`Scalar::total_add`] for
    /// the wrap/IEEE split and the rationale.
    fn total_mul(self, rhs: Self) -> Self;

    // The slice-kernel default methods below are the backend
    // extension surface — the per-type seam onto `hermes-simd`'s
    // SIMD-effect SSOT. They are mode-stable across the rebase (`hermes-simd`
    // override routes the floating-point slice-bytes-vectorisation at the
    // type-monomorphization site). They MUST stay on `Scalar`, not
    // `NumericElement`, because the kernel surface is backend-agnostic but
    // the SIMD dispatch is backend-specific.

    /// Elementwise `a + b` into `out` over equal-length contiguous slices.
    ///
    /// One of the four per-type seams onto the SIMD-effect SSOT (`hermes-simd`):
    /// the default is a scalar loop; reduced/extended precision and integer types
    /// use it unchanged, while `f32`/`f64` override these to delegate to
    /// `hermes_simd::elementwise_{add,sub,mul,div}`. Each op is independent per
    /// lane, so the SIMD result is bitwise-identical to the scalar default; no
    /// reassociation occurs. `out`, `a`, and `b` must have equal length.
    #[inline]
    fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x + y;
        }
    }

    /// Elementwise `a - b` into `out`. See [`Scalar::add_slice`].
    #[inline]
    fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x - y;
        }
    }

    /// Elementwise `a * b` into `out`. See [`Scalar::add_slice`].
    #[inline]
    fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x * y;
        }
    }

    /// Elementwise `a / b` into `out`. See [`Scalar::add_slice`].
    #[inline]
    fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x / y;
        }
    }

    /// Dot product of two equal-length contiguous slices.
    ///
    /// This is the per-type seam onto the SIMD-effect SSOT for vector products.
    /// The default is a native-precision scalar fold; `f32`/`f64` override to
    /// `hermes_simd::dot`. Floating-point SIMD may reassociate the summation,
    /// so callers that compare against a sequential fold must use an
    /// analytically derived epsilon bound.
    #[inline]
    fn dot_slice(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len(), "dot_slice: length mismatch");

        let mut acc = Self::ZERO;
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc += x * y;
        }
        acc
    }

    /// In-place multiplication of every contiguous slice element by `scalar`.
    ///
    /// This is the per-type seam onto the SIMD-effect SSOT for scalar scaling.
    /// The operation is lane-independent, so native-float SIMD overrides remain
    /// bitwise-identical to the scalar default for ordinary IEEE operands.
    #[inline]
    fn scale_slice(data: &mut [Self], scalar: Self) {
        for value in data {
            *value *= scalar;
        }
    }

    /// Fused scaled accumulate: `out[i] += alpha * x[i]` over a contiguous slice.
    ///
    /// The per-type seam onto the SIMD-effect SSOT for AXPY (BLAS level-1). The
    /// operation is lane-independent; native-float overrides route to
    /// `hermes_simd::axpy` and remain within the type's rounding error of this
    /// scalar default (differential tests use an epsilon bound, not bitwise
    /// equality). `x` and `out` must have equal length.
    #[inline]
    fn axpy_slice(alpha: Self, x: &[Self], out: &mut [Self]) {
        assert_eq!(x.len(), out.len(), "axpy_slice: length mismatch");
        for (o, &xi) in out.iter_mut().zip(x.iter()) {
            *o += alpha * xi;
        }
    }

    /// Sum of a contiguous slice — per-type seam onto the SIMD-effect SSOT.
    ///
    /// Default is a sequential left fold; `f32`/`f64` override to
    /// `hermes_simd::sum`. Summation is associative only approximately in
    /// floating point, so the SIMD result may differ from the sequential fold
    /// within the type's rounding error (differential tests use an epsilon
    /// bound, not bitwise equality). Empty slice sums to `Self::ZERO`.
    #[inline]
    fn sum_slice(s: &[Self]) -> Self {
        match s.split_first() {
            Some((&first, rest)) => {
                let mut acc = first;
                for &v in rest {
                    acc += v;
                }
                acc
            }
            None => Self::ZERO,
        }
    }

    /// Minimum of a non-empty contiguous slice. `f32`/`f64` override to
    /// `hermes_simd::min`. min is exactly associative, so the SIMD result is
    /// value-identical to the sequential fold for non-NaN inputs.
    #[inline]
    fn min_slice(s: &[Self]) -> Self {
        let mut acc = s[0];
        for &v in &s[1..] {
            if v < acc {
                acc = v;
            }
        }
        acc
    }

    /// Maximum of a non-empty contiguous slice. `f32`/`f64` override to
    /// `hermes_simd::max`. See [`Scalar::min_slice`].
    #[inline]
    fn max_slice(s: &[Self]) -> Self {
        let mut acc = s[0];
        for &v in &s[1..] {
            if v > acc {
                acc = v;
            }
        }
        acc
    }
}
