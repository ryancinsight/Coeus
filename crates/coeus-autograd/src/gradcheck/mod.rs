//! Numerical verification of backward passes by central finite differences.
//!
//! A backward implementation and a hand-derived closed form typed into a test
//! are not independent oracles when one author produces both: a mistake in the
//! derivation and a matching mistake in the implementation agree with each
//! other. [`fn@gradcheck`] supplies the independent oracle — it never reads the
//! backward code, only the *forward* function, and reconstructs the gradient
//! from forward evaluations alone.
//!
//! # Step-size derivation
//!
//! For the central difference
//!
//! ```text
//! D_h f(x) = (f(x + h) - f(x - h)) / (2h)
//! ```
//!
//! the Taylor expansion `f(x ± h) = f ± h·f' + h²/2·f'' ± h³/6·f''' + O(h⁴)`
//! gives
//!
//! ```text
//! D_h f(x) = f'(x) + (h²/6)·f'''(x) + O(h⁴)
//! ```
//!
//! so the **truncation** error is `(h²/6)·|f'''|`. Independently, each forward
//! evaluation is computed to a relative accuracy of about `ε` (machine epsilon
//! of the scalar type), so each carries an absolute error `≈ ε·|f|`. Their
//! difference divided by `2h` therefore carries a **round-off** error
//! `≈ 2·ε·|f| / (2h) = ε·|f| / h`. The total is
//!
//! ```text
//! E(h) = (h²/6)·|f'''| + ε·|f|/h
//! ```
//!
//! Truncation falls and round-off rises as `h` shrinks, so `E` has an interior
//! minimum. Setting `dE/dh = (h/3)·|f'''| − ε·|f|/h² = 0` gives
//!
//! ```text
//! h* = (3·ε·|f| / |f'''|)^(1/3)   and   E(h*) = O(ε^(2/3))
//! ```
//!
//! With `|f|` and `|f'''|` both `O(1)` in units of the input scale `s`, the
//! `O(1)` constant `3^(1/3) ≈ 1.44` is absorbed and the working step is
//!
//! ```text
//! h = ε^(1/3) · s,   s = max(|x|, 1)
//! ```
//!
//! taken per coordinate so that both small and large inputs get a step matched
//! to their own magnitude. The accuracy floor is then `≈ ε^(2/3)`:
//!
//! | scalar | `ε`       | `h/s ≈ ε^(1/3)` | floor `≈ ε^(2/3)` |
//! |--------|-----------|-----------------|-------------------|
//! | `f64`  | `2.2e-16` | `6.1e-6`        | `3.7e-11`         |
//! | `f32`  | `1.2e-7`  | `4.9e-3`        | `2.4e-5`          |
//!
//! ## Why not a fixed `h = 1e-6`
//!
//! A fixed `h = 1e-6` in `f64` is often paired with the claim that the error is
//! the truncation term `O(h²) ≈ 1e-12`. That reasoning drops the round-off
//! term, which at that step is `ε/h ≈ 2.2e-16 / 1e-6 ≈ 2.2e-10` — over two
//! orders of magnitude *larger* than the quoted truncation, and therefore the
//! term that actually governs the achievable accuracy. It is also ~6x worse
//! than the `3.7e-11` available at `h*`. Tolerances here are derived from
//! `ε^(2/3)`, the real floor, not from the truncation term alone.
//!
//! ## Realized step
//!
//! `x ± h` is rounded to the nearest representable value of `T` before the
//! forward runs, so the step actually taken is not exactly `2h`. The
//! denominator used is the realized `x₊ − x₋` read back from the perturbed
//! tensors, which removes that representation error rather than modelling it.
//!
//! # Zero-gradient guard
//!
//! A comparison in which both the analytic and the numeric gradient are zero
//! establishes nothing — it passes just as readily against a `backward` that
//! writes zeros and against a correct one. This is a real hazard rather than a
//! hypothetical: `sum(softmax(x))` is identically `1`, so its gradient is
//! *exactly* zero for every input, and a gradcheck built on that loss is
//! vacuous. [`fn@gradcheck`] rejects such a comparison with
//! [`GradcheckError::TriviallyZero`]; give the loss a non-uniform weighting so
//! the output Jacobian is actually probed.

mod check;
mod config;
mod error;
mod precision;

pub use check::{gradcheck, gradcheck_with};
pub use config::GradcheckConfig;
pub use error::GradcheckError;
