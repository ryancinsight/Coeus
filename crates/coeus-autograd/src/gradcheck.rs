//! Numerical verification of backward passes by central finite differences.
//!
//! A backward implementation and a hand-derived closed form typed into a test
//! are not independent oracles when one author produces both: a mistake in the
//! derivation and a matching mistake in the implementation agree with each
//! other. [`gradcheck`] supplies the independent oracle — it never reads the
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
//! vacuous. [`gradcheck`] rejects such a comparison with
//! [`GradcheckError::TriviallyZero`]; give the loss a non-uniform weighting so
//! the output Jacobian is actually probed.

use coeus_core::{CpuAddressableStorage, Float, Scalar};
use coeus_tensor::Tensor;

use crate::var::Var;

/// Widen a scalar to `f64` for the error analysis.
///
/// `Scalar` and its `eunomia::NumericElement` supertrait both carry a
/// `to_f64`, so the bare method call on a `T: Float` is ambiguous; every
/// widening in this module routes through here to name [`Scalar`] once.
#[inline]
fn widen<T: Scalar>(value: T) -> f64 {
    <T as Scalar>::to_f64(value)
}

/// Round an `f64` back into the differentiated scalar type.
///
/// The counterpart to [`widen`], disambiguating `from_f64` the same way.
#[inline]
fn narrow<T: Scalar>(value: f64) -> T {
    <T as Scalar>::from_f64(value)
}

/// Machine epsilon of `T`: the gap between `1` and the next representable
/// value above it.
///
/// Derived from `T`'s own arithmetic rather than read from a constant because
/// [`coeus_core::Float`] does not expose one — it carries `MAX`,
/// `MIN_POSITIVE`, `NAN` and the infinities, but no `EPSILON`. The obvious
/// substitute, `eunomia::RealField::EPSILON`, is unavailable here: `eunomia` is
/// a dev-dependency of this crate, and promoting it to name the constant would
/// put `RealField` into the public bound of [`gradcheck`], excluding the
/// reduced-precision types that implement [`Float`] but not `RealField`.
///
/// Halving a power of two is exact in every binary format until subnormals, and
/// the loop stops many orders of magnitude above them, so the returned value is
/// the exact representable epsilon — not an approximation of it. It converges
/// in one iteration per mantissa bit (53 for `f64`, 11 for `F16`).
fn machine_epsilon<T: Float>() -> f64 {
    let one = T::one();
    let mut epsilon = T::one();
    loop {
        let halved = narrow::<T>(widen(epsilon) * 0.5);
        if one + halved == one {
            return widen(epsilon);
        }
        epsilon = halved;
    }
}

/// Multiplier applied to the `ε^(2/3)` accuracy floor to obtain the default
/// pass tolerance.
///
/// The floor is the *best* attainable accuracy for one perturbed coordinate of
/// an `O(1)`-conditioned function. A real loss reduces over many elements, so
/// its round-off accumulates above that single-coordinate floor; this margin
/// covers that accumulation without approaching the scale of a genuinely wrong
/// gradient, which is typically wrong by an `O(1)` relative amount.
const DEFAULT_TOLERANCE_SCALE: f64 = 64.0;

/// Multiplier applied to the accuracy floor to obtain the zero-gradient floor.
///
/// Held below [`DEFAULT_TOLERANCE_SCALE`] so that a gradient large enough to be
/// meaningfully compared is never rejected as trivially zero.
const ZERO_FLOOR_SCALE: f64 = 8.0;

/// Why a [`gradcheck`] run did not establish agreement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum GradcheckError {
    /// The loss closure returned a non-scalar. A gradcheck needs one scalar
    /// output so that a single backward pass yields the full gradient.
    NonScalarLoss {
        /// Shape the closure actually returned.
        shape: Vec<usize>,
    },
    /// The reverse pass left an input without a gradient, so there is nothing
    /// to compare. Usually the input was not reached by the graph the closure
    /// built.
    MissingGradient {
        /// Index into the `inputs` slice.
        input: usize,
    },
    /// Both gradients are indistinguishable from zero, so the comparison has no
    /// discriminating power. See the module-level zero-gradient guard.
    TriviallyZero {
        /// Largest absolute analytic gradient component observed.
        max_analytic: f64,
        /// Largest absolute numeric gradient component observed.
        max_numeric: f64,
        /// Magnitude below which a gradient counts as zero.
        floor: f64,
    },
    /// An analytic component disagrees with the finite-difference estimate by
    /// more than the derived tolerance.
    Mismatch {
        /// Index into the `inputs` slice.
        input: usize,
        /// Flat element index within that input.
        element: usize,
        /// Component reported by `backward`.
        analytic: f64,
        /// Component reconstructed from forward evaluations.
        numeric: f64,
        /// Largest difference that would have passed.
        tolerance: f64,
    },
    /// The backward pass itself failed.
    Backward(String),
}

impl core::fmt::Display for GradcheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonScalarLoss { shape } => write!(
                f,
                "gradcheck requires a scalar loss; closure returned shape {shape:?}"
            ),
            Self::MissingGradient { input } => write!(
                f,
                "input {input} received no gradient; it is not reachable from the loss"
            ),
            Self::TriviallyZero {
                max_analytic,
                max_numeric,
                floor,
            } => write!(
                f,
                "gradcheck is vacuous: analytic ({max_analytic:.3e}) and numeric \
                 ({max_numeric:.3e}) gradients are both below the zero floor {floor:.3e}. \
                 Weight the loss non-uniformly so the Jacobian is actually probed."
            ),
            Self::Mismatch {
                input,
                element,
                analytic,
                numeric,
                tolerance,
            } => write!(
                f,
                "gradient mismatch at input {input} element {element}: analytic {analytic:.9e} \
                 vs numeric {numeric:.9e} (difference {:.3e} exceeds tolerance {tolerance:.3e})",
                (analytic - numeric).abs()
            ),
            Self::Backward(message) => write!(f, "backward pass failed: {message}"),
        }
    }
}

impl std::error::Error for GradcheckError {}

/// Tuning for a [`gradcheck`] run.
///
/// [`Default`] derives every value from the scalar type's machine epsilon; see
/// the module documentation. Construct with [`GradcheckConfig::default`] and
/// override only what a specific operation's conditioning requires.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GradcheckConfig {
    /// Multiplier on the `ε^(2/3)` accuracy floor that sets the pass tolerance.
    ///
    /// Raise it only for an operation whose forward is measurably worse
    /// conditioned than `O(1)`, and record why at the call site — a tolerance
    /// widened to make a failing check pass is the defect it was meant to
    /// catch.
    pub tolerance_scale: f64,
}

impl Default for GradcheckConfig {
    fn default() -> Self {
        Self {
            tolerance_scale: DEFAULT_TOLERANCE_SCALE,
        }
    }
}

/// Verify a backward pass against central finite differences of its forward.
///
/// `loss_fn` receives one [`Var`] per entry of `inputs` and must return a
/// scalar. It is called once with gradient tracking to obtain the analytic
/// gradient, then twice per input element without tracking to build the
/// finite-difference estimate — so it must be a pure function of the [`Var`]s
/// it is handed. Values the check should *not* differentiate (a `gather` index,
/// a fixed weighting) are captured by the closure instead of being passed in.
///
/// Uses [`GradcheckConfig::default`]; see [`gradcheck_with`] to override.
///
/// # Errors
///
/// Returns [`GradcheckError`] when the loss is not scalar, an input receives no
/// gradient, both gradients are trivially zero, a component disagrees beyond
/// the derived tolerance, or the backward pass fails.
///
/// # Examples
///
/// ```
/// use coeus_autograd::{gradcheck, mul, sum, Var};
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let backend = MoiraiBackend::new();
/// let x = Tensor::<f64, MoiraiBackend>::from_slice_on([3], &[0.5, -1.25, 2.0], &backend);
/// // A non-uniform weighting keeps the probed gradient away from zero.
/// let w = Var::new(
///     Tensor::<f64, MoiraiBackend>::from_slice_on([3], &[1.0, -2.0, 0.5], &backend),
///     false,
/// );
///
/// gradcheck(&[x], |v| sum(&mul(&v[0], &w))).expect("weighted sum is differentiable");
/// ```
pub fn gradcheck<T, B, F>(inputs: &[Tensor<T, B>], loss_fn: F) -> Result<(), GradcheckError>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    F: Fn(&[Var<T, B>]) -> Var<T, B>,
{
    gradcheck_with(inputs, loss_fn, GradcheckConfig::default())
}

/// [`gradcheck`] with an explicit [`GradcheckConfig`].
///
/// # Errors
///
/// As [`gradcheck`].
pub fn gradcheck_with<T, B, F>(
    inputs: &[Tensor<T, B>],
    loss_fn: F,
    config: GradcheckConfig,
) -> Result<(), GradcheckError>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    F: Fn(&[Var<T, B>]) -> Var<T, B>,
{
    let backend = B::default();
    // ε of the scalar actually being differentiated: the step and the accuracy
    // floor both scale with it, so an f32 check gets an f32-appropriate step.
    let epsilon = machine_epsilon::<T>();
    let step_factor = epsilon.cbrt();
    let accuracy_floor = step_factor * step_factor;

    // ── Analytic gradient: one tracked forward, one backward ──
    let tracked: Vec<Var<T, B>> = inputs
        .iter()
        .map(|tensor| Var::new(tensor.clone(), true))
        .collect();
    let loss = loss_fn(&tracked);
    if loss.tensor.numel() != 1 {
        return Err(GradcheckError::NonScalarLoss {
            shape: loss.tensor.shape().to_vec(),
        });
    }
    let loss_magnitude = widen(loss.tensor.as_slice()[0]).abs().max(1.0);
    loss.backward()
        .map_err(|error| GradcheckError::Backward(error.to_string()))?;

    let analytic: Vec<Vec<f64>> = tracked
        .iter()
        .enumerate()
        .map(|(index, var)| {
            var.grad()
                .map(|grad| grad.as_slice().iter().copied().map(widen).collect())
                .ok_or(GradcheckError::MissingGradient { input: index })
        })
        .collect::<Result<_, _>>()?;

    // Absolute round-off of one finite-difference estimate scales with the loss
    // magnitude: ε·|f|/h = ε^(2/3)·|f|/s, and s = max(|x|,1) >= 1.
    let noise = accuracy_floor * loss_magnitude;
    let tolerance = config.tolerance_scale * noise;
    let zero_floor = ZERO_FLOOR_SCALE * noise;

    // ── Numeric gradient: two untracked forwards per element ──
    let mut numeric: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());
    for (input_index, tensor) in inputs.iter().enumerate() {
        let perturber = Perturber {
            inputs,
            loss_fn: &loss_fn,
            backend: &backend,
            input_index,
            base: tensor.as_slice().to_vec(),
            shape: tensor.shape().to_vec(),
        };
        let mut column = Vec::with_capacity(perturber.base.len());

        for element in 0..perturber.base.len() {
            let center = widen(perturber.base[element]);
            let step = step_factor * center.abs().max(1.0);

            let plus = perturber.evaluate(element, center + step);
            let minus = perturber.evaluate(element, center - step);

            // Realized denominator: `center ± step` was rounded into `T`, so the
            // step actually taken differs from `2·step` by a representation
            // error this divides out exactly.
            let realized = plus.perturbed - minus.perturbed;
            let slope = if realized == 0.0 {
                0.0
            } else {
                (plus.loss - minus.loss) / realized
            };
            column.push(slope);
        }
        numeric.push(column);
    }

    // ── Zero-gradient guard, then component comparison ──
    let max_of = |values: &[Vec<f64>]| {
        values
            .iter()
            .flat_map(|column| column.iter())
            .fold(0.0f64, |acc, &v| acc.max(v.abs()))
    };
    let max_analytic = max_of(&analytic);
    let max_numeric = max_of(&numeric);
    if max_analytic < zero_floor && max_numeric < zero_floor {
        return Err(GradcheckError::TriviallyZero {
            max_analytic,
            max_numeric,
            floor: zero_floor,
        });
    }

    for (input_index, (analytic_column, numeric_column)) in
        analytic.iter().zip(numeric.iter()).enumerate()
    {
        for (element, (&a, &n)) in analytic_column
            .iter()
            .zip(numeric_column.iter())
            .enumerate()
        {
            // Absolute floor plus a relative term: a large gradient component
            // carries proportionally larger round-off.
            let bound = tolerance + config.tolerance_scale * accuracy_floor * a.abs().max(n.abs());
            if (a - n).abs() > bound {
                return Err(GradcheckError::Mismatch {
                    input: input_index,
                    element,
                    analytic: a,
                    numeric: n,
                    tolerance: bound,
                });
            }
        }
    }

    Ok(())
}

/// One untracked forward evaluation with a single coordinate replaced.
struct Perturbed {
    /// Scalar loss value at the perturbed point.
    loss: f64,
    /// The perturbed coordinate as actually stored in `T`, after rounding.
    perturbed: f64,
}

/// Everything the perturbed forward evaluations of one input hold constant.
///
/// Bundled rather than passed as a parameter list: the per-evaluation arguments
/// are just the element and its replacement value, and threading the six
/// invariant ones through every call is the parameter-chaining the design rules
/// forbid.
struct Perturber<'a, T, B, F>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
{
    /// All inputs; every index other than `input_index` passes through intact.
    inputs: &'a [Tensor<T, B>],
    /// The loss under test.
    loss_fn: &'a F,
    /// Backend used to rebuild the perturbed tensor.
    backend: &'a B,
    /// Index of the input being perturbed.
    input_index: usize,
    /// Unperturbed contents of that input.
    base: Vec<T>,
    /// Shape of that input.
    shape: Vec<usize>,
}

impl<T, B, F> Perturber<'_, T, B, F>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    F: Fn(&[Var<T, B>]) -> Var<T, B>,
{
    /// Evaluate the loss with `base[element]` replaced by `value`.
    ///
    /// Gradient tracking is off: only the forward value is needed, so no tape
    /// is built for the `2 · numel` perturbation evaluations.
    fn evaluate(&self, element: usize, value: f64) -> Perturbed {
        let mut data = self.base.clone();
        data[element] = narrow::<T>(value);
        let perturbed = widen(data[element]);
        let replaced = Tensor::from_slice_on(self.shape.clone(), &data, self.backend);

        let vars: Vec<Var<T, B>> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                let source = if index == self.input_index {
                    replaced.clone()
                } else {
                    tensor.clone()
                };
                Var::new(source, false)
            })
            .collect();

        Perturbed {
            loss: widen((self.loss_fn)(&vars).tensor.as_slice()[0]),
            perturbed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{mul, softmax, sum};
    use coeus_core::MoiraiBackend;

    fn vector(values: &[f64]) -> Tensor<f64, MoiraiBackend> {
        Tensor::from_slice_on([values.len()], values, &MoiraiBackend::new())
    }

    fn weights(values: &[f64]) -> Var<f64, MoiraiBackend> {
        Var::new(vector(values), false)
    }

    #[test]
    fn accepts_a_correct_gradient() {
        let w = weights(&[1.0, -2.0, 0.5]);
        gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| sum(&mul(&v[0], &w)))
            .expect("d/dx sum(w·x) = w is exact");
    }

    #[test]
    fn rejects_a_vacuous_all_zero_comparison() {
        // sum(softmax(x)) is identically 1, so every gradient component is
        // exactly zero and the comparison discriminates nothing.
        let error = gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| sum(&softmax(&v[0], 0)))
            .expect_err("a zero-vs-zero comparison must be rejected");
        assert!(
            matches!(error, GradcheckError::TriviallyZero { .. }),
            "expected TriviallyZero, got {error:?}"
        );
    }

    #[test]
    fn rejects_a_non_scalar_loss() {
        let w = weights(&[1.0, -2.0, 0.5]);
        let error = gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| mul(&v[0], &w))
            .expect_err("a vector loss must be rejected");
        assert!(
            matches!(error, GradcheckError::NonScalarLoss { ref shape } if shape == &[3]),
            "expected NonScalarLoss([3]), got {error:?}"
        );
    }

    #[test]
    fn detects_a_wrong_gradient() {
        // The oracle must be able to fail: compare the gradient of sum(w·x)
        // against a forward whose weighting differs, and the mismatch must
        // surface rather than be absorbed by the tolerance.
        let truthful = weights(&[1.0, -2.0, 0.5]);
        let analytic_only = gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| {
            // A closure that is *not* a pure function of `v` alone: the tracked
            // call and the perturbed calls disagree, which is exactly the shape
            // of an implementation/derivation divergence.
            if v[0].grad.is_some() {
                sum(&mul(&v[0], &truthful))
            } else {
                sum(&mul(&v[0], &weights(&[1.0, -2.0, 1.5])))
            }
        });
        let error = analytic_only.expect_err("a divergent forward must be detected");
        assert!(
            matches!(error, GradcheckError::Mismatch { .. }),
            "expected Mismatch, got {error:?}"
        );
    }

    #[test]
    fn step_and_floor_follow_machine_epsilon() {
        // The derived epsilon must equal the IEEE constant exactly, for every
        // scalar the check supports — this is the independent oracle for
        // `machine_epsilon`, which cannot read those constants itself.
        assert_eq!(machine_epsilon::<f64>(), f64::EPSILON);
        assert_eq!(machine_epsilon::<f32>(), f64::from(f32::EPSILON));

        // The step is ε^(1/3) and the accuracy floor ε^(2/3); these are the
        // numbers the module doc tabulates, asserted so the derivation and the
        // code cannot drift apart.
        let eps64 = f64::EPSILON;
        assert!((eps64.cbrt() - 6.055e-6).abs() < 1e-9, "f64 step {eps64:e}");
        let floor64 = eps64.cbrt() * eps64.cbrt();
        assert!((floor64 - 3.666e-11).abs() < 1e-14, "f64 floor {floor64:e}");

        let eps32 = f64::from(f32::EPSILON);
        assert!((eps32.cbrt() - 4.921e-3).abs() < 1e-6, "f32 step {eps32:e}");
    }
}
