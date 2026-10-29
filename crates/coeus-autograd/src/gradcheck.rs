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

use coeus_core::{CpuAddressableStorage, Float, NumericElement, Scalar};
use coeus_tensor::Tensor;

use crate::var::Var;

/// Round an `f64` back into the differentiated scalar type.
///
/// The counterpart of widening through `NumericElement::to_f64`.
#[inline]
fn narrow<T: Scalar + coeus_core::FloatElement>(value: f64) -> T {
    <T as coeus_core::FloatElement>::from_f64(value)
}

/// Machine epsilon of `T`: the gap between `1` and the next representable
/// value above it.
///
/// Derived from `T`'s own arithmetic rather than read from a constant because
/// [`coeus_core::Float`] does not expose one — it carries `MAX`,
/// `MIN_POSITIVE`, `NAN` and the infinities, but no `EPSILON`. The obvious
/// substitute, `eunomia::RealField::EPSILON`, is unavailable here: `eunomia` is
/// a dev-dependency of this crate, and promoting it to name the constant would
/// put `RealField` into the public bound of [`fn@gradcheck`], excluding the
/// reduced-precision types that implement [`Float`] but not `RealField`.
///
/// Halving a power of two is exact in every binary format until subnormals, and
/// the loop stops many orders of magnitude above them, so the returned value is
/// the exact representable epsilon — not an approximation of it. It converges
/// in one iteration per mantissa bit (53 for `f64`, 11 for `F16`).
fn machine_epsilon<T: Float + coeus_core::FloatElement>() -> f64 {
    let one = T::ONE;
    let mut epsilon = T::ONE;
    loop {
        let halved = narrow::<T>(epsilon.to_f64() * 0.5);
        if one + halved == one {
            return epsilon.to_f64();
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

/// Tuning for a [`fn@gradcheck`] run.
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

/// A gradient Jacobian stored as one flat buffer plus per-input offsets.
///
/// The check compares two Jacobians element by element, so a `Vec<Vec<f64>>`
/// per input buys no random access the comparison needs and costs one
/// allocation plus one pointer hop per input. One allocation for the whole
/// Jacobian keeps the values contiguous — the comparison then walks memory in
/// order, and the offsets are the only per-input state.
struct FlatJacobian {
    /// Every component, concatenated in input order.
    values: Vec<f64>,
    /// `offsets[i]..offsets[i + 1]` is input `i`'s column; length is
    /// `inputs + 1`.
    offsets: Vec<usize>,
}

impl FlatJacobian {
    /// Reserve one flat buffer sized by each input's element count.
    fn with_capacity(numels: impl IntoIterator<Item = usize>) -> Self {
        let mut offsets = vec![0usize];
        for numel in numels {
            let next = offsets[offsets.len() - 1] + numel;
            offsets.push(next);
        }
        let total = offsets[offsets.len() - 1];
        Self {
            values: Vec::with_capacity(total),
            offsets,
        }
    }

    /// Input `index`'s column.
    fn column(&self, index: usize) -> &[f64] {
        &self.values[self.offsets[index]..self.offsets[index + 1]]
    }

    /// How many components input `index` contributes.
    fn column_len(&self, index: usize) -> usize {
        self.offsets[index + 1] - self.offsets[index]
    }

    /// Largest absolute component across every column.
    fn max_abs(&self) -> f64 {
        self.values.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()))
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
/// let x = Tensor::<f64, MoiraiBackend>::from_slice_on([3], &[0.5, -1.25, 2.0], &backend)
///     .expect("invariant: example shape matches data");
/// // A non-uniform weighting keeps the probed gradient away from zero.
/// let w = Var::new(
///     Tensor::<f64, MoiraiBackend>::from_slice_on([3], &[1.0, -2.0, 0.5], &backend)
///         .expect("invariant: example shape matches data"),
///     false,
/// ).expect("invariant: constant example variable needs no gradient buffer");
///
/// gradcheck(&[x], |v| {
///     let product = mul(&v[0], &w).expect("invariant: equal-shaped example operands multiply");
///     sum(&product).expect("invariant: example reduction succeeds")
/// }).expect("invariant: weighted sum is differentiable");
/// ```
pub fn gradcheck<T, B, F>(inputs: &[Tensor<T, B>], loss_fn: F) -> Result<(), GradcheckError>
where
    T: Float + coeus_core::FloatElement,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    F: Fn(&[Var<T, B>]) -> Var<T, B>,
{
    gradcheck_with(inputs, loss_fn, GradcheckConfig::default())
}

/// [`fn@gradcheck`] with an explicit [`GradcheckConfig`].
///
/// # Errors
///
/// As [`fn@gradcheck`].
pub fn gradcheck_with<T, B, F>(
    inputs: &[Tensor<T, B>],
    loss_fn: F,
    config: GradcheckConfig,
) -> Result<(), GradcheckError>
where
    T: Float + coeus_core::FloatElement,
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
        .collect::<Result<_, _>>()
        .map_err(|error: B::Error| GradcheckError::Backward(error.to_string()))?;
    let loss = loss_fn(&tracked);
    if loss.tensor.numel() != 1 {
        return Err(GradcheckError::NonScalarLoss {
            shape: loss.tensor.shape().to_vec(),
        });
    }
    let loss_magnitude = loss.tensor.as_slice()[0].to_f64().abs().max(1.0);
    loss.backward()
        .map_err(|error| GradcheckError::Backward(error.to_string()))?;

    let mut analytic = FlatJacobian::with_capacity(tracked.iter().map(|var| var.tensor.numel()));
    for (index, var) in tracked.iter().enumerate() {
        let grad = var
            .grad()
            .ok_or(GradcheckError::MissingGradient { input: index })?;
        debug_assert_eq!(
            analytic.column_len(index),
            grad.as_slice().len(),
            "gradient buffer and its input must agree in length"
        );
        analytic
            .values
            .extend(grad.as_slice().iter().copied().map(NumericElement::to_f64));
    }

    // Absolute round-off of one finite-difference estimate scales with the loss
    // magnitude: ε·|f|/h = ε^(2/3)·|f|/s, and s = max(|x|,1) >= 1.
    let noise = accuracy_floor * loss_magnitude;
    let tolerance = config.tolerance_scale * noise;
    let zero_floor = ZERO_FLOOR_SCALE * noise;

    // ── Numeric gradient: two untracked forwards per element ──
    let mut numeric = FlatJacobian::with_capacity(inputs.iter().map(|tensor| tensor.numel()));
    for (input_index, tensor) in inputs.iter().enumerate() {
        let perturber = Perturber {
            inputs,
            loss_fn: &loss_fn,
            backend: &backend,
            input_index,
            base: tensor.as_slice().to_vec(),
            shape: tensor.shape().to_vec(),
        };
        debug_assert_eq!(
            numeric.column_len(input_index),
            perturber.base.len(),
            "the finite-difference column must match its input's element count"
        );

        for element in 0..perturber.base.len() {
            let center = perturber.base[element].to_f64();
            let step = step_factor * center.abs().max(1.0);

            let plus = perturber
                .evaluate(element, center + step)
                .map_err(|error| GradcheckError::Backward(error.to_string()))?;
            let minus = perturber
                .evaluate(element, center - step)
                .map_err(|error| GradcheckError::Backward(error.to_string()))?;

            // Realized denominator: `center ± step` was rounded into `T`, so the
            // step actually taken differs from `2·step` by a representation
            // error this divides out exactly.
            let realized = plus.perturbed - minus.perturbed;
            let slope = if realized == 0.0 {
                0.0
            } else {
                (plus.loss - minus.loss) / realized
            };
            numeric.values.push(slope);
        }
    }

    // ── Zero-gradient guard, then component comparison ──
    let max_analytic = analytic.max_abs();
    let max_numeric = numeric.max_abs();
    if max_analytic < zero_floor && max_numeric < zero_floor {
        return Err(GradcheckError::TriviallyZero {
            max_analytic,
            max_numeric,
            floor: zero_floor,
        });
    }

    for input_index in 0..inputs.len() {
        let analytic_column = analytic.column(input_index);
        let numeric_column = numeric.column(input_index);
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
    T: Float + coeus_core::FloatElement,
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
    T: Float + coeus_core::FloatElement,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    F: Fn(&[Var<T, B>]) -> Var<T, B>,
{
    /// Evaluate the loss with `base[element]` replaced by `value`.
    ///
    /// Gradient tracking is off: only the forward value is needed, so no tape
    /// is built for the `2 · numel` perturbation evaluations.
    fn evaluate(&self, element: usize, value: f64) -> Result<Perturbed, B::Error> {
        let mut data = self.base.clone();
        data[element] = narrow::<T>(value);
        let perturbed = data[element].to_f64();
        let replaced = Tensor::from_slice_on(self.shape.clone(), &data, self.backend)?;

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
            .collect::<Result<_, _>>()?;

        Ok(Perturbed {
            loss: (self.loss_fn)(&vars).tensor.as_slice()[0].to_f64(),
            perturbed,
        })
    }
}
mod error;
#[cfg(test)]
mod tests;

pub use error::GradcheckError;
