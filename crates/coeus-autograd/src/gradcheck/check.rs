//! The finite-difference verification runs.

use super::config::GradcheckConfig;
use super::error::GradcheckError;
use super::precision::{machine_epsilon, narrow, widen, ZERO_FLOOR_SCALE};
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, Float};
use coeus_tensor::Tensor;

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
    use super::super::precision::machine_epsilon;
    use super::super::{gradcheck, GradcheckError};
    use crate::ops::{mul, softmax, sum};
    use crate::var::Var;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

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
