//! Analytical problem fixtures for the least-squares verification.

use super::super::*;
use coeus_core::{FloatElement, NumericElement, Scalar};

/// `sqrt(ε)` for `T`, the accuracy floor of a first-order criterion.
pub(super) fn sqrt_epsilon<T: Scalar + FloatElement>() -> T {
    let mut epsilon = T::ONE;
    let two = <T as FloatElement>::from_f64(2.0);
    while T::ONE + epsilon / two > T::ONE {
        epsilon = epsilon / two;
    }
    epsilon.sqrt_val()
}

/// A linear model `r = A·p - b`, whose Jacobian is constant.
///
/// Gauss-Newton is exact on a linear residual, so this is the oracle that
/// separates "the step direction is right" from "the damping loop eventually
/// wanders to the answer".
pub(super) struct LinearProblem<T> {
    pub(super) matrix: Vec<T>,
    pub(super) target: Vec<T>,
    pub(super) parameters: usize,
}

impl<T: Scalar> LeastSquaresProblem<T> for LinearProblem<T> {
    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn parameter_count(&self) -> usize {
        self.parameters
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        for (row, slot) in residuals.iter_mut().enumerate() {
            let base = row * self.parameters;
            let mut sum = T::ZERO;
            for (column, parameter) in parameters.iter().enumerate() {
                sum += self.matrix[base + column] * *parameter;
            }
            *slot = sum - self.target[row];
        }
        Ok(())
    }

    fn jacobian(&self, _parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        jacobian.copy_from_slice(&self.matrix);
        Ok(())
    }
}

/// Two independent 2×2 linear systems sharing one residual/parameter shape.
pub(super) struct BatchedLinearProblem<T> {
    pub(super) matrix: Vec<T>,
    pub(super) target: Vec<T>,
    pub(super) problems: usize,
    pub(super) residuals: usize,
    pub(super) parameters: usize,
}

impl<T: Scalar> BatchedLeastSquaresProblem<T> for BatchedLinearProblem<T> {
    fn problem_count(&self) -> usize {
        self.problems
    }

    fn residual_count(&self) -> usize {
        self.residuals
    }

    fn parameter_count(&self) -> usize {
        self.parameters
    }

    fn residuals(
        &self,
        problem_index: usize,
        parameters: &[T],
        residuals: &mut [T],
    ) -> Result<(), ProblemError> {
        let matrix_start = problem_index * self.residuals * self.parameters;
        let target_start = problem_index * self.residuals;
        for (row, slot) in residuals.iter_mut().enumerate() {
            let row_start = matrix_start + row * self.parameters;
            let mut sum = T::ZERO;
            for (column, parameter) in parameters.iter().enumerate() {
                sum += self.matrix[row_start + column] * *parameter;
            }
            *slot = sum - self.target[target_start + row];
        }
        Ok(())
    }

    fn jacobian(
        &self,
        problem_index: usize,
        _parameters: &[T],
        jacobian: &mut [T],
    ) -> Result<(), ProblemError> {
        let start = problem_index * self.residuals * self.parameters;
        let end = start + self.residuals * self.parameters;
        jacobian.copy_from_slice(&self.matrix[start..end]);
        Ok(())
    }
}

/// Monoexponential decay `S(b) = s0 · exp(-b · d)`, the shape every diffusion
/// signal model is built from.
///
/// Parameters are `[s0, d]`; residuals are `model - measured`.
pub(super) struct DecayProblem<T> {
    pub(super) b_values: Vec<T>,
    pub(super) measured: Vec<T>,
}

impl<T: Scalar + FloatElement> DecayProblem<T> {
    fn model(&self, b: T, s0: T, d: T) -> T {
        <T as FloatElement>::from_f64(
            (-(NumericElement::to_f64(b)) * NumericElement::to_f64(d)).exp(),
        ) * s0
    }
}

impl<T: Scalar + FloatElement> LeastSquaresProblem<T> for DecayProblem<T> {
    fn residual_count(&self) -> usize {
        self.b_values.len()
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        let (s0, d) = (parameters[0], parameters[1]);
        for (index, slot) in residuals.iter_mut().enumerate() {
            *slot = self.model(self.b_values[index], s0, d) - self.measured[index];
        }
        Ok(())
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        // d/ds0 = exp(-b·d); d/dd = -b · s0 · exp(-b·d)
        let (s0, d) = (parameters[0], parameters[1]);
        for (index, b) in self.b_values.iter().enumerate() {
            let decay = <T as FloatElement>::from_f64(
                (-(NumericElement::to_f64(*b)) * NumericElement::to_f64(d)).exp(),
            );
            jacobian[index * 2] = decay;
            jacobian[index * 2 + 1] = T::ZERO - *b * s0 * decay;
        }
        Ok(())
    }
}

/// Rosenbrock in least-squares form: `r = [10(y - x²), 1 - x]`.
///
/// The published minimum is `(1, 1)` with zero residual. Its curved valley is
/// the standard check that damping adapts rather than taking a fixed step.
pub(super) struct RosenbrockProblem;

impl<T: Scalar + FloatElement> LeastSquaresProblem<T> for RosenbrockProblem {
    fn residual_count(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        let (x, y) = (parameters[0], parameters[1]);
        residuals[0] = <T as FloatElement>::from_f64(10.0) * (y - x * x);
        residuals[1] = T::ONE - x;
        Ok(())
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        let x = parameters[0];
        jacobian[0] = <T as FloatElement>::from_f64(-20.0) * x;
        jacobian[1] = <T as FloatElement>::from_f64(10.0);
        jacobian[2] = <T as FloatElement>::from_f64(-1.0);
        jacobian[3] = T::ZERO;
        Ok(())
    }
}

/// A model undefined for a negative second parameter, to exercise the
/// domain-rejection path.
pub(super) struct DomainLimitedProblem;

impl<T: Scalar + FloatElement> LeastSquaresProblem<T> for DomainLimitedProblem {
    fn residual_count(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        1
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        if parameters[0] < T::ZERO {
            return Err(ProblemError::Domain {
                reason: "parameter must be non-negative".to_owned(),
            });
        }
        residuals[0] = parameters[0].sqrt_val() - <T as FloatElement>::from_f64(2.0);
        residuals[1] = T::ZERO;
        Ok(())
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        let value = parameters[0];
        let guarded = if value > T::ZERO {
            value
        } else {
            <T as FloatElement>::from_f64(1e-12)
        };
        jacobian[0] = T::ONE / (<T as FloatElement>::from_f64(2.0) * guarded.sqrt_val());
        jacobian[1] = T::ZERO;
        Ok(())
    }
}
