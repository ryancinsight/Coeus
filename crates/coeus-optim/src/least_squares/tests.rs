//! Least-squares solver verification against analytical oracles.
//!
//! Every case has a known closed-form answer or a published minimum, so the
//! assertions check the recovered parameters rather than that the solver merely
//! returned. Tolerances derive from the working scalar's epsilon and the
//! problem's conditioning, never from what the implementation happens to hit.

mod fixtures;

use self::fixtures::{
    sqrt_epsilon, BatchedLinearProblem, DecayProblem, DomainLimitedProblem, LinearProblem,
    RosenbrockProblem,
};
use super::*;
use coeus_core::{FloatElement, NumericElement};

fn linear_problem_is_solved_exactly<T: LeastSquaresScalar>() {
    // 3x2 system with the exact solution [2, -1]: rows (1,0), (0,1), (1,1)
    // against targets 2, -1, 1 are consistent, so the minimum has zero residual.
    let problem = LinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            .map(<T as FloatElement>::from_f64)
            .to_vec(),
        target: [2.0, -1.0, 1.0].map(<T as FloatElement>::from_f64).to_vec(),
        parameters: 2,
    };

    let report = levenberg_marquardt(
        &problem,
        &[T::ZERO, T::ZERO],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("a consistent linear system is solvable");

    let tolerance = sqrt_epsilon::<T>() * <T as FloatElement>::from_f64(100.0);
    assert!(
        (report.parameters[0] - <T as FloatElement>::from_f64(2.0)).abs_val() < tolerance,
        "first parameter must recover 2, got {:?}",
        NumericElement::to_f64(report.parameters[0])
    );
    assert!(
        (report.parameters[1] - <T as FloatElement>::from_f64(-1.0)).abs_val() < tolerance,
        "second parameter must recover -1, got {:?}",
        NumericElement::to_f64(report.parameters[1])
    );
    assert!(
        report.termination.is_converged(),
        "a linear problem must converge by a derived criterion, got {:?}",
        report.termination
    );
}

fn decay_model_recovers_known_parameters<T: LeastSquaresScalar>() {
    // Noise-free measurements from s0 = 1000, d = 0.0007 over a realistic
    // diffusion b-value ladder. An exact fit exists, so the recovered
    // parameters must reproduce the generators.
    let s0 = <T as FloatElement>::from_f64(1000.0);
    let d = <T as FloatElement>::from_f64(7e-4);
    let b_values: Vec<T> = [0.0, 200.0, 400.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0]
        .map(<T as FloatElement>::from_f64)
        .to_vec();
    let measured: Vec<T> = b_values
        .iter()
        .map(|b| {
            <T as FloatElement>::from_f64(
                (-(NumericElement::to_f64(*b)) * NumericElement::to_f64(d)).exp(),
            ) * s0
        })
        .collect();

    let problem = DecayProblem { b_values, measured };

    let report = levenberg_marquardt(
        &problem,
        &[
            <T as FloatElement>::from_f64(500.0),
            <T as FloatElement>::from_f64(2e-3),
        ],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("a noise-free decay fit is solvable");

    // Relative tolerance: the parameters differ by six orders of magnitude, so
    // an absolute bound would be meaningless for one of them.
    let relative = sqrt_epsilon::<T>() * <T as FloatElement>::from_f64(1000.0);
    assert!(
        (report.parameters[0] - s0).abs_val() / s0 < relative,
        "s0 must recover 1000, got {}",
        NumericElement::to_f64(report.parameters[0])
    );
    assert!(
        (report.parameters[1] - d).abs_val() / d < relative,
        "d must recover 7e-4, got {}",
        NumericElement::to_f64(report.parameters[1])
    );
}

fn rosenbrock_reaches_its_published_minimum<T: LeastSquaresScalar>() {
    let report = levenberg_marquardt(
        &RosenbrockProblem,
        &[<T as FloatElement>::from_f64(-1.2), T::ONE],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("Rosenbrock is solvable from the standard start");

    let tolerance = sqrt_epsilon::<T>() * <T as FloatElement>::from_f64(100.0);
    assert!(
        (report.parameters[0] - T::ONE).abs_val() < tolerance
            && (report.parameters[1] - T::ONE).abs_val() < tolerance,
        "must reach the published minimum (1, 1), got ({}, {})",
        NumericElement::to_f64(report.parameters[0]),
        NumericElement::to_f64(report.parameters[1])
    );
}

fn domain_rejection_does_not_abort_the_solve<T: LeastSquaresScalar>() {
    // Starting near zero with a steep Jacobian drives the first trial step
    // negative, where the model is undefined. The solver must damp and recover
    // rather than propagate the domain error.
    let report = levenberg_marquardt(
        &DomainLimitedProblem,
        &[<T as FloatElement>::from_f64(0.01)],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("a domain rejection is a rejected step, not a solver failure");

    let tolerance = sqrt_epsilon::<T>() * <T as FloatElement>::from_f64(1000.0);
    assert!(
        (report.parameters[0] - <T as FloatElement>::from_f64(4.0)).abs_val() < tolerance,
        "sqrt(p) = 2 has the solution p = 4, got {}",
        NumericElement::to_f64(report.parameters[0])
    );
}

fn gradient_criterion_reports_stationarity<T: LeastSquaresScalar>() {
    // Started at the solution, the very first gradient test must fire, before
    // any step is taken.
    let problem = LinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0]
            .map(<T as FloatElement>::from_f64)
            .to_vec(),
        target: [3.0, 5.0].map(<T as FloatElement>::from_f64).to_vec(),
        parameters: 2,
    };

    let report = levenberg_marquardt(
        &problem,
        &[
            <T as FloatElement>::from_f64(3.0),
            <T as FloatElement>::from_f64(5.0),
        ],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("solvable");

    assert_eq!(
        report.termination,
        Termination::GradientTolerance,
        "starting at the minimum must terminate on the gradient criterion"
    );
    assert_eq!(
        report.iterations, 0,
        "no step is needed when the start is already stationary"
    );
}

fn iteration_limit_is_reported_as_unconverged<T: LeastSquaresScalar>() {
    let config = LevenbergMarquardtConfig::<T> {
        max_iterations: 1,
        ..LevenbergMarquardtConfig::default()
    };
    let report = levenberg_marquardt(
        &RosenbrockProblem,
        &[<T as FloatElement>::from_f64(-1.2), T::ONE],
        &config,
    )
    .expect("solvable");

    assert_eq!(report.termination, Termination::IterationLimit);
    assert!(
        !report.termination.is_converged(),
        "an exhausted budget must not be reported as convergence"
    );
}

fn underdetermined_problem_is_rejected<T: LeastSquaresScalar>() {
    let problem = LinearProblem::<T> {
        matrix: [1.0, 1.0].map(<T as FloatElement>::from_f64).to_vec(),
        target: [1.0].map(<T as FloatElement>::from_f64).to_vec(),
        parameters: 2,
    };

    let error = levenberg_marquardt(
        &problem,
        &[T::ZERO, T::ZERO],
        &LevenbergMarquardtConfig::default(),
    )
    .expect_err("one residual cannot determine two parameters");

    assert!(
        matches!(error, SolverError::Underdetermined { .. }),
        "must name the underdetermined contract, got {error}"
    );
}

fn parameter_count_mismatch_is_rejected<T: LeastSquaresScalar>() {
    let problem = LinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0]
            .map(<T as FloatElement>::from_f64)
            .to_vec(),
        target: [1.0, 1.0].map(<T as FloatElement>::from_f64).to_vec(),
        parameters: 2,
    };

    let error = levenberg_marquardt(&problem, &[T::ZERO], &LevenbergMarquardtConfig::default())
        .expect_err("a one-element start cannot initialize two parameters");

    assert!(matches!(
        error,
        SolverError::ParameterCount {
            expected: 2,
            actual: 1
        }
    ));
}

fn batched_linear_problems_recover_independent_minima<T: LeastSquaresScalar>() {
    let problem = BatchedLinearProblem::<T> {
        // A₀ = I, target₀ = [2, -1]; A₁ = [[2, 1], [1, 3]], target₁ = [4, 7].
        matrix: [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0]
            .map(<T as FloatElement>::from_f64)
            .to_vec(),
        target: [2.0, -1.0, 4.0, 7.0]
            .map(<T as FloatElement>::from_f64)
            .to_vec(),
        problems: 2,
        residuals: 2,
        parameters: 2,
    };

    let reports = batched_levenberg_marquardt(
        &problem,
        &[T::ZERO, T::ZERO, T::ZERO, T::ZERO],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("both independent linear systems are solvable");

    assert_eq!(reports.len(), 2, "the leading problem axis is preserved");
    let tolerance = sqrt_epsilon::<T>() * <T as FloatElement>::from_f64(100.0);
    let expected = [[2.0, -1.0], [1.0, 2.0]];
    for (report, expected_parameters) in reports.iter().zip(expected) {
        assert!(report.termination.is_converged());
        for (actual, expected) in report.parameters.iter().zip(expected_parameters) {
            assert!(((*actual - <T as FloatElement>::from_f64(expected)).abs_val()) < tolerance);
        }
    }
}

fn batched_parameter_count_mismatch_is_rejected<T: LeastSquaresScalar>() {
    let problem = BatchedLinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0]
            .map(<T as FloatElement>::from_f64)
            .to_vec(),
        target: [1.0, 1.0].map(<T as FloatElement>::from_f64).to_vec(),
        problems: 1,
        residuals: 2,
        parameters: 2,
    };

    let error =
        batched_levenberg_marquardt(&problem, &[T::ZERO], &LevenbergMarquardtConfig::default())
            .expect_err("the flattened leading-axis buffer has the wrong length");

    assert!(matches!(
        error,
        BatchedSolverError::ParameterCount {
            expected: 2,
            problems: 1,
            parameters_per_problem: 2,
            actual: 1
        }
    ));
}

/// Instantiate every generic body across the shipped scalar types.
///
/// A solver verified at one concrete type is unverified for the rest; `f32` in
/// particular exercises the epsilon-derived tolerances at a precision where a
/// hardcoded bound would silently pass or fail.
macro_rules! scalar_suite {
    ($module:ident, $scalar:ty) => {
        mod $module {
            #[test]
            fn linear_problem_is_solved_exactly() {
                super::linear_problem_is_solved_exactly::<$scalar>();
            }

            #[test]
            fn decay_model_recovers_known_parameters() {
                super::decay_model_recovers_known_parameters::<$scalar>();
            }

            #[test]
            fn rosenbrock_reaches_its_published_minimum() {
                super::rosenbrock_reaches_its_published_minimum::<$scalar>();
            }

            #[test]
            fn domain_rejection_does_not_abort_the_solve() {
                super::domain_rejection_does_not_abort_the_solve::<$scalar>();
            }

            #[test]
            fn gradient_criterion_reports_stationarity() {
                super::gradient_criterion_reports_stationarity::<$scalar>();
            }

            #[test]
            fn iteration_limit_is_reported_as_unconverged() {
                super::iteration_limit_is_reported_as_unconverged::<$scalar>();
            }

            #[test]
            fn underdetermined_problem_is_rejected() {
                super::underdetermined_problem_is_rejected::<$scalar>();
            }

            #[test]
            fn parameter_count_mismatch_is_rejected() {
                super::parameter_count_mismatch_is_rejected::<$scalar>();
            }

            #[test]
            fn batched_linear_problems_recover_independent_minima() {
                super::batched_linear_problems_recover_independent_minima::<$scalar>();
            }

            #[test]
            fn batched_parameter_count_mismatch_is_rejected() {
                super::batched_parameter_count_mismatch_is_rejected::<$scalar>();
            }
        }
    };
}

scalar_suite!(single_precision, f32);
scalar_suite!(double_precision, f64);
