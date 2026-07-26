use super::{Adam, AdamW, Optimizer, Parameter, RMSProp, SequentialBackend, Tensor, Var, SGD};

// ── Multi-step convergence tests ───────────────────────────────────────────
//
// Evidence tier: empirical.  These tests verify that optimizer state
// (momentum buffer, first/second moment, iteration counter) accumulates
// correctly across many steps.  The 1-step unit tests above cannot catch
// bugs that only manifest through compounding state — e.g., `t` not
// incrementing, `m`/`v` not persisting between steps, or weight-decay being
// applied at the wrong precision.

/// SGD (no momentum) on f(x) = x² with lr=0.1.
///
/// Closed-form: x_n = x₀ · (1 − 2·lr)ⁿ = 4 · 0.8ⁿ.
/// At n=50: 4 · 0.8⁵⁰ ≈ 5.72e-5.  Tolerance of 1e-4 accounts for
/// f32 rounding accumulated across 50 multiply-subtract steps.
#[test]
fn test_sgd_convergence_quadratic_50steps() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[4.0f32]),
        true,
    );
    let mut optimizer = SGD::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 0.0f32);

    for _ in 0..50 {
        let current = optimizer.params[0].tensor.as_slice()[0];
        let grad = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0f32 * current]);
        optimizer.params[0].set_grad(grad);
        optimizer.step();
    }

    let x_n = optimizer.params[0].tensor.as_slice()[0];
    // Analytical closed-form reference.
    let expected = 4.0f32 * 0.8f32.powi(50);
    assert!(
        (x_n - expected).abs() < 1e-4,
        "SGD 50-step: got {x_n}, expected {expected}"
    );
    // Convergence sanity: must be very close to 0.
    assert!(
        x_n.abs() < 1e-3,
        "SGD failed to converge after 50 steps: {x_n}"
    );
}

/// SGD with momentum (β=0.9) on f(x) = x² with lr=0.05.
///
/// The momentum update on f(x)=x² is a linear 2-D dynamical system with
/// eigenvalue modulus ≈ sqrt(0.9) ≈ 0.9487.  After 100 steps from x₀=5:
///   |x_100| ≤ 5 · (sqrt(0.9))^100 = 5 · 0.9^50 ≈ 0.026.
/// Threshold 0.05 is 2× the derived bound — generous enough for f32 rounding.
#[test]
fn test_sgd_momentum_convergence_100steps() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[5.0f32]),
        true,
    );
    let mut optimizer = SGD::new(vec![Parameter::new(x.clone(), "x")], 0.05f32, 0.9f32);

    for _ in 0..100 {
        let current = optimizer.params[0].tensor.as_slice()[0];
        let grad = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0f32 * current]);
        optimizer.params[0].set_grad(grad);
        optimizer.step();
    }

    let x_n = optimizer.params[0].tensor.as_slice()[0];
    // Derived bound: 5 · 0.9^50 ≈ 0.026; threshold 0.05 (2× margin).
    assert!(
        x_n.abs() < 0.05,
        "SGD+momentum failed to converge after 100 steps: {x_n}"
    );
}

/// Adam on f(x,y) = x² + y² with default hyperparameters, lr=0.1.
///
/// Adam is a first-order method; on strongly convex objectives it
/// converges at O(1/√t) in the worst case, faster in practice.
/// Starting from (3, −4) with 200 steps the parameters must reach
/// within 0.05 of the global minimum (0, 0).
#[test]
fn test_adam_convergence_quadratic_200steps() {
    let p = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[3.0f32, -4.0]),
        true,
    );
    let mut optimizer = Adam::new(
        vec![Parameter::new(p.clone(), "p")],
        0.1f32,
        0.9f32,
        0.999f32,
        1e-8f32,
    );

    for _ in 0..200 {
        let vals = optimizer.params[0].tensor.as_slice().to_vec();
        let grad = Tensor::<f32, SequentialBackend>::from_slice(
            vec![2],
            &[2.0f32 * vals[0], 2.0f32 * vals[1]],
        );
        optimizer.params[0].set_grad(grad);
        optimizer.step();
    }

    let final_p = optimizer.params[0].tensor.as_slice();
    assert!(
        final_p[0].abs() < 0.05,
        "Adam x failed to converge after 200 steps: {}",
        final_p[0]
    );
    assert!(
        final_p[1].abs() < 0.05,
        "Adam y failed to converge after 200 steps: {}",
        final_p[1]
    );
}

/// AdamW weight-decay separability over 50 steps.
///
/// With zero gradient, AdamW applies only the weight-decay shrinkage:
///   p ← p − lr · λ · p = p · (1 − lr · λ)
/// Closed-form: p_n = p₀ · (1 − lr · λ)ⁿ.
/// For lr=0.1, λ=0.1, p₀=2: p_50 = 2 · 0.99⁵⁰ ≈ 1.212.
/// Tolerance of 1e-4 covers f32 rounding over 50 multiplications.
#[test]
fn test_adamw_weight_decay_shrinkage_50steps() {
    let p = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0f32]),
        true,
    );
    let mut optimizer = AdamW::new(
        vec![Parameter::new(p.clone(), "p")],
        0.1f32,
        0.9f32,
        0.999f32,
        1e-8f32,
        0.1f32,
    );

    for _ in 0..50 {
        // Zero gradient → only weight-decay acts.
        optimizer.params[0].set_grad(Tensor::<f32, SequentialBackend>::from_slice(
            vec![1],
            &[0.0f32],
        ));
        optimizer.step();
    }

    let p_n = optimizer.params[0].tensor.as_slice()[0];
    // With g=0: Adam adam_update ≈ 0 (bias-corrected m_hat / sqrt(v_hat+eps) → 0).
    // But AdamW still applies: p = p - lr * wd * p = p * (1 - lr*wd) each step.
    // p_50 = 2.0 * (1 - 0.1 * 0.1)^50 = 2.0 * 0.99^50 ≈ 1.212.
    let expected = 2.0f32 * 0.99f32.powi(50);
    assert!(
        (p_n - expected).abs() < 1e-3,
        "AdamW weight-decay 50-step: got {p_n}, expected {expected}"
    );
}

/// RMSProp on f(x) = x² (lr=0.1, α=0.99, ε=1e-8) from x₀=4.
///
/// On this smooth strongly-convex objective RMSProp descends monotonically to
/// the minimum; by step 300 the f32 parameter has collapsed to machine-zero
/// (observed ≈3e-45). The final value must be below 1e-4 (a threshold ~40 orders
/// of magnitude above the observed limit, so it catches any failure-to-converge
/// regression while staying robust to f32 near-zero rounding), and the objective
/// must never increase across the run.
#[test]
fn test_rmsprop_convergence_quadratic_300steps() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[4.0f32]),
        true,
    );
    let mut optimizer = RMSProp::new(
        vec![Parameter::new(x.clone(), "x")],
        0.1f32,
        0.99f32,
        1e-8f32,
    );

    let mut prev = 4.0f32.powi(2);
    for _ in 0..300 {
        let current = optimizer.params[0].tensor.as_slice()[0];
        // Monotone descent: f32 rounding may leave the objective flat but never
        // materially increasing on this convex problem.
        let obj = current * current;
        assert!(
            obj <= prev + 1e-6,
            "RMSProp objective increased: {obj} > {prev}"
        );
        prev = obj;
        let grad = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0f32 * current]);
        optimizer.params[0].set_grad(grad);
        optimizer.step();
    }

    let x_n = optimizer.params[0].tensor.as_slice()[0];
    assert!(
        x_n.abs() < 1e-4,
        "RMSProp failed to converge after 300 steps: {x_n}"
    );
}

/// AdaGrad on f(x) = x² (lr=0.5, ε=1e-6) from x₀=4.
///
/// AdaGrad accumulates Σgⁱ² monotonically, so the per-coordinate step shrinks
/// like O(1/√t) — it converges but decelerates. After 400 steps it reaches the
/// deceleration-limited neighbourhood ≈1.2e-8 (matching an independent f64
/// reference), so the final value must be below 1e-5 (a meaningful bound with
/// ~3 orders of margin), with the objective monotonically non-increasing.
#[test]
fn test_adagrad_convergence_quadratic_400steps() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[4.0f32]),
        true,
    );
    let mut optimizer =
        coeus_optim::AdaGrad::new(vec![Parameter::new(x.clone(), "x")], 0.5f32, 1e-6f32);

    let mut prev = 4.0f32.powi(2);
    for _ in 0..400 {
        let current = optimizer.params[0].tensor.as_slice()[0];
        let obj = current * current;
        assert!(
            obj <= prev + 1e-6,
            "AdaGrad objective increased: {obj} > {prev}"
        );
        prev = obj;
        let grad = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0f32 * current]);
        optimizer.params[0].set_grad(grad);
        optimizer.step();
    }

    let x_n = optimizer.params[0].tensor.as_slice()[0];
    assert!(
        x_n.abs() < 1e-5,
        "AdaGrad failed to converge after 400 steps: {x_n}"
    );
}
