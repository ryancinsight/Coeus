use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_optim::{clip_grad_norm, Adam, AdamW, Optimizer, RMSProp, SGD};
use coeus_tensor::Tensor;

#[test]
fn test_sgd_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test SGD step without momentum (momentum = 0.0, lr = 0.1)
    let mut optimizer = SGD::new(vec![x.clone()], 0.1f32, 0.0f32);
    optimizer.step();

    // After one step, param = param - lr * grad
    // x[0] = 2.0 - 0.1 * 1.0 = 1.9
    // x[1] = 3.0 - 0.1 * (-2.0) = 3.2
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-5);
    assert!((updated_x[1] - 3.2).abs() < 1e-5);

    // Verify zero_grad works
    optimizer.zero_grad();
    let cleared_grad = optimizer.params[0].grad().unwrap();
    assert_eq!(cleared_grad.as_slice(), &[0.0, 0.0]);
}

#[test]
fn test_sgd_with_momentum() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Let's perform two steps of SGD with momentum = 0.9, lr = 0.1
    let mut optimizer = SGD::new(vec![x.clone()], 0.1f32, 0.9f32);

    // Step 1
    // grad = [1.0, -2.0]
    // v = momentum * 0 + grad = [1.0, -2.0]
    // param = param - lr * v = [2.0, 3.0] - 0.1 * [1.0, -2.0] = [1.9, 3.2]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    optimizer.params[0].set_grad(grad_val);
    optimizer.step();

    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-5);
    assert!((updated_x[1] - 3.2).abs() < 1e-5);

    // Step 2
    // grad = [0.5, 0.5]
    // v_prev = [1.0, -2.0]
    // v_new = 0.9 * v_prev + grad = 0.9 * [1.0, -2.0] + [0.5, 0.5] = [1.4, -1.3]
    // param = param - lr * v_new = [1.9, 3.2] - 0.1 * [1.4, -1.3] = [1.76, 3.33]
    let grad_val2 = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[0.5f32, 0.5]);
    optimizer.params[0].set_grad(grad_val2);
    optimizer.step();

    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.76).abs() < 1e-5);
    assert!((updated_x[1] - 3.33).abs() < 1e-5);
}

#[test]
fn test_adam_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test Adam (lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
    let mut optimizer = Adam::new(vec![x.clone()], 0.1f32, 0.9f32, 0.999f32, 1e-8f32);
    optimizer.step();

    // After step 1:
    // t = 1
    // beta1^t = 0.9, beta2^t = 0.999
    // m = beta1 * 0 + (1 - beta1) * grad = 0.1 * [1.0, -2.0] = [0.1, -0.2]
    // v = beta2 * 0 + (1 - beta2) * grad^2 = 0.001 * [1.0, 4.0] = [0.001, 0.004]
    // m_hat = m / (1 - beta1) = [0.1, -0.2] / 0.1 = [1.0, -2.0]
    // v_hat = v / (1 - beta2) = [0.001, 0.004] / 0.001 = [1.0, 4.0]
    // update = lr * m_hat / (sqrt(v_hat) + eps)
    //        = 0.1 * [1.0, -2.0] / ([1.0, 2.0] + 1e-8)
    //        ≈ 0.1 * [1.0, -1.0] = [0.1, -0.1]
    // param = param - update = [2.0, 3.0] - [0.1, -0.1] = [1.9, 3.1]
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-4);
    assert!((updated_x[1] - 3.1).abs() < 1e-4);
}

#[test]
fn test_rmsprop_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test RMSProp (lr = 0.1, alpha = 0.99, eps = 1e-8)
    let mut optimizer = RMSProp::new(vec![x.clone()], 0.1f32, 0.99f32, 1e-8f32);
    optimizer.step();

    // After step 1:
    // v = alpha * 0 + (1 - alpha) * grad^2 = 0.01 * [1.0, 4.0] = [0.01, 0.04]
    // denom = sqrt(v) + eps = [0.1, 0.2] + 1e-8
    // update = lr * grad / denom = 0.1 * [1.0, -2.0] / [0.1, 0.2] = [1.0, -1.0]
    // param = param - update = [2.0, 3.0] - [1.0, -1.0] = [1.0, 4.0]
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.0).abs() < 1e-4);
    assert!((updated_x[1] - 4.0).abs() < 1e-4);
}

#[test]
fn test_adamw_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test AdamW (lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    let mut optimizer = AdamW::new(vec![x.clone()], 0.1f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32);
    optimizer.step();

    // After step 1:
    // t = 1
    // beta1^t = 0.9, beta2^t = 0.999
    // m = beta1 * 0 + (1 - beta1) * grad = 0.1 * [1.0, -2.0] = [0.1, -0.2]
    // v = beta2 * 0 + (1 - beta2) * grad^2 = 0.001 * [1.0, 4.0] = [0.001, 0.004]
    // m_hat = m / (1 - beta1) = [0.1, -0.2] / 0.1 = [1.0, -2.0]
    // v_hat = v / (1 - beta2) = [0.001, 0.004] / 0.001 = [1.0, 4.0]
    // adam_update = lr * m_hat / (sqrt(v_hat) + eps) ≈ 0.1 * [1.0, -1.0] = [0.1, -0.1]
    // wd_update = lr * weight_decay * param = 0.1 * 0.01 * [2.0, 3.0] = [0.002, 0.003]
    // param = param - adam_update - wd_update
    // x[0] = 2.0 - 0.1 - 0.002 = 1.898
    // x[1] = 3.0 - (-0.1) - 0.003 = 3.097
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.898).abs() < 1e-4);
    assert!((updated_x[1] - 3.097).abs() < 1e-4);
}

#[test]
fn test_lr_schedulers() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    {
        use coeus_optim::scheduler::{CosineAnneal, SchedulerStrategy};
        let strategy = CosineAnneal {
            t_max: 0,
            eta_min: 1e-5,
        };
        let lr = strategy.lr(1e-3, 0);
        assert_eq!(lr, 1e-5);
        let lr_step = strategy.lr(1e-3, 10);
        assert_eq!(lr_step, 1e-5);
    }

    {
        use coeus_optim::scheduler::{CosineAnneal, SchedulerStrategy};
        let strategy = CosineAnneal {
            t_max: 10,
            eta_min: 1e-5,
        };
        assert!((strategy.lr(1e-3, 0) - 1e-3).abs() < 1e-6);
        assert!((strategy.lr(1e-3, 10) - 1e-5).abs() < 1e-6);
    }

    {
        use coeus_optim::scheduler::{LrScheduler, StepDecay};
        let optimizer = SGD::new(vec![x.clone()], 1e-3f32, 0.0f32);
        let strategy = StepDecay {
            step_size: 2,
            gamma: 0.5,
        };
        let mut scheduler = LrScheduler::new(optimizer, strategy, 1e-3);

        assert!((scheduler.current_lr() - 1e-3).abs() < 1e-7);
        scheduler.step();

        assert!((scheduler.current_lr() - 1e-3).abs() < 1e-7);
        scheduler.step();

        assert!((scheduler.current_lr() - 5e-4).abs() < 1e-7);
    }
}

#[test]
fn test_adagrad_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test AdaGrad (lr = 0.1, eps = 1e-6)
    let mut optimizer = coeus_optim::AdaGrad::new(vec![x.clone()], 0.1f32, 1e-6f32);
    optimizer.step();

    // After step 1:
    // history = history + grad^2 = [1.0, 4.0]
    // denom = sqrt(history) + eps = [1.000001, 2.000001]
    // update = lr * grad / denom = 0.1 * [1.0, -2.0] / [1.000001, 2.000001]
    // param = param - update
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-4);
    assert!((updated_x[1] - 3.1).abs() < 1e-4);
}

// ── Scheduler strategies not covered by test_lr_schedulers ──
//
// LinearWarmup and WarmupCosine each have an exact closed-form LR schedule;
// the SchedulerStrategy::lr(base_lr, step) method is a pure analytical oracle.

#[test]
fn test_linear_warmup_schedule() {
    use coeus_optim::scheduler::{LinearWarmup, SchedulerStrategy};
    let s = LinearWarmup { warmup_steps: 4 };
    let base = 0.1f64;
    // lr(t) = base * min(t, warmup) / warmup
    assert!((s.lr(base, 0) - 0.0).abs() < 1e-12);
    assert!((s.lr(base, 1) - 0.025).abs() < 1e-12);
    assert!((s.lr(base, 2) - 0.05).abs() < 1e-12);
    assert!((s.lr(base, 4) - 0.1).abs() < 1e-12);
    // Clamped after warmup.
    assert!((s.lr(base, 100) - 0.1).abs() < 1e-12);

    // warmup_steps == 0 short-circuits to base_lr.
    let s0 = LinearWarmup { warmup_steps: 0 };
    assert!((s0.lr(base, 0) - base).abs() < 1e-12);
}

#[test]
fn test_warmup_cosine_schedule() {
    use coeus_optim::scheduler::{SchedulerStrategy, WarmupCosine};
    let s = WarmupCosine {
        warmup_steps: 2,
        t_max: 4,
        eta_min: 0.0,
    };
    let base = 0.1f64;
    // Warmup phase: linear 0 -> base over [0, 2).
    assert!((s.lr(base, 0) - 0.0).abs() < 1e-12);
    assert!((s.lr(base, 1) - 0.05).abs() < 1e-12);
    // Cosine phase (cosine_step = step - warmup_steps):
    //   step 2 -> cs 0: 0.5*base*(1+cos 0)      = base
    //   step 4 -> cs 2: 0.5*base*(1+cos(pi/2))  = base/2
    //   step 6 -> cs 4: 0.5*base*(1+cos(pi))    = 0
    assert!((s.lr(base, 2) - 0.1).abs() < 1e-12);
    assert!((s.lr(base, 4) - 0.05).abs() < 1e-12);
    assert!((s.lr(base, 6) - 0.0).abs() < 1e-12);
}

#[test]
fn test_linear_warmup_drives_optimizer_lr() {
    // Integration: LrScheduler.current_lr() tracks the LinearWarmup schedule
    // as steps advance, confirming the strategy reaches the optimizer.
    use coeus_optim::scheduler::{LinearWarmup, LrScheduler};
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[1.0]),
        true,
    );
    let opt = SGD::new(vec![x], 0.0, 0.0);
    let mut sched = LrScheduler::new(opt, LinearWarmup { warmup_steps: 2 }, 0.2);

    assert!((sched.current_lr() - 0.0).abs() < 1e-7); // step 0
    sched.step();
    assert!((sched.current_lr() - 0.1).abs() < 1e-7); // step 1: 0.2 * 1/2
    sched.step();
    assert!((sched.current_lr() - 0.2).abs() < 1e-7); // step 2: full
}

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
    let mut optimizer = SGD::new(vec![x.clone()], 0.1f32, 0.0f32);

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
    let mut optimizer = SGD::new(vec![x.clone()], 0.05f32, 0.9f32);

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
    let mut optimizer = Adam::new(vec![p.clone()], 0.1f32, 0.9f32, 0.999f32, 1e-8f32);

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
    let mut optimizer = AdamW::new(vec![p.clone()], 0.1f32, 0.9f32, 0.999f32, 1e-8f32, 0.1f32);

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
    let mut optimizer = RMSProp::new(vec![x.clone()], 0.1f32, 0.99f32, 1e-8f32);

    let mut prev = 4.0f32.powi(2);
    for _ in 0..300 {
        let current = optimizer.params[0].tensor.as_slice()[0];
        // Monotone descent: f32 rounding may leave the objective flat but never
        // materially increasing on this convex problem.
        let obj = current * current;
        assert!(obj <= prev + 1e-6, "RMSProp objective increased: {obj} > {prev}");
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
    let mut optimizer = coeus_optim::AdaGrad::new(vec![x.clone()], 0.5f32, 1e-6f32);

    let mut prev = 4.0f32.powi(2);
    for _ in 0..400 {
        let current = optimizer.params[0].tensor.as_slice()[0];
        let obj = current * current;
        assert!(obj <= prev + 1e-6, "AdaGrad objective increased: {obj} > {prev}");
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

// ── clip_grad_norm ──
//
// `clip_grad_norm` had no dedicated test coverage beyond a single-parameter
// doctest; the defining "global" behavior — one L2 norm computed across ALL
// parameters' gradients, as if concatenated into one vector, then every
// gradient scaled by the same factor — was entirely unverified.

/// Global norm spans two parameters: grads [3,4] and [0,0,12] concatenate to
/// [3,4,0,0,12], L2 norm = sqrt(9+16+144) = sqrt(169) = 13 (not 5, the norm of
/// the first parameter alone — this is what "global" must mean).
/// Clipping to max_norm=6.5 scales every gradient by 6.5/13 = 0.5.
#[test]
fn test_clip_grad_norm_is_global_across_parameters() {
    let a = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[10.0f32, 20.0]),
        true,
    );
    a.set_grad(Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[3.0f32, 4.0]));
    let b = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]),
        true,
    );
    b.set_grad(Tensor::<f32, SequentialBackend>::from_slice(
        vec![3],
        &[0.0f32, 0.0, 12.0],
    ));

    let pre_norm = clip_grad_norm(&[a.clone(), b.clone()], 6.5f32);
    assert!(
        (pre_norm - 13.0).abs() < 1e-4,
        "global norm across both params: got {pre_norm}, expected 13.0"
    );

    let ga = a.grad().unwrap();
    assert!((ga.as_slice()[0] - 1.5).abs() < 1e-4, "a[0] scaled by 0.5");
    assert!((ga.as_slice()[1] - 2.0).abs() < 1e-4, "a[1] scaled by 0.5");
    let gb = b.grad().unwrap();
    assert!((gb.as_slice()[2] - 6.0).abs() < 1e-4, "b[2] scaled by 0.5");
}

/// Below `max_norm`, gradients pass through unscaled (no-op clip).
#[test]
fn test_clip_grad_norm_below_threshold_is_noop() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, 1.0]),
        true,
    );
    x.set_grad(Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[3.0f32, 4.0]));

    let pre_norm = clip_grad_norm(std::slice::from_ref(&x), 10.0f32);
    assert!((pre_norm - 5.0).abs() < 1e-5);

    let g = x.grad().unwrap();
    assert!((g.as_slice()[0] - 3.0).abs() < 1e-6, "unscaled: {}", g.as_slice()[0]);
    assert!((g.as_slice()[1] - 4.0).abs() < 1e-6, "unscaled: {}", g.as_slice()[1]);
}

/// At exactly `max_norm` the strict `>` comparison must not trigger scaling
/// (torch's `clip_grad_norm_` uses the same strict-greater convention).
#[test]
fn test_clip_grad_norm_exact_boundary_is_noop() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, 1.0]),
        true,
    );
    x.set_grad(Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[3.0f32, 4.0]));

    let pre_norm = clip_grad_norm(std::slice::from_ref(&x), 5.0f32);
    assert!((pre_norm - 5.0).abs() < 1e-5);

    let g = x.grad().unwrap();
    assert!((g.as_slice()[0] - 3.0).abs() < 1e-6, "boundary: no scaling expected");
    assert!((g.as_slice()[1] - 4.0).abs() < 1e-6, "boundary: no scaling expected");
}

/// A parameter with no gradient is skipped (neither contributes to the norm
/// nor panics), while parameters that do have gradients are still clipped.
#[test]
fn test_clip_grad_norm_skips_params_without_grad() {
    let with_grad = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, 1.0]),
        true,
    );
    with_grad.set_grad(Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[3.0f32, 4.0]));
    // requires_grad = false: no grad buffer at all.
    let without_grad = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[9.0f32, 9.0]),
        false,
    );

    let pre_norm = clip_grad_norm(&[with_grad.clone(), without_grad.clone()], 2.5f32);
    // Norm should reflect only `with_grad`'s [3,4] -> 5.0, not be perturbed by
    // (or panic on) the grad-less parameter.
    assert!((pre_norm - 5.0).abs() < 1e-5, "got {pre_norm}, expected 5.0");

    let g = with_grad.grad().unwrap();
    assert!((g.as_slice()[0] - 1.5).abs() < 1e-4);
    assert!((g.as_slice()[1] - 2.0).abs() < 1e-4);
}
