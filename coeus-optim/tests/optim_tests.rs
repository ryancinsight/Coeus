use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_optim::{Adam, AdamW, Optimizer, RMSProp, SGD};
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
