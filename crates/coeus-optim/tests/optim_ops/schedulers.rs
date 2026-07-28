use super::{Parameter, SGD, SequentialBackend, Tensor, Var};

#[test]
fn test_lr_schedulers() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]).expect("construct tensor");
    let x = Var::new(x_val, true).expect("construct variable");

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
        let optimizer = SGD::new(vec![Parameter::new(x.clone(), "x")], 1e-3f32, 0.0f32)
            .expect("construct SGD optimizer");
        let strategy = StepDecay {
            step_size: 2,
            gamma: 0.5,
        };
        let mut scheduler = LrScheduler::new(optimizer, strategy, 1e-3);

        assert!((scheduler.current_lr() - 1e-3).abs() < 1e-7);
        scheduler.step().expect("run scheduler step");

        assert!((scheduler.current_lr() - 1e-3).abs() < 1e-7);
        scheduler.step().expect("run scheduler step");

        assert!((scheduler.current_lr() - 5e-4).abs() < 1e-7);
    }
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
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let opt = SGD::new(vec![Parameter::new(x, "x")], 0.0, 0.0).expect("construct SGD optimizer");
    let mut sched = LrScheduler::new(opt, LinearWarmup { warmup_steps: 2 }, 0.2);

    assert!((sched.current_lr() - 0.0).abs() < 1e-7); // step 0
    sched.step().expect("run scheduler step");
    assert!((sched.current_lr() - 0.1).abs() < 1e-7); // step 1: 0.2 * 1/2
    sched.step().expect("run scheduler step");
    assert!((sched.current_lr() - 0.2).abs() < 1e-7); // step 2: full
}
