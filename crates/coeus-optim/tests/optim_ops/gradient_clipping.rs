use super::{SequentialBackend, Tensor, Var, clip_grad_norm};

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
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[10.0f32, 20.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    a.set_grad(Tensor::<f32, SequentialBackend>::from_slice(
        vec![2],
        &[3.0f32, 4.0],
    ).expect("construct tensor"));
    let b = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    b.set_grad(Tensor::<f32, SequentialBackend>::from_slice(
        vec![3],
        &[0.0f32, 0.0, 12.0],
    ).expect("construct tensor"));

    let pre_norm = clip_grad_norm(&[a.clone(), b.clone()], 6.5f32).expect("clip gradients");
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
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, 1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    x.set_grad(Tensor::<f32, SequentialBackend>::from_slice(
        vec![2],
        &[3.0f32, 4.0],
    ).expect("construct tensor"));

    let pre_norm = clip_grad_norm(std::slice::from_ref(&x), 10.0f32).expect("clip gradients");
    assert!((pre_norm - 5.0).abs() < 1e-5);

    let g = x.grad().unwrap();
    assert!(
        (g.as_slice()[0] - 3.0).abs() < 1e-6,
        "unscaled: {}",
        g.as_slice()[0]
    );
    assert!(
        (g.as_slice()[1] - 4.0).abs() < 1e-6,
        "unscaled: {}",
        g.as_slice()[1]
    );
}

/// At exactly `max_norm` the strict `>` comparison must not trigger scaling
/// (torch's `clip_grad_norm_` uses the same strict-greater convention).
#[test]
fn test_clip_grad_norm_exact_boundary_is_noop() {
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, 1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    x.set_grad(Tensor::<f32, SequentialBackend>::from_slice(
        vec![2],
        &[3.0f32, 4.0],
    ).expect("construct tensor"));

    let pre_norm = clip_grad_norm(std::slice::from_ref(&x), 5.0f32).expect("clip gradients");
    assert!((pre_norm - 5.0).abs() < 1e-5);

    let g = x.grad().unwrap();
    assert!(
        (g.as_slice()[0] - 3.0).abs() < 1e-6,
        "boundary: no scaling expected"
    );
    assert!(
        (g.as_slice()[1] - 4.0).abs() < 1e-6,
        "boundary: no scaling expected"
    );
}

/// A parameter with no gradient is skipped (neither contributes to the norm
/// nor panics), while parameters that do have gradients are still clipped.
#[test]
fn test_clip_grad_norm_skips_params_without_grad() {
    let with_grad = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, 1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    with_grad.set_grad(Tensor::<f32, SequentialBackend>::from_slice(
        vec![2],
        &[3.0f32, 4.0],
    ).expect("construct tensor"));
    // requires_grad = false: no grad buffer at all.
    let without_grad = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[9.0f32, 9.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let pre_norm =
        clip_grad_norm(&[with_grad.clone(), without_grad.clone()], 2.5f32).expect("clip gradients");
    // Norm should reflect only `with_grad`'s [3,4] -> 5.0, not be perturbed by
    // (or panic on) the grad-less parameter.
    assert!(
        (pre_norm - 5.0).abs() < 1e-5,
        "got {pre_norm}, expected 5.0"
    );

    let g = with_grad.grad().unwrap();
    assert!((g.as_slice()[0] - 1.5).abs() < 1e-4);
    assert!((g.as_slice()[1] - 2.0).abs() < 1e-4);
}
