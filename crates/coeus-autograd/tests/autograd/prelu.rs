#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::{prelu, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

// prelu(x, w) = x where x > 0, else w*x. PyTorch's kink convention: the
// gradient at x = 0 is w (the negative-branch slope), not 1.

#[test]
fn test_prelu_scalar_weight_forward_and_backward() {
    let backend = MoiraiBackend::new();
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(
            vec![5],
            &[-2.0, -1.0, 0.0, 0.5, 1.0],
            &backend,
        ),
        true,
    );
    let w = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![1], &[0.25], &backend),
        true,
    );
    let out = prelu(&x, &w);
    assert_eq!(
        out.tensor.as_slice(),
        &[-0.5, -0.25, 0.0, 0.5, 1.0],
        "fwd prelu"
    );
    sum(&out)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    // dx = w at x<=0 (INCLUDING the kink x=0), 1 at x>0.
    assert_eq!(
        x.grad().unwrap().as_slice(),
        &[0.25, 0.25, 0.25, 1.0, 1.0],
        "grad_x (kink lands on the negative branch)"
    );
    // dw = sum of x over the x<=0 region: -2 + -1 + 0 = -3.
    assert_eq!(w.grad().unwrap().as_slice(), &[-3.0], "grad_w");
}

/// A per-channel weight `[C]` on a rank-4 `[N,C,H,W]` input must broadcast
/// against the CHANNEL axis (dim 1), not NumPy's default right-aligned
/// trailing axis (which would incorrectly align `C` with `W`).
#[test]
fn test_prelu_per_channel_weight_broadcasts_on_channel_axis() {
    let backend = MoiraiBackend::new();
    // [N=1, C=2, H=1, W=2]: channel 0 = [-1,-2], channel 1 = [-3,-4].
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(
            vec![1, 2, 1, 2],
            &[-1.0, -2.0, -3.0, -4.0],
            &backend,
        ),
        true,
    );
    // Distinct per-channel slopes: channel 0 -> 0.1, channel 1 -> 0.9. If the
    // weight broadcast against the trailing (W) axis instead of the channel
    // axis, channel 1's second element would incorrectly reuse channel 0's
    // slope (weight index 0 has only 2 elements, W-broadcast would wrap/panic
    // on shape mismatch) — this test fails loudly if the axis is wrong.
    let w = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![2], &[0.1, 0.9], &backend),
        true,
    );
    let out = prelu(&x, &w);
    assert_eq!(
        out.tensor.as_slice(),
        &[-0.1, -0.2, -2.7, -3.6],
        "channel 0 scaled by 0.1, channel 1 by 0.9"
    );
    sum(&out)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    // dw[0] = sum over channel 0's elements (both negative): -1 + -2 = -3.
    // dw[1] = sum over channel 1's elements: -3 + -4 = -7.
    assert_eq!(
        w.grad().unwrap().as_slice(),
        &[-3.0, -7.0],
        "grad_w per-channel"
    );
}

/// A scalar weight `[1]` on a rank-4 input broadcasts trivially against every
/// element regardless of axis.
#[test]
fn test_prelu_scalar_weight_broadcasts_over_rank4_input() {
    let backend = MoiraiBackend::new();
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(
            vec![1, 2, 1, 2],
            &[-1.0, -2.0, 3.0, -4.0],
            &backend,
        ),
        true,
    );
    let w = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice_on(vec![1], &[0.5], &backend),
        true,
    );
    let out = prelu(&x, &w);
    assert_eq!(
        out.tensor.as_slice(),
        &[-0.5, -1.0, 3.0, -2.0],
        "scalar weight fwd"
    );
    sum(&out)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    // dw = sum of x over x<=0 elements: -1 + -2 + -4 = -7.
    assert_eq!(
        w.grad().unwrap().as_slice(),
        &[-7.0],
        "grad_w scalar over rank4"
    );
}
