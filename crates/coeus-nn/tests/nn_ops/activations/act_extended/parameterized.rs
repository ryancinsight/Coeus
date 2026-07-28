use super::support::{assert_close_slice, close};
use super::{prelu, Module, MoiraiBackend, Optimizer, PReLU, Tensor, Var, SGD};

// ── PReLU ────────────────────────────────────────────────────────────────────
//
// PReLU/PyTorch contract: y = x if x > 0 else α · x; dx = 1 if x > 0 else α.
// At the kink position x = 0 the gradient equals α, matching PyTorch's
// `F.prelu` semantics (verified empirically: PReLU(0.0).backward() -> 0.25 for
// α = 0.25). The same convention applies to LeakyReLU since both share
// the underlying negative-slope contract in PyTorch.

fn prelu_expected(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

fn prelu_grad_expected(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

#[test]
fn prelu_forward_and_backward() {
    let alpha = 0.25_f64;
    // x = 0.0 is included to exercise the kink position.  Under Coeus'
    // PReLU contract the gradient at x = 0 is α (matches PyTorch).
    let data = vec![-2.0_f64, -1.0, 0.0, 0.5, 1.0];
    let expected: Vec<f64> = data.iter().map(|&x| prelu_expected(x, alpha)).collect();
    let expected_grad: Vec<f64> = data
        .iter()
        .map(|&x| prelu_grad_expected(x, alpha))
        .collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let weight = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1], &[alpha]),
        true,
    );
    let output = prelu(&input, &weight);
    assert_close_slice("prelu_forward", output.tensor.as_slice(), &expected, 1e-12);
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().expect("prelu requires grad");
    assert_close_slice("prelu_backward", grad.as_slice(), &expected_grad, 1e-12);
    // grad_weight = sum of x over the x<=0 region: -2 + -1 + 0 = -3.0.
    let grad_w = weight.grad().expect("prelu weight requires grad");
    close(grad_w.as_slice()[0], -3.0, 1e-12);
}

#[test]
fn prelu_module_weight_learns_via_optimizer_round_trip() {
    // Regression pin (mirrors `test_load_parameters_applies_optimizer_step_to_
    // the_module` for Linear): SGD::step mutates its owned named parameters in
    // place, detached (copy-on-write) from the clone taken via parameters(),
    // so without PReLU::load_parameters writing the update back into
    // module.weight, the module's own field would silently stay unchanged.
    let mut module = PReLU::<f64, MoiraiBackend>::new(1, 0.25);
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[-2.0, 3.0]),
        false,
    );
    let output = module.forward(&x); // prelu([-2,3], w=0.25) = [-0.5, 3.0]
    coeus_autograd::sum(&output)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let lr = 0.1;
    let mut opt = SGD::new(module.named_parameters(), lr, 0.0);
    opt.step();
    module
        .load_named_parameters(&opt.params)
        .expect("optimizer inventory must match module paths");

    // grad_w = sum over x<=0 of x = -2.0 (only the first element).
    // w' = w - lr * grad_w = 0.25 - 0.1*(-2.0) = 0.45.
    close(module.weight.tensor.as_slice()[0], 0.45, 1e-12);

    // The updated weight must actually be used on the next forward pass:
    // prelu([-2,3], w=0.45) = [-0.9, 3.0].
    let x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[-2.0, 3.0]),
        false,
    );
    let output2 = module.forward(&x2);
    assert_close_slice(
        "prelu_after_sgd_step",
        output2.tensor.as_slice(),
        &[-0.9, 3.0],
        1e-12,
    );
}

// ── LeakyReLU subgradient at x = 0 (documented contract parity vs PyTorch) ──

#[test]
fn leaky_relu_kink_at_zero_returns_slope() {
    // PyTorch's `F.leaky_relu(x, 0.01)` returns gradient α at x = 0.
    // Coeus' `leaky_relu` (the same LeakyReluGradTag the autograd PreluNode
    // reuses) must agree at the kink. Both ops therefore share the
    // `x > 0 ? 1 : α` predicate rather than `x >= 0`.
    let data = vec![0.0_f64];
    let slope = 0.01_f64;
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = coeus_nn::leaky_relu(&input, slope);
    assert_close_slice(
        "leaky_relu_kink_out",
        output.tensor.as_slice(),
        &data,
        1e-12,
    );
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().expect("leaky_relu requires grad");
    let expected_grad = vec![slope];
    assert_close_slice("leaky_relu_kink_dx", grad.as_slice(), &expected_grad, 1e-12);
}

// ── Clamp subgradient at x = min and x = max (documented contract parity vs PyTorch) ──

/// PyTorch's `aten::clamp_backward_kernel` returns `1` at both boundaries
/// `x == min` and `x == max` (the indicator `1_{lo <= x <= hi}` is inclusive
/// on both ends). Coeus' `ClampNode::backward` must agree at the kink.
#[test]
fn clamp_kink_at_boundary_returns_one() {
    let lo = -1.0_f64;
    let hi = 2.0_f64;
    let data = vec![-1.0_f64, 2.0_f64]; // exact min and exact max
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = coeus_autograd::clamp(&input, lo, hi);
    // Forward at the boundary is unchanged (clamp(x, x, x) = x).
    assert_close_slice("clamp_kink_out", output.tensor.as_slice(), &data, 1e-12);
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().expect("clamp requires grad");
    // Backward at both kink positions must be 1 per PyTorch convention.
    let expected_grad = vec![1.0_f64, 1.0_f64];
    assert_close_slice("clamp_kink_dx", grad.as_slice(), &expected_grad, 1e-12);
}
