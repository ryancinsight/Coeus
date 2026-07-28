//! Distribution and likelihood loss contracts.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::gaussian_nll_loss;
use coeus_nn::kl_divergence;
use coeus_nn::poisson_nll;
use coeus_tensor::Tensor;

#[test]
fn test_poisson_nll() {
    // log-input form: loss = mean(exp(z) - y*z); d/dz = (exp(z)-y)/n; d/dy = -z/n.
    let zs = [0.0_f64, 1.0, -0.5];
    let ys = [2.0_f64, 0.0, 3.0];
    let n = zs.len() as f64;

    let input = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &zs), true);
    let target = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &ys), true);

    let loss = poisson_nll(&input, &target);
    assert_eq!(loss.tensor.shape(), &[1]);

    let mut expected = 0.0;
    for (&z, &y) in zs.iter().zip(ys.iter()) {
        expected += z.exp() - y * z;
    }
    expected /= n;
    let loss_val = loss.tensor.as_slice()[0];
    assert!(
        (loss_val - expected).abs() <= 1e-12,
        "poisson_nll forward: got {loss_val:.17}, expected {expected:.17}"
    );

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input.grad().expect("input must receive a gradient");
    let target_grad = target.grad().expect("target must receive a gradient");
    for (i, ((&z, &y), (&gz, &gt))) in zs
        .iter()
        .zip(ys.iter())
        .zip(
            input_grad
                .as_slice()
                .iter()
                .zip(target_grad.as_slice().iter()),
        )
        .enumerate()
    {
        let exp_gz = (z.exp() - y) / n;
        let exp_gt = -z / n;
        assert!(
            (gz - exp_gz).abs() <= 1e-12,
            "poisson_nll d/d_input[{i}]: got {gz:.17}, expected {exp_gz:.17}"
        );
        assert!(
            (gt - exp_gt).abs() <= 1e-12,
            "poisson_nll d/d_target[{i}]: got {gt:.17}, expected {exp_gt:.17}"
        );
    }
}

#[test]
fn test_kl_divergence_loss() {
    let input_data = [0.25_f64.ln(), 0.75_f64.ln()];
    let target_data = [0.25_f64, 0.75_f64];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &input_data),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &target_data),
        false,
    );

    let loss = kl_divergence(&input, &target);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!(loss.tensor.as_slice()[0].abs() <= 2.0 * f64::EPSILON);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().expect("invariant: KL input requires grad");
    let grad_slice = grad.as_slice();
    assert!((grad_slice[0] + 0.125).abs() < 1e-12);
    assert!((grad_slice[1] + 0.375).abs() < 1e-12);
}

#[test]
fn test_gaussian_nll_loss() {
    // input=[1,2], target=[1.5,1], var=[0.5,2], full=false.
    // Per element 0.5*((in-t)^2/var + ln var):
    //   i0: 0.5*(0.25/0.5 + ln 0.5) = 0.5*(0.5 - 0.6931472) = -0.0965736
    //   i1: 0.5*(1/2     + ln 2  ) = 0.5*(0.5 + 0.6931472) =  0.5965736
    //   mean = 0.25.
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.0, 2.0]),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.5, 1.0]),
        false,
    );
    let var = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[0.5, 2.0]),
        false,
    );

    let loss = gaussian_nll_loss(&input, &target, &var, false);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!((loss.tensor.as_slice()[0] - 0.25).abs() < 1e-12);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    // d loss/d input_i = (in_i - t_i)/(N*var_i): [-0.5/1, 1/4] = [-0.5, 0.25].
    let grad = input.grad().expect("gnll input grad");
    let g = grad.as_slice();
    assert!((g[0] - (-0.5)).abs() < 1e-12, "gnll grad0: {}", g[0]);
    assert!((g[1] - 0.25).abs() < 1e-12, "gnll grad1: {}", g[1]);

    // full=true adds the constant 0.5*ln(2π) (mean of a per-element constant).
    let input2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.0, 2.0]),
        false,
    );
    let loss_full = gaussian_nll_loss(&input2, &target, &var, true);
    let expected_full = 0.25 + 0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!((loss_full.tensor.as_slice()[0] - expected_full).abs() < 1e-12);
}
