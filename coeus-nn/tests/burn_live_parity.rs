use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor as BurnTensor, TensorData};
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{cross_entropy_loss, softmax, Module};
use coeus_tensor::Tensor as CoeusTensor;

type BurnBackend = NdArray<f32>;

fn dev() -> NdArrayDevice {
    NdArrayDevice::default()
}

fn bvec<const D: usize>(t: BurnTensor<BurnBackend, D>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = 512.0 * f32::EPSILON * (1.0 + want.abs());
        let diff = (got - want).abs();
        assert!(
            diff <= tolerance,
            "{label}[{index}]: actual={got}, expected={want}, diff={diff}, tol={tolerance}"
        );
    }
}

fn assert_close_rel(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: actual={a:.6} expected={e:.6} diff={diff:.2e}"
        );
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────

#[test]
fn softmax_matches_burn_ndarray_reference() {
    let logits = [1.5_f32, 0.5, -0.5, -1.0, 2.0, 0.0];
    let coeus_logits = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &logits),
        false,
    );
    let coeus = softmax(&coeus_logits, 1);
    let burn_logits =
        BurnTensor::<BurnBackend, 2>::from_data(TensorData::new(logits.to_vec(), [2, 3]), &dev());
    let burn_values = bvec(burn::tensor::activation::softmax(burn_logits, 1));
    assert_close("softmax", coeus.tensor.as_slice(), &burn_values);
}

// ── Cross-entropy loss ────────────────────────────────────────────────────────

#[test]
fn cross_entropy_loss_matches_burn_ndarray_reference() {
    let logits = [1.5_f32, 0.5, -0.5, -1.0, 2.0, 0.0];
    let targets = [0_usize, 1_usize];
    let coeus_logits = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &logits),
        true,
    );
    let coeus = cross_entropy_loss(&coeus_logits, &targets);
    let burn_logits =
        BurnTensor::<BurnBackend, 2>::from_data(TensorData::new(logits.to_vec(), [2, 3]), &dev());
    let burn_sm = bvec(burn::tensor::activation::softmax(burn_logits, 1));
    let burn_loss = targets
        .iter()
        .enumerate()
        .map(|(r, &t)| -burn_sm[r * 3 + t].ln())
        .sum::<f32>()
        / targets.len() as f32;
    assert_close("cross_entropy_loss", coeus.tensor.as_slice(), &[burn_loss]);
}

// ── Log-softmax (forward + backward) ────────────────────────────────────────────

#[test]
fn log_softmax_forward_and_backward_match_burn() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();
    let data = vec![1.5f32, 0.5, -0.5, -1.0, 2.0, 0.0];

    // Forward parity vs burn `activation::log_softmax`.
    let fwd_v = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data),
        false,
    );
    let fwd_c = coeus_autograd::log_softmax(&fwd_v, 1);
    let xb_fwd: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close(
        "log_softmax",
        fwd_c.tensor.as_slice(),
        &bvec(burn::tensor::activation::log_softmax(xb_fwd, 1)),
    );

    // Backward parity vs burn autodiff: d/dx sum(log_softmax(x)).
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::log_softmax(&xv, 1)).backward();
    let dx_c = xv.grad().unwrap();
    let xb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &device).require_grad();
    let grads = burn::tensor::activation::log_softmax(xb.clone(), 1)
        .sum()
        .backward();
    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    assert_close("log_softmax_bwd dx", dx_c.as_slice(), &dx_b);
}

// ── Elementwise arithmetic ────────────────────────────────────────────────────

#[test]
fn add_sub_mul_div_match_burn() {
    let backend = SequentialBackend::new();
    let a = vec![4.0f32, -2.0, 6.0, 3.0, 1.0, -0.5];
    let b = vec![2.0f32, 1.0, 3.0, -2.0, 0.5, 4.0];
    let ac = CoeusTensor::from_slice(vec![2, 3], &a);
    let bc = CoeusTensor::from_slice(vec![2, 3], &b);
    let ab: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(a.clone(), [2, 3]), &dev());
    let bb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(b.clone(), [2, 3]), &dev());
    assert_close(
        "add",
        coeus_ops::add(&ac, &bc, &backend).as_slice(),
        &bvec(ab.clone() + bb.clone()),
    );
    assert_close(
        "sub",
        coeus_ops::sub(&ac, &bc, &backend).as_slice(),
        &bvec(ab.clone() - bb.clone()),
    );
    assert_close(
        "mul",
        coeus_ops::mul(&ac, &bc, &backend).as_slice(),
        &bvec(ab.clone() * bb.clone()),
    );
    assert_close(
        "div",
        coeus_ops::div(&ac, &bc, &backend).as_slice(),
        &bvec(ab / bb),
    );
}

// ── Activations ───────────────────────────────────────────────────────────────

#[test]
fn relu_matches_burn() {
    let backend = SequentialBackend::new();
    let data = vec![-2.0f32, 0.0, 1.5, -0.5, 3.0, -1.0];
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close(
        "relu",
        coeus_ops::relu(&CoeusTensor::from_slice(vec![2, 3], &data), &backend).as_slice(),
        &bvec(burn::tensor::activation::relu(xb)),
    );
}

#[test]
fn sigmoid_tanh_match_burn() {
    let backend = SequentialBackend::new();
    let data = vec![-2.0f32, 0.0, 1.0, -1.0, 2.0, -3.0];
    let xc = CoeusTensor::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close(
        "sigmoid",
        coeus_ops::sigmoid(&xc, &backend).as_slice(),
        &bvec(burn::tensor::activation::sigmoid(xb.clone())),
    );
    assert_close(
        "tanh",
        coeus_ops::tanh(&xc, &backend).as_slice(),
        &bvec(xb.tanh()),
    );
}

#[test]
fn gelu_silu_match_burn() {
    let backend = SequentialBackend::new();
    let data = vec![-1.0f32, 0.0, 1.0, 0.5, -0.5, 2.0];
    let xc = CoeusTensor::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close_rel(
        "gelu",
        coeus_ops::gelu(&xc, &backend).as_slice(),
        &bvec(burn::tensor::activation::gelu(xb.clone())),
        1e-4,
    );
    let silu_b = xb.clone() * burn::tensor::activation::sigmoid(xb);
    assert_close(
        "silu",
        coeus_ops::silu(&xc, &backend).as_slice(),
        &bvec(silu_b),
    );
}

#[test]
fn mish_softplus_leaky_relu_match_burn() {
    let backend = SequentialBackend::new();
    let data = vec![-2.0f32, -1.0, -0.5, 0.5, 1.0, 2.0];
    let xc = CoeusTensor::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close(
        "mish",
        coeus_ops::mish(&xc, &backend).as_slice(),
        &bvec(burn::tensor::activation::mish(xb.clone())),
    );
    // Burn `softplus(x, beta)` = (1/beta) ln(1 + exp(beta x)); beta = 1 matches
    // the coeus `softplus` contract ln(1 + exp(x)).
    assert_close(
        "softplus",
        coeus_ops::softplus(&xc, &backend).as_slice(),
        &bvec(burn::tensor::activation::softplus(xb.clone(), 1.0)),
    );
    assert_close(
        "leaky_relu",
        coeus_ops::leaky_relu(&xc, &backend, 0.01).as_slice(),
        &bvec(burn::tensor::activation::leaky_relu(xb, 0.01)),
    );
}

#[test]
fn exp_log_sqrt_neg_abs_match_burn() {
    let backend = SequentialBackend::new();
    let pos = vec![0.1f32, 1.0, 4.0, 9.0];
    let any = vec![-2.0f32, 3.0, -0.5, 1.5];
    let pc = CoeusTensor::from_slice(vec![2, 2], &pos);
    let ac = CoeusTensor::from_slice(vec![2, 2], &any);
    let pb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(pos.clone(), [2, 2]), &dev());
    let ab: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(any.clone(), [2, 2]), &dev());
    assert_close(
        "exp",
        coeus_ops::exp(&pc, &backend).as_slice(),
        &bvec(pb.clone().exp()),
    );
    assert_close(
        "log",
        coeus_ops::log(&pc, &backend).as_slice(),
        &bvec(pb.clone().log()),
    );
    assert_close(
        "sqrt",
        coeus_ops::sqrt(&pc, &backend).as_slice(),
        &bvec(pb.sqrt()),
    );
    assert_close(
        "neg",
        coeus_ops::neg(&ac, &backend).as_slice(),
        &bvec(ab.clone().neg()),
    );
    assert_close(
        "abs",
        coeus_ops::abs(&ac, &backend).as_slice(),
        &bvec(ab.abs()),
    );
}

#[test]
fn sin_cos_match_burn() {
    let backend = SequentialBackend::new();
    let data = vec![0.0f32, 0.5, 1.0, 1.5, 2.0, core::f32::consts::PI];
    let xc = CoeusTensor::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close(
        "sin",
        coeus_ops::sin(&xc, &backend).as_slice(),
        &bvec(xb.clone().sin()),
    );
    assert_close(
        "cos",
        coeus_ops::cos(&xc, &backend).as_slice(),
        &bvec(xb.cos()),
    );
}

// ── Matmul ────────────────────────────────────────────────────────────────────

#[test]
fn matmul_2d_matches_burn() {
    let backend = SequentialBackend::new();
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let out_c = coeus_ops::matmul(
        &CoeusTensor::from_slice(vec![2, 3], &a),
        &CoeusTensor::from_slice(vec![3, 2], &b),
        &backend,
    );
    let ab: BurnTensor<BurnBackend, 2> = BurnTensor::from_data(TensorData::new(a, [2, 3]), &dev());
    let bb: BurnTensor<BurnBackend, 2> = BurnTensor::from_data(TensorData::new(b, [3, 2]), &dev());
    assert_close("matmul_2d", out_c.as_slice(), &bvec(ab.matmul(bb)));
}

#[test]
fn batched_matmul_matches_burn() {
    let backend = SequentialBackend::new();
    let a: Vec<f32> = (0..2 * 3 * 4).map(|x| x as f32 * 0.1).collect();
    let b: Vec<f32> = (0..2 * 4 * 2).map(|x| x as f32 * 0.2 - 0.5).collect();
    let out_c = coeus_ops::matmul(
        &CoeusTensor::from_slice(vec![2, 3, 4], &a),
        &CoeusTensor::from_slice(vec![2, 4, 2], &b),
        &backend,
    );
    let ab: BurnTensor<BurnBackend, 3> =
        BurnTensor::from_data(TensorData::new(a.clone(), [2, 3, 4]), &dev());
    let bb: BurnTensor<BurnBackend, 3> =
        BurnTensor::from_data(TensorData::new(b.clone(), [2, 4, 2]), &dev());
    assert_close_rel(
        "batched_matmul",
        out_c.as_slice(),
        &bvec(ab.matmul(bb)),
        1e-4,
    );
}

// ── Reductions ────────────────────────────────────────────────────────────────

#[test]
fn reductions_match_burn() {
    let backend = SequentialBackend::new();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xc = CoeusTensor::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close(
        "sum_axis0",
        coeus_ops::sum_axis(&xc, 0, &backend).as_slice(),
        &bvec(xb.clone().sum_dim(0)),
    );
    assert_close(
        "sum_axis1",
        coeus_ops::sum_axis(&xc, 1, &backend).as_slice(),
        &bvec(xb.clone().sum_dim(1)),
    );
    assert_close(
        "mean_axis0",
        coeus_ops::mean_axis(&xc, 0, &backend).as_slice(),
        &bvec(xb.clone().mean_dim(0)),
    );
    assert_close(
        "mean_axis1",
        coeus_ops::mean_axis(&xc, 1, &backend).as_slice(),
        &bvec(xb.clone().mean_dim(1)),
    );
    assert_close(
        "max_axis0",
        coeus_ops::max_axis(&xc, 0, &backend).as_slice(),
        &bvec(xb.clone().max_dim(0)),
    );
    assert_close(
        "min_axis0",
        coeus_ops::min_axis(&xc, 0, &backend).as_slice(),
        &bvec(xb.min_dim(0)),
    );
}

#[test]
fn statistical_ops_match_burn() {
    let backend = SequentialBackend::new();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xc = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());

    // Global var / std use a closed-form Bessel-corrected oracle because Burn
    // 0.16 exposes `var(dim)`, not global `var()` / `std()`.
    let v_global = coeus_ops::var(&xc, true, &backend);
    let v_global_ref = 3.5f32;
    assert_close("var_global", &[v_global], &[v_global_ref]);
    let s_global = coeus_ops::std_dev(&xc, true, &backend);
    assert_close("std_global", &[s_global], &[v_global_ref.sqrt()]);

    // Per-axis var matches Burn's Bessel-corrected `var(dim)` API; per-axis std
    // is the square root of that same variance oracle.
    let v_axis1 = coeus_ops::var_axis(&xc, 1, true, &backend);
    let v_burn_axis1 = xb.clone().var(1).into_data().to_vec().unwrap();
    assert_close("var_axis1", v_axis1.as_slice(), &v_burn_axis1);
    let s_axis1 = coeus_ops::std_dev_axis(&xc, 1, true, &backend);
    let s_burn_axis1: Vec<f32> = v_burn_axis1.iter().map(|v| v.sqrt()).collect();
    assert_close("std_axis1", s_axis1.as_slice(), &s_burn_axis1);

    // Population (unbiased=false) global var matches `var` after dividing
    // by N instead of N-1: Bessel correction factor (N-1)/N.
    let v_biased = coeus_ops::var(&xc, false, &backend);
    let n = xc.numel() as f32;
    assert_close(
        "var_biased_global",
        &[v_biased],
        &[v_global_ref * (n - 1.0) / n],
    );

    // L2 norm matches Burn's `l2_norm` over flattened input.
    let n2 = coeus_ops::norm(&xc, &backend);
    let n_burn = bvec(xb.clone().powf_scalar(2.0).sum())[0].sqrt();
    assert_close("norm_l2", &[n2], &[n_burn]);
}

// ── Linear layer (same weights) ───────────────────────────────────────────────

#[test]
fn linear_forward_matches_burn() {
    let backend = SequentialBackend::new();
    let x = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 2.5];
    let w = vec![0.5f32, -0.5, 1.0, 0.0, 2.0, -1.0];
    let b = vec![0.2f32, -0.1];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &x),
        false,
    );
    let mut lin = coeus_nn::Linear::<f32, SequentialBackend>::new(3, 2, true);
    lin.weight = Var::new(CoeusTensor::from_slice(vec![2, 3], &w), true);
    lin.bias = Some(Var::new(CoeusTensor::from_slice(vec![2], &b), true));
    let out_c = lin.forward(&xv);
    let xb: BurnTensor<BurnBackend, 2> = BurnTensor::from_data(TensorData::new(x, [2, 3]), &dev());
    let wb: BurnTensor<BurnBackend, 2> = BurnTensor::from_data(TensorData::new(w, [2, 3]), &dev());
    let bb: BurnTensor<BurnBackend, 1> = BurnTensor::from_data(TensorData::new(b, [2]), &dev());
    let out_b = xb.matmul(wb.transpose()) + bb.unsqueeze::<2>();
    assert_close("linear_forward", out_c.tensor.as_slice(), &bvec(out_b));
    let _ = backend;
}

#[test]
fn linear_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let x = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 2.5];
    let w = vec![0.5f32, -0.5, 1.0, 0.0, 2.0, -1.0];
    let b = vec![0.2f32, -0.1];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &x),
        true,
    );
    let mut lin = coeus_nn::Linear::<f32, SequentialBackend>::new(3, 2, true);
    lin.weight = Var::new(CoeusTensor::from_slice(vec![2, 3], &w), true);
    lin.bias = Some(Var::new(CoeusTensor::from_slice(vec![2], &b), true));
    coeus_autograd::sum(&lin.forward(&xv)).backward();
    let dx_c = xv.grad().unwrap();
    let dw_c = lin.weight.grad().unwrap();
    let db_c = lin.bias.as_ref().unwrap().grad().unwrap();
    let device: NdArrayDevice = Default::default();
    let xb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(x, [2, 3]), &device).require_grad();
    let wb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(w, [2, 3]), &device).require_grad();
    let bt: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(b, [2]), &device).require_grad();
    let grads = (xb.clone().matmul(wb.clone().transpose()) + bt.clone().unsqueeze::<2>())
        .sum()
        .backward();
    assert_close(
        "linear_bwd dx",
        dx_c.as_slice(),
        &xb.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    assert_close(
        "linear_bwd dw",
        dw_c.as_slice(),
        &wb.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    assert_close(
        "linear_bwd db",
        db_c.as_slice(),
        &bt.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
}

// ── ReLU backward ─────────────────────────────────────────────────────────────

#[test]
fn relu_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let data = vec![-1.0f32, 2.0, -0.5, 0.0, 3.0, -2.0];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::relu(&xv)).backward();
    let dx_c = xv.grad().unwrap();
    let device: NdArrayDevice = Default::default();
    let xb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &device).require_grad();
    let grads = burn::tensor::activation::relu(xb.clone()).sum().backward();
    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    assert_close("relu_bwd dx", dx_c.as_slice(), &dx_b);
}

// ── Activation backward (sigmoid/tanh/silu/gelu) vs Burn autodiff ───────────────

#[test]
fn activation_backward_match_burn() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();
    let data = vec![-1.5f32, -0.5, 0.25, 0.5, 1.5, 2.0];

    let coeus_var = || {
        Var::new(
            CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data),
            true,
        )
    };
    let burn_var = || -> BurnTensor<AB, 2> {
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &device).require_grad()
    };
    let burn_grad = |xb: &BurnTensor<AB, 2>, grads| {
        xb.grad(grads).unwrap().into_data().to_vec::<f32>().unwrap()
    };

    // sigmoid
    let xv = coeus_var();
    coeus_autograd::sum(&coeus_autograd::sigmoid(&xv)).backward();
    let xb = burn_var();
    let g = burn::tensor::activation::sigmoid(xb.clone())
        .sum()
        .backward();
    assert_close(
        "sigmoid_bwd",
        xv.grad().unwrap().as_slice(),
        &burn_grad(&xb, &g),
    );

    // tanh
    let xv = coeus_var();
    coeus_autograd::sum(&coeus_autograd::tanh(&xv)).backward();
    let xb = burn_var();
    let g = xb.clone().tanh().sum().backward();
    assert_close(
        "tanh_bwd",
        xv.grad().unwrap().as_slice(),
        &burn_grad(&xb, &g),
    );

    // silu
    let xv = coeus_var();
    coeus_autograd::sum(&coeus_autograd::silu(&xv)).backward();
    let xb = burn_var();
    let g = burn::tensor::activation::silu(xb.clone()).sum().backward();
    assert_close(
        "silu_bwd",
        xv.grad().unwrap().as_slice(),
        &burn_grad(&xb, &g),
    );

    // Burn 0.16 GELU forward is exact-erf, but its default GELU backward uses
    // the tanh-approximation derivative. Compare that contract to Coeus'
    // explicit tanh-approximation GELU rather than weakening the exact-GELU
    // gradient bound.
    let xv = coeus_var();
    coeus_autograd::sum(&coeus_autograd::gelu_tanh(&xv)).backward();
    let xb = burn_var();
    let g = burn::tensor::activation::gelu(xb.clone()).sum().backward();
    assert_close(
        "gelu_tanh_bwd",
        xv.grad().unwrap().as_slice(),
        &burn_grad(&xb, &g),
    );
}

// ── Sin/Cos backward ──────────────────────────────────────────────────────────

#[test]
fn sin_cos_backward_match_burn() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let data = vec![0.5f32, 1.0, 1.5, 2.0];
    let device: NdArrayDevice = Default::default();
    // sin backward
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::sin(&xv)).backward();
    let dx_sin_c = xv.grad().unwrap();
    let xb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(data.clone(), [4]), &device).require_grad();
    let grads_sin = xb.clone().sin().sum().backward();
    assert_close(
        "sin_bwd dx",
        dx_sin_c.as_slice(),
        &xb.grad(&grads_sin)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    // cos backward
    let xv2 = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::cos(&xv2)).backward();
    let dx_cos_c = xv2.grad().unwrap();
    let xb2: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(data.clone(), [4]), &device).require_grad();
    let grads_cos = xb2.clone().cos().sum().backward();
    assert_close(
        "cos_bwd dx",
        dx_cos_c.as_slice(),
        &xb2.grad(&grads_cos)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
}

// ── Matmul backward ───────────────────────────────────────────────────────────

#[test]
fn matmul_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![0.5f32, -0.5, 1.0, 0.0, 2.0, -1.0];
    let av = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a),
        true,
    );
    let bv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3, 2], &b),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::matmul(&av, &bv)).backward();
    let device: NdArrayDevice = Default::default();
    let ab: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(a, [2, 3]), &device).require_grad();
    let bb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(b, [3, 2]), &device).require_grad();
    let grads = ab.clone().matmul(bb.clone()).sum().backward();
    assert_close(
        "matmul_bwd da",
        av.grad().unwrap().as_slice(),
        &ab.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    assert_close(
        "matmul_bwd db",
        bv.grad().unwrap().as_slice(),
        &bb.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
}

// ── MSE loss ──────────────────────────────────────────────────────────────────

#[test]
fn mse_loss_matches_burn() {
    use burn::nn::loss::{MseLoss, Reduction};
    let pred = vec![1.0f32, 2.0, 3.0, 0.5];
    let target = vec![1.5f32, 1.5, 4.0, 0.0];
    let pv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &pred),
        false,
    );
    let tv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &target),
        false,
    );
    let loss_c = coeus_nn::mse_loss(&pv, &tv);
    let pb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(pred, [2, 2]), &dev());
    let tb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(target, [2, 2]), &dev());
    let loss_b: f32 = MseLoss::new()
        .forward(pb, tb, Reduction::Mean)
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert_close("mse_loss", loss_c.tensor.as_slice(), &[loss_b]);
}

#[test]
fn probability_loss_forward_and_backward_match_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::nn::loss::{BinaryCrossEntropyLossConfig, HuberLossConfig, MseLoss, Reduction};
    use burn::tensor::Int;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let pred = vec![0.2_f32, 0.7, 0.6, 0.9];
    let target = vec![0.0_f32, 1.0, 1.0, 0.0];
    let pred_var = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &pred),
        true,
    );
    let target_var = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &target),
        false,
    );
    coeus_nn::binary_cross_entropy(&pred_var, &target_var, 1.0e-7).backward();
    let pred_b: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(pred.clone(), [4]), &device).require_grad();
    let target_b: BurnTensor<AB, 1, Int> =
        BurnTensor::from_data(TensorData::new(vec![0_i64, 1, 1, 0], [4]), &device);
    let grads = BinaryCrossEntropyLossConfig::new()
        .init(&device)
        .forward(pred_b.clone(), target_b)
        .backward();
    assert_close(
        "binary_cross_entropy",
        coeus_nn::binary_cross_entropy(&pred_var, &target_var, 1.0e-7)
            .tensor
            .as_slice(),
        &bvec(BinaryCrossEntropyLossConfig::new().init(&device).forward(
            BurnTensor::<BurnBackend, 1>::from_data(TensorData::new(pred.clone(), [4]), &dev()),
            BurnTensor::<BurnBackend, 1, Int>::from_data(
                TensorData::new(vec![0_i64, 1, 1, 0], [4]),
                &dev(),
            ),
        )),
    );
    assert_close(
        "binary_cross_entropy_bwd",
        pred_var.grad().unwrap().as_slice(),
        &pred_b
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );

    let y_hat = vec![-1.5_f32, -0.25, 0.5, 2.0];
    let y = vec![0.0_f32, 0.25, -0.5, 1.25];
    let y_hat_var = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &y_hat),
        true,
    );
    let y_var = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &y),
        false,
    );
    coeus_nn::mse_loss(&y_hat_var, &y_var).backward();
    let y_hat_b: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(y_hat.clone(), [4]), &device).require_grad();
    let y_b: BurnTensor<AB, 1> = BurnTensor::from_data(TensorData::new(y.clone(), [4]), &device);
    let grads = MseLoss::new()
        .forward(y_hat_b.clone(), y_b, Reduction::Mean)
        .backward();
    assert_close(
        "mse_loss_bwd",
        y_hat_var.grad().unwrap().as_slice(),
        &y_hat_b
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );

    let huber_pred = vec![-1.5_f32, -0.25, 0.25, 1.75];
    let huber_target = vec![0.0_f32, 0.25, 0.0, 0.5];
    let huber_pred_var = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &huber_pred),
        true,
    );
    let huber_target_var = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &huber_target),
        false,
    );
    coeus_nn::huber_loss(&huber_pred_var, &huber_target_var, 1.0).backward();
    let huber_pred_b: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(huber_pred.clone(), [4]), &device).require_grad();
    let huber_target_b: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(huber_target.clone(), [4]), &device);
    let grads = HuberLossConfig::new(1.0)
        .init()
        .forward(huber_pred_b.clone(), huber_target_b, Reduction::Mean)
        .backward();
    assert_close(
        "huber_loss_bwd",
        huber_pred_var.grad().unwrap().as_slice(),
        &huber_pred_b
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

#[test]
fn layernorm_forward_matches_burn_manual() {
    let backend = SequentialBackend::new();
    let x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0];
    let w = vec![1.2f32, 0.8, 1.0, 0.9];
    let b = vec![0.1f32, -0.1, 0.2, 0.0];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &x),
        false,
    );
    let mut ln = coeus_nn::LayerNorm::<f32, SequentialBackend>::new(4, 1e-5);
    ln.weight = Var::new(CoeusTensor::from_slice(vec![4], &w), true);
    ln.bias = Var::new(CoeusTensor::from_slice(vec![4], &b), true);
    let out_c = ln.forward(&xv);
    let xb: BurnTensor<BurnBackend, 2> = BurnTensor::from_data(TensorData::new(x, [2, 4]), &dev());
    let wb: BurnTensor<BurnBackend, 1> = BurnTensor::from_data(TensorData::new(w, [4]), &dev());
    let bk: BurnTensor<BurnBackend, 1> = BurnTensor::from_data(TensorData::new(b, [4]), &dev());
    let mean = xb.clone().mean_dim(1);
    let xc = xb - mean;
    let std = (xc.clone().powf_scalar(2.0).mean_dim(1) + 1e-5f32).sqrt();
    let out_b = xc / std * wb.unsqueeze::<2>() + bk.unsqueeze::<2>();
    assert_close_rel("layernorm_fwd", out_c.tensor.as_slice(), &bvec(out_b), 1e-4);
    let _ = backend;
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_forward_matches_burn_manual() {
    let backend = SequentialBackend::new();
    let x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0];
    let w = vec![1.0f32, 0.8, 1.2, 0.9];
    let eps = 1e-6f64;
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &x),
        false,
    );
    let mut rms = coeus_nn::RMSNorm::<f32, SequentialBackend>::new(4, eps);
    rms.weight = Var::new(CoeusTensor::from_slice(vec![4], &w), true);
    let out_c = rms.forward(&xv);
    // Manual RMSNorm: x / rms(x) * weight  where rms(x) = sqrt(mean(x^2) + eps)
    let xb: BurnTensor<BurnBackend, 2> = BurnTensor::from_data(TensorData::new(x, [2, 4]), &dev());
    let wb: BurnTensor<BurnBackend, 1> = BurnTensor::from_data(TensorData::new(w, [4]), &dev());
    let rms_b = (xb.clone().powf_scalar(2.0).mean_dim(1) + 1e-6f32).sqrt();
    let out_b = xb / rms_b * wb.unsqueeze::<2>();
    assert_close_rel("rmsnorm_fwd", out_c.tensor.as_slice(), &bvec(out_b), 1e-4);
    let _ = backend;
}

// ── LayerNorm backward ──────────────────────────────────────────────────────────

#[test]
fn layernorm_backward_matches_burn_autodiff() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();
    let x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0];
    let w = vec![1.2f32, 0.8, 1.0, 0.9];
    let b = vec![0.1f32, -0.1, 0.2, 0.0];
    let eps = 1e-5f32;

    // Coeus backward of sum(layernorm(x)).
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &x),
        true,
    );
    let mut ln = coeus_nn::LayerNorm::<f32, SequentialBackend>::new(4, eps as f64);
    ln.weight = Var::new(CoeusTensor::from_slice(vec![4], &w), true);
    ln.bias = Var::new(CoeusTensor::from_slice(vec![4], &b), true);
    coeus_autograd::sum(&ln.forward(&xv)).backward();

    // Burn autodiff over the identical normalization formula.
    let xb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(x, [2, 4]), &device).require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(w, [4]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(b, [4]), &device).require_grad();
    let mean = xb.clone().mean_dim(1);
    let xc = xb.clone() - mean;
    let std = (xc.clone().powf_scalar(2.0).mean_dim(1) + eps).sqrt();
    let out_b = xc / std * wb.clone().unsqueeze::<2>() + bk.clone().unsqueeze::<2>();
    let grads = out_b.sum().backward();
    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();
    let to_vec2 = |t: BurnTensor<NdArray<f32>, 2>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "layernorm_bwd dx",
        xv.grad().unwrap().as_slice(),
        &to_vec2(xb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "layernorm_bwd dw",
        ln.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "layernorm_bwd db",
        ln.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── RMSNorm backward ────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_backward_matches_burn_autodiff() {
    use burn::backend::autodiff::Autodiff;
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();
    let x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0];
    let w = vec![1.0f32, 0.8, 1.2, 0.9];
    let eps = 1e-6f32;

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &x),
        true,
    );
    let mut rms = coeus_nn::RMSNorm::<f32, SequentialBackend>::new(4, eps as f64);
    rms.weight = Var::new(CoeusTensor::from_slice(vec![4], &w), true);
    coeus_autograd::sum(&rms.forward(&xv)).backward();

    let xb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(x, [2, 4]), &device).require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(w, [4]), &device).require_grad();
    let rms_b = (xb.clone().powf_scalar(2.0).mean_dim(1) + eps).sqrt();
    let out_b = xb.clone() / rms_b * wb.clone().unsqueeze::<2>();
    let grads = out_b.sum().backward();
    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();
    let to_vec2 = |t: BurnTensor<NdArray<f32>, 2>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "rmsnorm_bwd dx",
        xv.grad().unwrap().as_slice(),
        &to_vec2(xb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "rmsnorm_bwd dw",
        rms.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── Clamp ─────────────────────────────────────────────────────────────────────

#[test]
fn clamp_matches_burn() {
    let data = vec![-3.0f32, -0.5, 0.5, 1.5, 2.5, 4.0];
    let lo = -1.0f32;
    let hi = 2.0f32;
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data),
        false,
    );
    let out_c = coeus_autograd::clamp(&xv, lo, hi);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    assert_close("clamp", out_c.tensor.as_slice(), &bvec(xb.clamp(lo, hi)));
}

// ── Flip ──────────────────────────────────────────────────────────────────────

#[test]
fn flip_matches_burn_manual() {
    let backend = SequentialBackend::new();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xc = CoeusTensor::from_slice(vec![2, 3], &data);
    let out_c = coeus_ops::flip(&xc, 1, &backend);
    // Manual flip along axis 1 (columns): [1,2,3,4,5,6] → [3,2,1,6,5,4]
    let expected = vec![3.0f32, 2.0, 1.0, 6.0, 5.0, 4.0];
    assert_close("flip_axis1", out_c.as_slice(), &expected);
}

// ── Shape ops ─────────────────────────────────────────────────────────────────

#[test]
fn stack_matches_burn() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let ac = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &a);
    let bc = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &b);
    let out_c: CoeusTensor<f32, SequentialBackend> = coeus_ops::stack(&[&ac, &bc], 0);
    let ab: BurnTensor<BurnBackend, 1> = BurnTensor::from_data(TensorData::new(a, [3]), &dev());
    let bb: BurnTensor<BurnBackend, 1> = BurnTensor::from_data(TensorData::new(b, [3]), &dev());
    assert_close(
        "stack",
        out_c.as_slice(),
        &bvec(BurnTensor::<BurnBackend, 1>::stack::<2>(vec![ab, bb], 0)),
    );
}

#[test]
fn cat_reshape_t_match_burn() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xc = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data);
    let xb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), [2, 3]), &dev());
    // t() (2D transpose): returns a non-contiguous view; materialise for comparison
    assert_close(
        "t",
        xc.clone().t().to_contiguous().as_slice(),
        &bvec(xb.clone().transpose()),
    );
    // Reshape [2,3] → [3,2]
    assert_close(
        "reshape",
        xc.reshape([3, 2]).as_slice(),
        &bvec(xb.reshape([3, 2])),
    );
    // Cat along dim 0
    let a2 = vec![1.0f32, 2.0, 3.0, 4.0];
    let b2 = vec![5.0f32, 6.0, 7.0, 8.0];
    let ac = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &a2);
    let bc = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &b2);
    let ab2: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(a2, [2, 2]), &dev());
    let bb2: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(b2, [2, 2]), &dev());
    let cat_out: CoeusTensor<f32, SequentialBackend> = coeus_ops::cat(&[&ac, &bc], 0);
    assert_close(
        "cat_dim0",
        cat_out.as_slice(),
        &bvec(BurnTensor::<BurnBackend, 2>::cat(vec![ab2, bb2], 0)),
    );
}

// ── Sort / where_cond ─────────────────────────────────────────────────────────

#[test]
fn sort_correctness() {
    let backend = SequentialBackend::new();
    let data = vec![3.0f32, 1.0, 4.0, 1.5, 9.0, 2.6];
    let x = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![6], &data);
    let (sorted, _idx) = coeus_ops::sort(&x, 0, false, &backend);
    let sorted_vals = sorted.as_slice();
    for i in 0..sorted_vals.len() - 1 {
        assert!(
            sorted_vals[i] <= sorted_vals[i + 1],
            "sort: not ascending at {i}"
        );
    }
}

#[test]
fn where_cond_correctness() {
    let backend = SequentialBackend::new();
    let cond: CoeusTensor<f32, SequentialBackend> =
        CoeusTensor::from_slice(vec![4], &[1.0f32, 0.0, 1.0, 0.0]);
    let on_t: CoeusTensor<f32, SequentialBackend> =
        CoeusTensor::from_slice(vec![4], &[10.0f32, 20.0, 30.0, 40.0]);
    let on_f: CoeusTensor<f32, SequentialBackend> =
        CoeusTensor::from_slice(vec![4], &[-1.0f32, -2.0, -3.0, -4.0]);
    let out = coeus_ops::where_cond(&cond, &on_t, &on_f, &backend);
    assert_close("where_cond", out.as_slice(), &[10.0, -2.0, 30.0, -4.0]);
}

// ── Conv1d (functional, manual reference) ────────────────────────────────────

#[test]
fn conv1d_forward_matches_manual_reference() {
    use coeus_ops::BackendOps;
    let backend = SequentialBackend::new();
    // batch=1, in_channels=2, length=6, out_channels=2, kernel=3
    let input: Vec<f32> = (0..12).map(|x| x as f32 * 0.1).collect();
    let weight: Vec<f32> = (0..12).map(|x| x as f32 * 0.05 - 0.3).collect();
    let bias_data: Vec<f32> = vec![0.1f32, -0.1];
    let (b, ic, l, oc, k) = (1usize, 2, 6, 2, 3);
    let out_len = l - k + 1;

    let in_t = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![b, ic, l], &input);
    let wt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![oc, ic, k], &weight);
    let bt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![oc], &bias_data);
    let mut cpu_out = CoeusTensor::<f32, SequentialBackend>::zeros(vec![b, oc, out_len]);
    // Pre-compute output layout to avoid simultaneous mut/shared borrow.
    let out_layout = cpu_out.layout().clone();
    backend.conv1d(
        in_t.storage(),
        in_t.layout(),
        wt.storage(),
        wt.layout(),
        Some(bt.storage()),
        1,
        0,
        1,
        cpu_out.storage_mut(),
        &out_layout,
    );

    // Manual reference (cross-correlation + bias)
    let out_ref: Vec<f32> = {
        let in_s = in_t.as_slice();
        let w_s = wt.as_slice();
        let b_s = bt.as_slice();
        let mut v = vec![0.0f32; b * oc * out_len];
        for bi in 0..b {
            for o in 0..oc {
                for t in 0..out_len {
                    let mut acc = b_s[o];
                    for c in 0..ic {
                        for ki in 0..k {
                            acc +=
                                w_s[o * ic * k + c * k + ki] * in_s[bi * ic * l + c * l + t + ki];
                        }
                    }
                    v[bi * oc * out_len + o * out_len + t] = acc;
                }
            }
        }
        v
    };
    assert_close_rel("conv1d_fwd", cpu_out.as_slice(), &out_ref, 1e-4);
}

// ── Conv2d (functional, manual reference) ────────────────────────────────────

#[test]
fn conv2d_forward_matches_manual_reference() {
    use coeus_ops::BackendOps;
    let backend = SequentialBackend::new();
    let (b, ic, h, w, oc, kh, kw) = (1usize, 2, 4, 4, 2, 2, 2);
    let (oh, ow) = (h - kh + 1, w - kw + 1);

    let input: Vec<f32> = (0..b * ic * h * w).map(|x| x as f32 * 0.1 - 0.8).collect();
    let weight: Vec<f32> = (0..oc * ic * kh * kw)
        .map(|x| x as f32 * 0.15 - 0.5)
        .collect();
    let bias_data: Vec<f32> = vec![0.05f32, -0.05];

    let in_t = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![b, ic, h, w], &input);
    let wt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![oc, ic, kh, kw], &weight);
    let bt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![oc], &bias_data);
    let mut cpu_out = CoeusTensor::<f32, SequentialBackend>::zeros(vec![b, oc, oh, ow]);
    let out_layout = cpu_out.layout().clone();
    backend.conv2d(
        in_t.storage(),
        in_t.layout(),
        wt.storage(),
        wt.layout(),
        Some(bt.storage()),
        1,
        0,
        1,
        cpu_out.storage_mut(),
        &out_layout,
    );

    // Manual reference
    let expected: Vec<f32> = {
        let in_s = in_t.as_slice();
        let w_s = wt.as_slice();
        let b_s = bt.as_slice();
        let mut v = vec![0.0f32; b * oc * oh * ow];
        for bi in 0..b {
            for o in 0..oc {
                for yi in 0..oh {
                    for xi in 0..ow {
                        let mut acc = b_s[o];
                        for c in 0..ic {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    acc += w_s[o * ic * kh * kw + c * kh * kw + ki * kw + kj]
                                        * in_s[bi * ic * h * w
                                            + c * h * w
                                            + (yi + ki) * w
                                            + (xi + kj)];
                                }
                            }
                        }
                        v[bi * oc * oh * ow + o * oh * ow + yi * ow + xi] = acc;
                    }
                }
            }
        }
        v
    };
    assert_close_rel("conv2d_fwd", cpu_out.as_slice(), &expected, 1e-4);
}

// ── MaxPool2d (manual reference) ──────────────────────────────────────────────

#[test]
fn max_pool2d_forward_matches_manual_reference() {
    use coeus_ops::BackendOps;
    let backend = SequentialBackend::new();
    let (b, c, h, w) = (2, 2, 4, 4);
    let (oh, ow, ks, st) = (2, 2, 2, 2);
    let data: Vec<f32> = (0..b * c * h * w).map(|x| x as f32 * 0.1).collect();
    let x = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![b, c, h, w], &data);
    let mut out = CoeusTensor::<f32, SequentialBackend>::zeros(vec![b, c, oh, ow]);
    let out_layout = out.layout().clone();
    backend.max_pool2d(
        x.storage(),
        x.layout(),
        ks,
        st,
        0,
        1,
        out.storage_mut(),
        &out_layout,
    );

    // Manual reference
    let x_s = x.as_slice();
    let expected: Vec<f32> = (0..b * c * oh * ow)
        .map(|flat| {
            let bi = flat / (c * oh * ow);
            let rem = flat % (c * oh * ow);
            let ci = rem / (oh * ow);
            let rem2 = rem % (oh * ow);
            let yi = rem2 / ow;
            let xi = rem2 % ow;
            let mut mx = f32::NEG_INFINITY;
            for ki in 0..ks {
                for kj in 0..ks {
                    let v = x_s[bi * c * h * w + ci * h * w + (yi * st + ki) * w + (xi * st + kj)];
                    if v > mx {
                        mx = v;
                    }
                }
            }
            mx
        })
        .collect();
    assert_close("max_pool2d_fwd", out.as_slice(), &expected);
}

// ── Autograd through where_cond ───────────────────────────────────────────────

#[test]
fn where_cond_backward_passes_grad_to_true_and_false() {
    let cond = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0f32, 0.0, 1.0, 0.0]),
        false,
    );
    let on_t = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[2.0f32, 3.0, 4.0, 5.0]),
        true,
    );
    let on_f = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[10.0f32, 11.0, 12.0, 13.0]),
        true,
    );
    let out = coeus_autograd::where_cond(&cond, &on_t, &on_f);
    coeus_autograd::sum(&out).backward();
    let gt = on_t.grad().unwrap();
    let gf = on_f.grad().unwrap();
    assert_close("where_cond_bwd true", gt.as_slice(), &[1.0, 0.0, 1.0, 0.0]);
    assert_close("where_cond_bwd false", gf.as_slice(), &[0.0, 1.0, 0.0, 1.0]);
}

// ── Flip backward ─────────────────────────────────────────────────────────────

#[test]
fn flip_backward_passes_grad() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data),
        true,
    );
    let out = coeus_autograd::flip(&xv, 1);
    coeus_autograd::sum(&out).backward();
    let grad = xv.grad().unwrap();
    assert_close("flip_bwd", grad.as_slice(), &[1.0; 6]);
}

// ── AvgPool2d (manual reference) ─────────────────────────────────────────────

#[test]
fn avg_pool2d_forward_matches_manual_reference() {
    use coeus_ops::BackendOps;
    let backend = SequentialBackend::new();
    let (b, c, h, w) = (2, 2, 4, 4);
    let (oh, ow, ks, st) = (2, 2, 2, 2);
    let data: Vec<f32> = (0..b * c * h * w).map(|x| x as f32 * 0.1).collect();
    let x = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![b, c, h, w], &data);
    let mut out = CoeusTensor::<f32, SequentialBackend>::zeros(vec![b, c, oh, ow]);
    let out_layout = out.layout().clone();
    backend.avg_pool2d(
        x.storage(),
        x.layout(),
        ks,
        st,
        0,
        1,
        out.storage_mut(),
        &out_layout,
    );

    let x_s = x.as_slice();
    let ks_sq = (ks * ks) as f32;
    let expected: Vec<f32> = (0..b * c * oh * ow)
        .map(|flat| {
            let bi = flat / (c * oh * ow);
            let rem = flat % (c * oh * ow);
            let ci = rem / (oh * ow);
            let rem2 = rem % (oh * ow);
            let yi = rem2 / ow;
            let xi = rem2 % ow;
            let mut acc = 0.0f32;
            for ki in 0..ks {
                for kj in 0..ks {
                    acc += x_s[bi * c * h * w + ci * h * w + (yi * st + ki) * w + (xi * st + kj)];
                }
            }
            acc / ks_sq
        })
        .collect();
    assert_close("avg_pool2d_fwd", out.as_slice(), &expected);
}

// ── GlobalAvgPool2d (reduces spatial to 1×1) ─────────────────────────────────

#[test]
fn global_avg_pool2d_reduces_spatial_to_one() {
    use coeus_nn::GlobalAvgPool2d;
    let backend = SequentialBackend::new();
    let _ = backend;
    let data: Vec<f32> = (0..16).map(|x| x as f32 + 1.0).collect();
    // [1, 1, 4, 4] — values 1..16
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        false,
    );
    let pool = GlobalAvgPool2d::<f32, SequentialBackend>::new();
    use coeus_nn::Module;
    let out = pool.forward(&xv);
    assert_eq!(out.tensor.shape(), &[1, 1, 1, 1], "global avg pool shape");
    let expected_mean = data.iter().sum::<f32>() / data.len() as f32;
    assert_close("global_avg_pool2d", out.tensor.as_slice(), &[expected_mean]);
}

// ── GlobalMaxPool2d (reduces spatial to 1×1) ─────────────────────────────────

#[test]
fn global_max_pool2d_reduces_spatial_to_one() {
    use coeus_nn::GlobalMaxPool2d;
    let data: Vec<f32> = (0..16).map(|x| x as f32 + 1.0).collect();
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        false,
    );
    let pool = GlobalMaxPool2d::<f32, SequentialBackend>::new();
    use coeus_nn::Module;
    let out = pool.forward(&xv);
    assert_eq!(out.tensor.shape(), &[1, 1, 1, 1], "global max pool shape");
    // max of 1..16 is 16.0
    assert_close("global_max_pool2d", out.tensor.as_slice(), &[16.0]);
}

// ── BatchNorm1d forward (manual reference) ───────────────────────────────────

#[test]
fn batchnorm1d_forward_matches_manual_reference() {
    use coeus_nn::BatchNorm1d;
    // BatchNorm1d expects [N, C, L]: batch=1, channels=2, length=4
    // scale=1, bias=0 initially; training mode normalises per-channel across N*L
    let data: Vec<f32> = (0..8).map(|x| x as f32 + 1.0).collect(); // [1,2,4]
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 4], &data),
        false,
    );
    let bn = BatchNorm1d::<f32, SequentialBackend>::new(2, 1e-5, 0.1);
    use coeus_nn::Module;
    let out = bn.forward(&xv);
    assert_eq!(out.tensor.shape(), &[1, 2, 4], "batchnorm1d output shape");

    // Output should be near zero mean and unit variance per channel
    let out_s = out.tensor.as_slice();
    // channel 0: elements [1,2,3,4] → mean=2.5
    let c0: &[f32] = &out_s[..4];
    let c0_mean = c0.iter().sum::<f32>() / 4.0;
    assert!(
        c0_mean.abs() < 1e-4,
        "batchnorm1d channel0 mean {c0_mean} should be near 0"
    );
    // channel 1: elements [5,6,7,8] → mean=6.5
    let c1: &[f32] = &out_s[4..];
    let c1_mean = c1.iter().sum::<f32>() / 4.0;
    assert!(
        c1_mean.abs() < 1e-4,
        "batchnorm1d channel1 mean {c1_mean} should be near 0"
    );
}

// ── GlobalAvgPool2d non-square input ─────────────────────────────────────────

#[test]
fn global_avg_pool2d_handles_non_square_spatial() {
    use coeus_nn::GlobalAvgPool2d;
    use coeus_nn::Module;
    // [1, 1, 2, 4] — mean of each row then mean of the result equals overall mean
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 4], &data),
        false,
    );
    let pool = GlobalAvgPool2d::<f32, SequentialBackend>::new();
    let out = pool.forward(&xv);
    assert_eq!(
        out.tensor.shape(),
        &[1, 1, 1, 1],
        "non-square global avg shape"
    );
    let expected = data.iter().sum::<f32>() / data.len() as f32; // 4.5
    assert_close(
        "global_avg_pool2d_non_square",
        out.tensor.as_slice(),
        &[expected],
    );
}

// ── GlobalMaxPool2d backward ──────────────────────────────────────────────────

#[test]
fn global_max_pool2d_backward_passes_grad_to_max_position() {
    use coeus_nn::GlobalMaxPool2d;
    use coeus_nn::Module;
    // [1, 1, 2, 3]: max at position (0,2) in row-major → index 2 overall
    let data: Vec<f32> = vec![1.0, 3.0, 5.0, 2.0, 4.0, 0.5];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 3], &data),
        true,
    );
    let pool = GlobalMaxPool2d::<f32, SequentialBackend>::new();
    let out = pool.forward(&xv);
    assert_eq!(out.tensor.shape(), &[1, 1, 1, 1]);
    assert_close("global_max_pool2d_bwd_fwd", out.tensor.as_slice(), &[5.0]);
    out.backward();
    let grad = xv.grad().unwrap();
    // Only the maximum element receives gradient 1.0; all others receive 0.0
    let gv: Vec<f32> = grad.as_slice().to_vec();
    assert_eq!(gv.len(), 6);
    let total: f32 = gv.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "total grad should be 1: {total}"
    );
    assert!((gv[2] - 1.0).abs() < 1e-6, "max position grad: {}", gv[2]);
}
