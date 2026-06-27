use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{activation as burn_act, Tensor as BurnTensor, TensorData};
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{cross_entropy_loss, softmax, Module};
use coeus_ops::{leaky_relu, log_softmax_axis, mish, sigmoid, silu, softplus, tanh};
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

    // Global var / std: Burn 0.16's `var(dim)` reduces one axis; flatten
    // to a 1-D tensor first, then `var(0)` followed by `into_scalar` to
    // obtain the global Bessel-corrected oracle.
    let v_global = coeus_ops::var(&xc, true, &backend);
    let v_global_burn = xb.clone().flatten::<1>(0, 1).var(0).into_scalar();
    assert_close("var_global", &[v_global], &[v_global_burn]);
    let s_global = coeus_ops::std_dev(&xc, true, &backend);
    let s_global_burn = xb.clone().flatten::<1>(0, 1).var(0).into_scalar().sqrt();
    assert_close("std_global", &[s_global], &[s_global_burn]);

    // Per-axis var matches Burn's Bessel-corrected `var(dim)` API; per-axis
    // std is the square root of that same variance oracle.
    let v_axis1 = coeus_ops::var_axis(&xc, 1, true, &backend);
    let v_burn_axis1 = xb.clone().var(1).into_data().to_vec().unwrap();
    assert_close("var_axis1", v_axis1.as_slice(), &v_burn_axis1);
    let s_axis1 = coeus_ops::std_dev_axis(&xc, 1, true, &backend);
    let s_burn_axis1 = bvec(xb.clone().var(1).sqrt());
    assert_close("std_axis1", s_axis1.as_slice(), &s_burn_axis1);

    // Population (unbiased=false) global variance matches Burn's
    // `var_bias(dim)` on the flattened input — Bessel-correction skipped,
    // so it is `N/(N-1)` smaller than the unbiased variant.
    let v_biased = coeus_ops::var(&xc, false, &backend);
    let v_biased_burn = xb.clone().flatten::<1>(0, 1).var_bias(0).into_scalar();
    assert_close("var_biased_global", &[v_biased], &[v_biased_burn]);

    // L2 norm matches Burn's `powf_scalar(2).sum().sqrt()` over flattened
    // input (matches torch.linalg.vector_norm default ord=2).
    let n2 = coeus_ops::norm(&xc, &backend);
    let n_burn = bvec(xb.clone().powf_scalar(2.0).sum())[0].sqrt();
    assert_close("norm_l2", &[n2], &[n_burn]);

    // L_p norm parity: ord ∈ {1, 2, 3}. Coeus folds via
    // `coeus_ops::norm_p` (host-side `T::powf` accumulation with final
    // `^(1/p)`); Burn uses `powf_scalar(p).sum().powf_scalar(1/p)` over
    // the flattened input. Reduction order differs (Coeus is per-bucket
    // host fold vs Burn's fused pipeline), so the assertion uses the
    // forward-equivalent lambda with a reduction-order-sensitive bound
    // derived in `docs/backlog.md` MS-66.
    for &ord in &[1.0f64, 2.0, 3.0] {
        let coeus_p = coeus_ops::norm_p(&xc, ord as f32, &backend);
        let sum_p = bvec(xb.clone().powf_scalar(ord as f32).sum())[0];
        let burn_p = sum_p.powf(1.0 / ord as f32);
        let label = format!("norm_l{ord}");
        assert_close(&label, &[coeus_p], &[burn_p]);
    }

    // Per-axis Lp norm uses the same ord-p contract with the selected
    // dimension reduced to size 1, matching Burn's sum_dim keepdim shape.
    let coeus_axis1_l2 = coeus_ops::norm_p_axis(&xc, 2.0, 1, &backend);
    let burn_axis1_l2 = bvec(xb.clone().powf_scalar(2.0).sum_dim(1).powf_scalar(0.5));
    assert_close("norm_l2_axis1", coeus_axis1_l2.as_slice(), &burn_axis1_l2);

    let coeus_axis0_l1 = coeus_ops::norm_p_axis(&xc, 1.0, 0, &backend);
    let burn_axis0_l1 = bvec(xb.clone().abs().sum_dim(0));
    assert_close("norm_l1_axis0", coeus_axis0_l1.as_slice(), &burn_axis0_l1);
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

// ── meshgrid / tile forward ────────────────────────────────────────────────

#[test]
fn meshgrid_ij_matches_manual_reference() {
    let backend = SequentialBackend::new();
    let x = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &[0.0f32, 1.0, 2.0]);
    let y = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2], &[10.0f32, 20.0]);
    let grids = coeus_ops::meshgrid(&[&x, &y], "ij", &backend);
    assert_eq!(grids.len(), 2);
    // grid_x varies along axis 0
    assert_close(
        "meshgrid_grid_x",
        grids[0].as_slice(),
        &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    );
    // grid_y varies along axis 1
    assert_close(
        "meshgrid_grid_y",
        grids[1].as_slice(),
        &[10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
    );
}

#[test]
fn tile_forward_and_backward() {
    let backend = SequentialBackend::new();
    let x = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]);
    let out = coeus_ops::tile(&x, &[2], &backend);
    assert_eq!(out.shape(), &[6]);
    assert_close("tile_1d", out.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    // 2-D tiling
    let m = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
    let m2 = coeus_ops::tile(&m, &[1, 3], &backend);
    assert_eq!(m2.shape(), &[2, 6]);
    assert_close(
        "tile_2d_row0",
        &m2.as_slice()[..6],
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );

    // Tracked tile backward: grad sums over copies.
    let xg = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::tile(&xg, &[3])).backward();
    // Each element copied 3× → gradient = 3 for each.
    assert_close("tile_bwd", xg.grad().unwrap().as_slice(), &[3.0, 3.0, 3.0]);
    let _ = backend;
}

#[test]
fn cumprod_forward_and_backward() {
    // Forward: [1,2,3,4] → [1,2,6,24]
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]),
        false,
    );
    let backend = SequentialBackend::new();
    let out = coeus_ops::cumprod(&x.tensor, 0, &backend);
    assert_close("cumprod_fwd", out.as_slice(), &[1.0, 2.0, 6.0, 24.0]);

    // Backward: d(sum(cumprod))/dx using the suffix-sum formula.
    let xg = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::cumprod(&xg, 0)).backward();
    // out=[1,2,6], sum=9
    // grad[0] = (1+2+6)/1 = 9
    // grad[1] = (2+6)/2   = 4
    // grad[2] = 6/3       = 2
    let grad = xg.grad().unwrap();
    assert_close_rel("cumprod_bwd", grad.as_slice(), &[9.0, 4.0, 2.0], 1e-5);
    let _ = backend;
}

// ── diag / diagonal forward + backward ──────────────────────────────────────

#[test]
fn diag_diagonal_forward_and_backward() {
    let data = vec![1.0f32, 2.0, 3.0];
    let v = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &data),
        false,
    );
    let backend = SequentialBackend::new();

    // diag forward
    let m = coeus_ops::diag(&v.tensor, 0, &backend);
    assert_eq!(m.shape(), &[3, 3]);
    assert_close(
        "diag_fwd",
        m.as_slice(),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );

    // diagonal forward (extract main diagonal of the matrix)
    let mat = CoeusTensor::<f32, SequentialBackend>::from_slice(
        vec![3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let d = coeus_ops::diagonal(&mat, 0, &backend);
    assert_close("diagonal_fwd", d.as_slice(), &[1.0, 5.0, 9.0]);

    // diag backward: grad flows via diagonal
    let vg = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::diag(&vg, 0)).backward();
    assert_close("diag_bwd", vg.grad().unwrap().as_slice(), &[1.0, 1.0, 1.0]);

    let _ = backend;
}

#[test]
fn tril_triu_forward_and_backward() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // tril forward: elements above main diagonal → 0
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3, 3], &data),
        false,
    );
    let lo = coeus_autograd::tril(&xv, 0);
    assert_close(
        "tril_fwd",
        lo.tensor.as_slice(),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
    );

    // tril backward: gradient is zero at positions that were masked out
    let xg = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3, 3], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::tril(&xg, 0)).backward();
    assert_close(
        "tril_bwd",
        xg.grad().unwrap().as_slice(),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    );

    // triu forward: elements below main diagonal → 0
    let xv2 = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3, 3], &data),
        false,
    );
    let hi = coeus_autograd::triu(&xv2, 0);
    assert_close(
        "triu_fwd",
        hi.tensor.as_slice(),
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
    );
}

// ── roll forward + backward ───────────────────────────────────────────────────

#[test]
fn roll_forward_and_backward() {
    let data = vec![0.0f32, 1.0, 2.0, 3.0];
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &data),
        false,
    );
    // shift +1: last element wraps to front
    let rolled = coeus_autograd::roll(&xv, &[1], &[0]);
    assert_close("roll_fwd", rolled.tensor.as_slice(), &[3.0, 0.0, 1.0, 2.0]);

    // backward: all-ones gradient rolled by -1 is still all-ones
    let xg = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &data),
        true,
    );
    coeus_autograd::sum(&coeus_autograd::roll(&xg, &[1], &[0])).backward();
    assert_close("roll_bwd", xg.grad().unwrap().as_slice(), &[1.0; 4]);
}

// ── LayerNorm forward_nd (3-D input, matches 2-D reshape path) ───────────────

#[test]
fn layernorm_forward_nd_3d_matches_reshape_reference() {
    // [batch=2, seq=3, d=4] input — forward_nd should give identical output to
    // manual reshape→LayerNorm→reshape.
    let data: Vec<f32> = (0..24).map(|v| v as f32 * 0.1 - 1.2).collect();
    let (batch, seq, d) = (2, 3, 4);

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d], &data),
        true,
    );

    let w = vec![1.2f32, 0.8, 1.0, 0.9];
    let b = vec![0.1f32, -0.1, 0.05, -0.05];
    let eps = 1e-5;

    let mut ln =
        coeus_nn::normalization::layernorm::LayerNorm::<f32, SequentialBackend>::new(d, eps);
    ln.weight = Var::new(CoeusTensor::from_slice(vec![d], &w), true);
    ln.bias = Var::new(CoeusTensor::from_slice(vec![d], &b), true);

    let out_nd = ln.forward_nd(&xv);

    // Manual reference: reshape → LayerNorm 2-D → reshape back.
    let x_flat = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch * seq, d], &data),
        false,
    );
    let mut ln2 =
        coeus_nn::normalization::layernorm::LayerNorm::<f32, SequentialBackend>::new(d, eps);
    ln2.weight = Var::new(CoeusTensor::from_slice(vec![d], &w), false);
    ln2.bias = Var::new(CoeusTensor::from_slice(vec![d], &b), false);
    use coeus_nn::Module;
    let out_2d = ln2.forward(&x_flat);

    // Shapes must match ([batch, seq, d] vs [batch*seq, d] — flat values equal).
    assert_close_rel(
        "layernorm_nd_3d",
        out_nd.tensor.to_contiguous().as_slice(),
        out_2d.tensor.as_slice(),
        1e-4,
    );

    // Backward gradient must also flow through the 3-D path.
    coeus_autograd::sum(&out_nd).backward();
    assert!(
        xv.grad().is_some(),
        "layernorm forward_nd backward did not propagate gradient"
    );
}

// ── ConvTranspose1d (stride-2, manual reference) ──────────────────────────────
#[test]
fn conv_transpose1d_stride2_matches_manual_reference() {
    use coeus_ops::BackendOps;
    let backend = SequentialBackend::new();

    // batch=1, C_in=2, L=3 → C_out=2, K=2, stride=2 → L_out = (3-1)*2 + 2 = 6
    let input: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, -2.0, 0.5];
    let weight: Vec<f32> = vec![1.0, 0.5, -0.5, 0.25, 0.0, -1.0, 1.0, 0.5]; // [C_in=2, C_out=2, K=2]
    let (n, c_in, l, c_out, k, stride, padding, op, dilation) = (1usize, 2, 3, 2, 2, 2, 0, 0, 1);
    let l_out =
        coeus_ops::conv_transpose::conv_transpose1d_output_len(l, k, stride, padding, op, dilation);
    assert_eq!(l_out, 6);

    let in_t = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c_in, l], &input);
    let wt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![c_in, c_out, k], &weight);
    let mut out = CoeusTensor::<f32, SequentialBackend>::zeros(vec![n, c_out, l_out]);
    let out_layout = out.layout().clone();
    backend.conv_transpose1d(
        in_t.storage(),
        in_t.layout(),
        wt.storage(),
        wt.layout(),
        None,
        stride,
        padding,
        op,
        dilation,
        out.storage_mut(),
        &out_layout,
    );

    // Manual reference: scatter input→output via stride=2 mapping.
    // For each (n_i, c_in_i, l_i): for each (c_out_j, k_i):
    //   output[n_i, c_out_j, l_i * stride + k_i] += input[n_i, c_in_i, l_i] * weight[c_in_i, c_out_j, k_i]
    let mut expected = vec![0.0f32; n * c_out * l_out];
    let in_s = in_t.as_slice();
    let w_s = wt.as_slice();
    for ni in 0..n {
        for ci in 0..c_in {
            for li in 0..l {
                for co in 0..c_out {
                    for ki in 0..k {
                        let pos = li * stride + ki;
                        if pos < l_out {
                            expected[ni * c_out * l_out + co * l_out + pos] += in_s
                                [ni * c_in * l + ci * l + li]
                                * w_s[ci * c_out * k + co * k + ki];
                        }
                    }
                }
            }
        }
    }
    assert_close_rel("conv_transpose1d_stride2", out.as_slice(), &expected, 1e-4);
    let _ = backend;
}

// ── ConvTranspose2d (stride-1, manual reference) ──────────────────────────────

#[test]
fn conv_transpose2d_unit_stride_matches_manual_reference() {
    use coeus_ops::BackendOps;
    let backend = SequentialBackend::new();

    // batch=1, C_in=1, H=3, W=3, C_out=1, KH=2, KW=2, stride=1 → H_out=4, W_out=4
    let input: Vec<f32> = (0..9).map(|v| v as f32).collect();
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // identity-ish 2×2 kernel
    let (n, c_in, h, w, c_out, kh, kw, stride, padding, op, dilation) =
        (1usize, 1, 3, 3, 1, 2, 2, 1, 0, 0, 1);
    let (h_out, w_out) = coeus_ops::conv_transpose::conv_transpose2d_output_dims(
        h, w, kh, kw, stride, padding, op, dilation,
    );
    assert_eq!((h_out, w_out), (4, 4));

    let in_t = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c_in, h, w], &input);
    let wt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![c_in, c_out, kh, kw], &weight);
    let mut out = CoeusTensor::<f32, SequentialBackend>::zeros(vec![n, c_out, h_out, w_out]);
    let out_layout = out.layout().clone();
    backend.conv_transpose2d(
        in_t.storage(),
        in_t.layout(),
        wt.storage(),
        wt.layout(),
        None,
        stride,
        padding,
        op,
        dilation,
        out.storage_mut(),
        &out_layout,
    );

    // Manual scatter reference.
    let mut expected = vec![0.0f32; n * c_out * h_out * w_out];
    let in_s = in_t.as_slice();
    let w_s = wt.as_slice();
    for ni in 0..n {
        for ci in 0..c_in {
            for yi in 0..h {
                for xi in 0..w {
                    for co in 0..c_out {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let py = yi * stride + ki;
                                let px = xi * stride + kj;
                                if py < h_out && px < w_out {
                                    expected[ni * c_out * h_out * w_out
                                        + co * h_out * w_out
                                        + py * w_out
                                        + px] += in_s[ni * c_in * h * w + ci * h * w + yi * w + xi]
                                        * w_s[ci * c_out * kh * kw + co * kh * kw + ki * kw + kj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_close_rel(
        "conv_transpose2d_unit_stride",
        out.as_slice(),
        &expected,
        1e-4,
    );
    let _ = backend;
}

// ── amax / amin / prod (manual references) ────────────────────────────────────

#[test]
fn amax_amin_prod_match_manual_reference() {
    let backend = SequentialBackend::new();
    let data = vec![3.0f32, -1.0, 5.0, 2.0, -4.0, 0.5];
    let x = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data);

    let amax_val = coeus_ops::amax(&x, &backend);
    assert!(
        (amax_val - 5.0f32).abs() < 1e-4,
        "amax: got {amax_val}, expected 5.0"
    );

    let amin_val = coeus_ops::amin(&x, &backend);
    assert!(
        (amin_val - (-4.0f32)).abs() < 1e-4,
        "amin: got {amin_val}, expected -4.0"
    );

    let prod_val = coeus_ops::prod(&x, &backend);
    let expected_prod: f32 = data.iter().product();
    assert!(
        (prod_val - expected_prod).abs() / expected_prod.abs().max(1e-6) < 1e-4,
        "prod: got {prod_val}, expected {expected_prod}"
    );
    let _ = backend;
}

// ── no_grad context (Rust-level) ──────────────────────────────────────────────

#[test]
fn no_grad_context_does_not_track() {
    // Outside no_grad: requires_grad=true should produce tracked var.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]),
        true,
    );
    let out_tracked = coeus_autograd::relu(&xv);
    // Creator is set when requires_grad is active.
    assert!(
        out_tracked.creator.is_some(),
        "expected creator outside no_grad"
    );

    // Inside no_grad context: creator should be None even though the input has requires_grad.
    coeus_autograd::grad_mode::push_no_grad();
    let xv2 = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3], &[1.0f32, 2.0, 3.0]),
        true,
    );
    let out_nograd = coeus_autograd::relu(&xv2);
    coeus_autograd::grad_mode::pop_no_grad();
    assert!(
        out_nograd.creator.is_none(),
        "expected no creator inside no_grad"
    );
}

// ── FeedForward forward shape contract ───────────────────────────────────────
//
// We can't compare against Burn exactly (weight init differs), but we can
// verify shape/rank and that forward produces finite, non-trivially-zero output.

#[test]
fn feed_forward_forward_shape_contract() {
    use coeus_nn::{FeedForward, Module};
    let (batch, seq, d_model, d_ff) = (2, 4, 8, 16);
    let ffn = FeedForward::<f32, SequentialBackend>::new(d_model, d_ff, 0.0);

    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::zeros(vec![batch, seq, d_model]),
        false,
    );
    let out = ffn.forward(&x);
    assert_eq!(
        out.tensor.shape(),
        &[batch, seq, d_model],
        "ffn output shape"
    );
    // With bias=true and non-zero linear weights, output should not be all-zero
    // for a non-zero input. Use a non-trivial input.
    let x2 = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(
            vec![1, 1, d_model],
            &vec![1.0f32; d_model],
        ),
        false,
    );
    let out2 = ffn.forward(&x2);
    // At least some outputs should be non-zero (weights are Xavier init, not all zero).
    let n_nonzero = out2
        .tensor
        .as_slice()
        .iter()
        .filter(|&&v| v.abs() > 1e-10)
        .count();
    assert!(
        n_nonzero > 0,
        "ffn produced all-zero output for non-zero input"
    );
}

// ── Multi-head attention forward: identity weights, analytical SDPA reference ──
//
// With W_q = W_k = W_v = W_o = I (identity, no bias) and H=1 head:
//   Q = K = V = X  (projections are identity)
//   A = softmax(X @ X^T / sqrt(d_model))
//   context = A @ X
//   output = context @ I^T = context
// We compute this reference in Burn autodiff tensors and compare elementwise.

#[test]
fn multi_head_attention_identity_weights_matches_analytical_sdpa() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::{Module, MultiHeadAttention, NullMask};
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    // H = 1 head so d_k = d_model; keeps the reference simple.
    let (batch, seq, d_model) = (1usize, 3, 4);
    let data: Vec<f32> = (0..batch * seq * d_model)
        .map(|x| (x as f32 + 1.0) * 0.1)
        .collect(); // [0.1, 0.2, ..., 1.2]

    // Coeus MHA with H=1, no bias, identity projection weights.
    let mut mha = MultiHeadAttention::<f32, SequentialBackend, 1, NullMask>::new(d_model, false);
    // Set W_q = W_k = W_v = W_o = identity [d_model, d_model].
    let eye: Vec<f32> = (0..d_model * d_model)
        .map(|i| if i % (d_model + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let eye_t = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_model, d_model], &eye);
    mha.w_q = Var::new(eye_t.clone(), false);
    mha.w_k = Var::new(eye_t.clone(), false);
    mha.w_v = Var::new(eye_t.clone(), false);
    mha.w_o = Var::new(eye_t, false);

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &data),
        true,
    );
    let out_c = mha.forward(&xv);

    // Burn autodiff reference: identity SDPA over the same input.
    // Q = K = V = X [batch, seq, d_model]; A = softmax(X @ X^T / sqrt(d)), ctx = A @ X.
    let xb: BurnTensor<AB, 3> = BurnTensor::from_data(
        TensorData::new(data.clone(), [batch, seq, d_model]),
        &device,
    )
    .require_grad();
    let scale = (d_model as f32).sqrt();
    // [batch, seq, seq] = [batch, seq, d] @ [batch, d, seq]
    let attn_logits = xb.clone().matmul(xb.clone().swap_dims(1, 2)) / scale;
    // softmax over last dim (seq axis = axis 2)
    let attn_w = burn_act::softmax(attn_logits, 2);
    // [batch, seq, d] = [batch, seq, seq] @ [batch, seq, d]
    let ctx: BurnTensor<AB, 3> = attn_w.matmul(xb.clone());
    let grads = ctx.clone().sum().backward();

    let out_b: Vec<f32> = ctx.detach().into_data().to_vec::<f32>().unwrap();
    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();

    assert_close_rel("mha_identity_fwd", out_c.tensor.as_slice(), &out_b, 1e-4);

    // Backward: backward gradients through Coeus MHA.
    coeus_autograd::sum(&out_c).backward();
    assert_close_rel(
        "mha_identity_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &dx_b,
        1e-4,
    );
}

// ── Multi-head attention backward matches Burn MHA module ────────────────────
//
// Uses Burn's full MultiHeadAttention module (not a manual formula) with
// non-trivial projection weights to verify forward + backward parity.
//
// Weight convention: Coeus stores projection weights as `[out, in]` and computes
// `x @ W^T`; Burn Linear stores `[in, out]` and computes `x @ W`. Burn weights
// are transposed on setup, and Burn weight gradients are transposed back before
// comparison.
//
// Configuration: d_model=2, H=1 (d_k=2), batch=1, seq=3, no bias.
// Tolerance: 1e-4 (f32 summation over small matrices).

#[test]
fn multi_head_attention_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::nn::attention::{MhaInput, MultiHeadAttentionConfig};
    use coeus_nn::{Module, MultiHeadAttention, NullMask};

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (batch, seq, d_model) = (1usize, 3, 2);

    let data: Vec<f32> = vec![0.1, 0.4, 0.7, -0.2, -0.3, 0.9];
    // Coeus: stores W [d_model, d_model], computes x @ W^T.
    // Burn Linear: stores W [d_in, d_out], computes x @ W  (no transpose in forward!).
    // Convention match: burn_W = coeus_W^T.
    let wq: Vec<f32> = vec![0.8, 0.2, -0.1, 0.7];
    let wk: Vec<f32> = vec![0.6, -0.3, 0.4, 0.9];
    let wv: Vec<f32> = vec![0.5, 0.1, 0.3, -0.4];
    let wo: Vec<f32> = vec![0.7, 0.3, -0.2, 0.6];
    // Transpose of each 2×2 row-major matrix: [a,b,c,d] → [a,c,b,d].
    let t2x2 = |w: &[f32]| vec![w[0], w[2], w[1], w[3]];
    let wq_t = t2x2(&wq);
    let wk_t = t2x2(&wk);
    let wv_t = t2x2(&wv);
    let wo_t = t2x2(&wo);

    // ── Coeus ──
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &data),
        true,
    );
    let mut mha = MultiHeadAttention::<f32, SequentialBackend, 1, NullMask>::new(d_model, false);
    mha.w_q = Var::new(CoeusTensor::from_slice(vec![d_model, d_model], &wq), true);
    mha.w_k = Var::new(CoeusTensor::from_slice(vec![d_model, d_model], &wk), true);
    mha.w_v = Var::new(CoeusTensor::from_slice(vec![d_model, d_model], &wv), true);
    mha.w_o = Var::new(CoeusTensor::from_slice(vec![d_model, d_model], &wo), true);
    let out_c = mha.forward(&xv);
    coeus_autograd::sum(&out_c).backward();

    // ── Burn MHA module (same weights, no bias) ──
    let xb: BurnTensor<AB, 3> = BurnTensor::from_data(
        TensorData::new(data.clone(), [batch, seq, d_model]),
        &device,
    )
    .require_grad();
    // Config::new(d_model, n_heads) — d_model is the first field in the struct.
    // dropout=0.0: Burn defaults to 0.1; we need deterministic attention scores.
    let mut mha_b = MultiHeadAttentionConfig::new(d_model, 1)
        .with_dropout(0.0)
        .with_quiet_softmax(false)
        .init::<AB>(&device);
    // Burn Linear weight shape is [d_in, d_out] and computes x @ W (no transpose).
    // Set burn_W = coeus_W^T so both produce x @ coeus_W^T.
    mha_b.query.weight =
        burn::module::Param::from_data(TensorData::new(wq_t.clone(), [d_model, d_model]), &device);
    mha_b.key.weight =
        burn::module::Param::from_data(TensorData::new(wk_t.clone(), [d_model, d_model]), &device);
    mha_b.value.weight =
        burn::module::Param::from_data(TensorData::new(wv_t.clone(), [d_model, d_model]), &device);
    mha_b.output.weight =
        burn::module::Param::from_data(TensorData::new(wo_t.clone(), [d_model, d_model]), &device);
    // Disable bias to match Coeus (bias=false).
    mha_b.query.bias = None;
    mha_b.key.bias = None;
    mha_b.value.bias = None;
    mha_b.output.bias = None;

    let input = MhaInput::self_attn(xb.clone());
    let out_b = mha_b.forward(input);
    let grads = out_b.context.clone().sum().backward();

    // Forward context: .detach() returns AB type; into_data() works on both backends.
    assert_close_rel(
        "mha_bwd_fwd",
        out_c.tensor.as_slice(),
        &out_b.context.detach().into_data().to_vec::<f32>().unwrap(),
        1e-4,
    );
    // dx: .grad() returns Option<Tensor<NdArray<f32>, D>> (inner backend).
    assert_close_rel(
        "mha_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &xb.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        1e-4,
    );
    // dW_q, dW_k, dW_v, dW_o: transpose Burn [in, out] gradients back to
    // Coeus [out, in] storage before comparing.
    let burn_dwq = t2x2(
        &mha_b
            .query
            .weight
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    let burn_dwk = t2x2(
        &mha_b
            .key
            .weight
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    let burn_dwv = t2x2(
        &mha_b
            .value
            .weight
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    let burn_dwo = t2x2(
        &mha_b
            .output
            .weight
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    );
    assert_close_rel(
        "mha_bwd_dwq",
        mha.w_q.grad().unwrap().as_slice(),
        &burn_dwq,
        1e-4,
    );
    assert_close_rel(
        "mha_bwd_dwk",
        mha.w_k.grad().unwrap().as_slice(),
        &burn_dwk,
        1e-4,
    );
    assert_close_rel(
        "mha_bwd_dwv",
        mha.w_v.grad().unwrap().as_slice(),
        &burn_dwv,
        1e-4,
    );
    assert_close_rel(
        "mha_bwd_dwo",
        mha.w_o.grad().unwrap().as_slice(),
        &burn_dwo,
        1e-4,
    );
}

// ── Multi-head attention forward shape contract ───────────────────────────────

#[test]
fn multi_head_attention_forward_shape_contract() {
    use coeus_nn::{Module, MultiHeadAttention, NullMask};
    let (batch, seq, d_model) = (2, 6, 16);
    // H = 4 heads (const generic; d_model = 16 must be divisible by H = 4).
    let mha = MultiHeadAttention::<f32, SequentialBackend, 4, NullMask>::new(d_model, true);
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(
            vec![batch, seq, d_model],
            &vec![0.1f32; batch * seq * d_model],
        ),
        false,
    );
    let out = mha.forward(&x);
    assert_eq!(
        out.tensor.shape(),
        &[batch, seq, d_model],
        "mha output shape"
    );
    // With non-trivial input, output should not be all-zero.
    let n_nonzero = out
        .tensor
        .as_slice()
        .iter()
        .filter(|&&v| v.abs() > 1e-10)
        .count();
    assert!(
        n_nonzero > 0,
        "mha produced all-zero output for non-zero input"
    );
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

// ── TransformerEncoderLayer: identity-weight value parity ─────────────────────
//
// Set all MHA weights and FFN linear weights to identity / zeros, LayerNorm
// weights to ones / zeros, then verify the forward output against a manual
// reference computed in Burn autodiff tensors.
//
// With identity MHA (W_q=W_k=W_v=W_o=I, no bias) and identity FFN (W1=I, b1=0,
// W2=I, b2=0) and LayerNorm(γ=1, β=0):
//   attn_out = LN(X + SDPA(X, X, X))
//   ffn_out  = LN(attn_out + ReLU(attn_out) @ I^T + 0)
// This is analytically reproducible.

#[test]
fn transformer_encoder_layer_identity_weights_matches_analytical() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::{Module, NullMask, TransformerEncoderLayer};

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    // H=1, d_model=4, d_ff=4, no dropout
    let (batch, seq, d_model, d_ff) = (1usize, 3, 4, 4);
    let data: Vec<f32> = (0..batch * seq * d_model)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    // Build Coeus TransformerEncoderLayer and force identity/zero weights.
    let mut layer =
        TransformerEncoderLayer::<f32, SequentialBackend, 1, NullMask>::new(d_model, d_ff, 0.0);

    let eye4: Vec<f32> = (0..d_model * d_model)
        .map(|i| if i % (d_model + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let zeros4 = vec![0.0f32; d_model];
    let ones4 = vec![1.0f32; d_model];
    let eye_t = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_model, d_model], &eye4);
    let zt = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_model], &zeros4);
    let ot = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_model], &ones4);

    // MHA weights (field: self_attn)
    layer.self_attn.w_q = Var::new(eye_t.clone(), false);
    layer.self_attn.w_k = Var::new(eye_t.clone(), false);
    layer.self_attn.w_v = Var::new(eye_t.clone(), false);
    layer.self_attn.w_o = Var::new(eye_t.clone(), false);
    layer.self_attn.b_q = None;
    layer.self_attn.b_k = None;
    layer.self_attn.b_v = None;
    layer.self_attn.b_o = None;
    // LayerNorm1 (post-attention): γ=1, β=0
    layer.norm1.weight = Var::new(ot.clone(), false);
    layer.norm1.bias = Var::new(zt.clone(), false);
    // FFN linear1: [d_ff, d_model] → identity; linear2: [d_model, d_ff] → identity
    let eye_ff = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_ff, d_model], &eye4);
    let zt_ff = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_ff], &zeros4);
    let eye_ff2 = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![d_model, d_ff], &eye4);
    layer.ffn.linear1.weight = Var::new(eye_ff, false);
    layer.ffn.linear1.bias = Some(Var::new(zt_ff, false));
    layer.ffn.linear2.weight = Var::new(eye_ff2, false);
    layer.ffn.linear2.bias = Some(Var::new(zt.clone(), false));
    // LayerNorm2 (post-FFN): γ=1, β=0
    layer.norm2.weight = Var::new(ot.clone(), false);
    layer.norm2.bias = Var::new(zt.clone(), false);

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &data),
        true,
    );
    let out_c = layer.forward(&xv);

    // Burn autodiff reference: Pre-LN encoder — LN→SDPA→residual→LN→FFN(GELU)→residual.
    // Matches forward_with_mask in encoder_layer.rs (dropout p=0 so pass-through).
    let xb: BurnTensor<AB, 3> = BurnTensor::from_data(
        TensorData::new(data.clone(), [batch, seq, d_model]),
        &device,
    )
    .require_grad();
    let scale = (d_model as f32).sqrt();

    // Helper: LayerNorm over last dim (population var, γ=1, β=0)
    let ln3 = |x: BurnTensor<AB, 3>| -> BurnTensor<AB, 3> {
        let mean = x.clone().sum_dim(2) / (d_model as f32);
        let diff = x - mean;
        let var = diff.clone().powf_scalar(2.0).sum_dim(2) / (d_model as f32);
        diff / (var.add_scalar(1e-5_f32)).sqrt()
    };

    // Sub-layer 1: LN(X) → SDPA → residual
    let normed1 = ln3(xb.clone());
    let logits = normed1.clone().matmul(normed1.clone().swap_dims(1, 2)) / scale;
    let attn_w = burn_act::softmax(logits, 2);
    let ctx = attn_w.matmul(normed1);
    let x = xb.clone() + ctx; // residual (dropout p=0)

    // Sub-layer 2: LN(x) → identity FFN(GELU) → residual
    let normed2 = ln3(x.clone());
    // GELU: 0.5 * t * (1 + tanh(sqrt(2/π) * (t + 0.044715 * t^3)))
    let sqrt_2_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let gelu = |t: BurnTensor<AB, 3>| -> BurnTensor<AB, 3> {
        let c = t.clone().powf_scalar(3.0).mul_scalar(0.044715);
        let inner = (t.clone() + c).mul_scalar(sqrt_2_pi);
        let tanh_val = inner.tanh();
        t.clone() * (tanh_val.add_scalar(1.0)) * 0.5
    };
    // FFN with identity W1=W2=I, b1=b2=0: output = gelu(normed2 @ I^T + 0) @ I^T + 0 = gelu(normed2)
    let ffn_out = gelu(normed2);
    let output = x.clone() + ffn_out; // residual (dropout p=0)
    let grads = output.clone().sum().backward();

    let out_b: Vec<f32> = output.detach().into_data().to_vec::<f32>().unwrap();
    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();

    assert_close_rel(
        "transformer_enc_identity_fwd",
        out_c.tensor.as_slice(),
        &out_b,
        1e-3,
    );

    coeus_autograd::sum(&out_c).backward();
    assert_close_rel(
        "transformer_enc_identity_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &dx_b,
        1e-3,
    );
}

// ── TransformerEncoderLayer forward shape + gradient contract ─────────────────
//
// A full value comparison against Burn NdArray isn't possible because the
// weight initialisation is different. We instead verify the shape contract and
// that backward correctly computes non-zero gradients for all parameters.

#[test]
fn transformer_encoder_layer_forward_backward_shape_contract() {
    use coeus_nn::{Module, NullMask, TransformerEncoderLayer};
    const H: usize = 2;
    let (batch, seq, d_model, d_ff) = (2, 4, 8, 16);

    let layer =
        TransformerEncoderLayer::<f32, SequentialBackend, H, NullMask>::new(d_model, d_ff, 0.0);
    let params = layer.parameters();
    assert!(!params.is_empty(), "encoder layer has parameters");

    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(
            vec![batch, seq, d_model],
            &vec![0.1f32; batch * seq * d_model],
        ),
        true,
    );
    let out = layer.forward(&x);
    assert_eq!(
        out.tensor.shape(),
        &[batch, seq, d_model],
        "encoder layer output shape"
    );

    // Backward: verify gradients flow to input and all parameters.
    coeus_autograd::sum(&out).backward();
    assert!(x.grad().is_some(), "encoder layer: input grad must be set");
    let input_grad_numel: f32 = x.grad().unwrap().as_slice().iter().map(|v| v.abs()).sum();
    assert!(
        input_grad_numel > 0.0,
        "encoder layer: input grad is all zero"
    );
    for (i, p) in params.iter().enumerate() {
        assert!(
            p.grad().is_some(),
            "encoder layer parameter {i} has no gradient"
        );
    }
}

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

// ── BatchNorm1d backward analytical parity ───────────────────────────────────

#[test]
fn batchnorm1d_backward_bias_and_weight_grads_match_analytical() {
    use coeus_nn::BatchNorm1d;
    // [N=1, C=2, L=3]: m = N*L = 3 per channel.
    // Channel 0: [1, 2, 3] → x_hat zero-mean → weight grad ≈ 0, bias grad = 3.
    // Channel 1: [4, 5, 6] → x_hat zero-mean → weight grad ≈ 0, bias grad = 3.
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [N=1, C=2, L=3]
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &data),
        true,
    );
    let bn = BatchNorm1d::<f32, SequentialBackend>::new(2, 1e-5, 0.1);
    use coeus_nn::Module;
    let out = bn.forward(&xv);
    // Loss = sum(output) → upstream gradient is all-ones.
    coeus_autograd::sum(&out).backward();

    // Bias gradient: dL/d_beta_c = sum_{N,L} 1 = m = 3 for each channel.
    let bg = bn
        .bias
        .grad()
        .expect("bias.grad must be set after backward");
    let bg_s = bg.as_slice();
    assert_eq!(bg_s.len(), 2, "bias grad length");
    let m_f = 3.0_f32;
    assert!(
        (bg_s[0] - m_f).abs() < 1e-4,
        "bias[0] grad={} expected={m_f}",
        bg_s[0]
    );
    assert!(
        (bg_s[1] - m_f).abs() < 1e-4,
        "bias[1] grad={} expected={m_f}",
        bg_s[1]
    );

    // Weight gradient: dL/d_gamma_c = sum_{N,L} x_hat_{c} * 1 = 0 (x_hat zero-mean).
    let wg = bn
        .weight
        .grad()
        .expect("weight.grad must be set after backward");
    let wg_s = wg.as_slice();
    assert_eq!(wg_s.len(), 2, "weight grad length");
    assert!(
        wg_s[0].abs() < 1e-4,
        "weight[0] grad={} expected≈0",
        wg_s[0]
    );
    assert!(
        wg_s[1].abs() < 1e-4,
        "weight[1] grad={} expected≈0",
        wg_s[1]
    );

    // Input gradient must be set and have same shape as input.
    let ig = xv.grad().expect("input.grad must be set after backward");
    assert_eq!(ig.ndim(), 3, "input grad rank");
    assert_eq!(ig.shape(), &[1, 2, 3], "input grad shape");
    // BatchNorm1d backward: sum of input grads per channel = 0 (normalization
    // property: dl/dx sums to zero across the normalization dim).
    let ig_s = ig.as_slice();
    let ig_c0_sum: f32 = ig_s[..3].iter().sum();
    let ig_c1_sum: f32 = ig_s[3..].iter().sum();
    assert!(
        ig_c0_sum.abs() < 1e-4,
        "input grad[c0] sum={ig_c0_sum} expected≈0"
    );
    assert!(
        ig_c1_sum.abs() < 1e-4,
        "input grad[c1] sum={ig_c1_sum} expected≈0"
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

// ── SGD step (manual reference, no Burn dep) ──────────────────────────────────
//
// SGD without momentum: θ_new = θ - lr * g
// This test uses an exact closed-form reference rather than live Burn,
// because Burn's SGD API requires a TrainingConfig setup that differs.

#[test]
fn sgd_step_matches_analytical_reference() {
    use coeus_optim::traits::Optimizer;
    use coeus_optim::SGD;
    let lr = 0.1f64;
    let momentum = 0.0f64;
    let params_data = vec![3.0f64, -2.0, 1.5, 0.5];
    let grads_data = vec![1.0f64, 2.0, -0.5, 4.0];

    // Set up param as Var with requires_grad = true.
    let p = Var::new(
        CoeusTensor::<f64, SequentialBackend>::from_slice(vec![4], &params_data),
        true,
    );
    // Manually inject gradient.
    if let Some(ref g) = p.grad {
        *g.write() = CoeusTensor::from_slice(vec![4], &grads_data);
    }

    let mut opt = SGD::new(vec![p.clone()], lr, momentum);
    opt.step();
    opt.zero_grad();

    // Expected: θ_new[i] = θ[i] - lr * g[i]
    let expected: Vec<f64> = params_data
        .iter()
        .zip(grads_data.iter())
        .map(|(&theta, &g)| theta - lr * g)
        .collect();
    let actual = opt.params[0].tensor.as_slice().to_vec();
    assert_close_rel(
        "sgd_step",
        &actual.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &expected.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        1e-6,
    );
}

// ── Adam step (manual reference) ─────────────────────────────────────────────
//
// Adam update rule (no weight decay, β1=0.9, β2=0.999, ε=1e-8):
//   m_t = β1 * m_{t-1} + (1 - β1) * g_t
//   v_t = β2 * v_{t-1} + (1 - β2) * g_t²
//   m̂_t = m_t / (1 - β1^t)
//   v̂_t = v_t / (1 - β2^t)
//   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

#[test]
fn adam_step_matches_analytical_reference() {
    use coeus_optim::traits::Optimizer;
    use coeus_optim::Adam;
    let lr = 0.001f64;
    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let eps = 1e-8f64;
    let params_data = vec![1.0f64, -0.5, 2.0, 0.3];
    let grads_data = vec![0.5f64, 1.0, -0.2, 0.8];

    let p = Var::new(
        CoeusTensor::<f64, SequentialBackend>::from_slice(vec![4], &params_data),
        true,
    );
    if let Some(ref g) = p.grad {
        *g.write() = CoeusTensor::from_slice(vec![4], &grads_data);
    }

    let mut opt = Adam::new(vec![p.clone()], lr, beta1, beta2, eps);
    opt.step(); // step t=1

    // Closed-form expected for t=1:
    let expected: Vec<f64> = params_data
        .iter()
        .zip(grads_data.iter())
        .map(|(&theta, &g)| {
            let m1 = (1.0 - beta1) * g;
            let v1 = (1.0 - beta2) * g * g;
            let m_hat = m1 / (1.0 - beta1);
            let v_hat = v1 / (1.0 - beta2);
            theta - lr * m_hat / (v_hat.sqrt() + eps)
        })
        .collect();
    let actual = opt.params[0].tensor.as_slice().to_vec();
    assert_close_rel(
        "adam_step",
        &actual.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &expected.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        1e-5,
    );
}

// ── ConvTranspose1d backward (gradient correctness) ───────────────────────────

#[test]
fn conv_transpose1d_backward_gradient_correctness() {
    // Input [1,1,2], weight [1,1,2], no bias, stride=1.
    // Forward: output [1,1,3] (L_out = (2-1)*1 + 2 = 3)
    // seed = [1, 1, 1] (all-ones grad)
    // grad_input[0,0,i] = Σ_{co,k} seed[0,co, i*1+k] * w[0,co,k]
    //   → position 0: seed[0]=1.0 * w[0]=1.0 + seed[1]=1.0 * w[1]=0.5 = 1.5
    //   → position 1: seed[1]=1.0 * w[0]=1.0 + seed[2]=1.0 * w[1]=0.5 = 1.5
    // grad_weight[0,0,k] = Σ_{n,l} input[n,0,l] * seed[n,0, l+k]
    //   → k=0: 2.0*1 + 3.0*1 = 5.0  (l=0→seed[0], l=1→seed[1])
    //   → k=1: 2.0*1 + 3.0*1 = 5.0  (l=0→seed[1], l=1→seed[2])
    let input = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2], &[2.0, 3.0]),
        true,
    );
    let weight = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2], &[1.0, 0.5]),
        true,
    );
    let backend = SequentialBackend::new();
    let out_tensor =
        coeus_ops::conv_transpose1d(&input.tensor, &weight.tensor, None, 1, 0, 0, 1, &backend);
    let out = coeus_autograd::conv_transpose1d(&input, &weight, &None, out_tensor, 1, 0, 0, 1);
    let seed = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3], &[1.0, 1.0, 1.0]);
    out.backward_with_seed(seed);

    let gi = input.grad().unwrap();
    let gw = weight.grad().unwrap();
    assert_close_rel("ct1d_bwd_gi", gi.as_slice(), &[1.5, 1.5], 1e-5);
    assert_close_rel("ct1d_bwd_gw", gw.as_slice(), &[5.0, 5.0], 1e-5);
    let _ = backend;
}

// ── ConvTranspose2d backward (gradient correctness) ───────────────────────────

#[test]
fn conv_transpose2d_backward_gradient_correctness() {
    // Input [1,1,2,2] all-ones, weight [1,1,1,1]=[2.0], bias [1]=[0].
    // Forward: out[n,0,h,w] = input[n,0,h,w]*2.0 = 2.0 (all positions) → [2,2,2,2]
    // Backward seed = all-ones [1,1,2,2]:
    //   grad_input[n,0,h,w] = Σ_k seed[n,0,h,w] * weight = 1*2 = 2
    //   grad_weight = Σ_{n,h,w} input*seed = 1*1 * 4 elements = 4
    let input = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &[1.0; 4]),
        true,
    );
    let weight = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 1, 1], &[2.0]),
        true,
    );
    let backend = SequentialBackend::new();
    let out_tensor =
        coeus_ops::conv_transpose2d(&input.tensor, &weight.tensor, None, 1, 0, 0, 1, &backend);
    let out = coeus_autograd::conv_transpose2d(&input, &weight, &None, out_tensor, 1, 0, 0, 1);
    assert_eq!(out.tensor.shape(), &[1, 1, 2, 2]);
    assert_close("ct2d_fwd", out.tensor.to_contiguous().as_slice(), &[2.0; 4]);

    let seed = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &[1.0; 4]);
    out.backward_with_seed(seed);

    let gi = input.grad().unwrap();
    let gw = weight.grad().unwrap();
    assert_close("ct2d_bwd_gi", gi.to_contiguous().as_slice(), &[2.0; 4]);
    assert_close("ct2d_bwd_gw", gw.to_contiguous().as_slice(), &[4.0]);
    let _ = backend;
}

// ── ConvTranspose1d backward matches Burn autodiff ───────────────────────────
//
// Input [N=1, Cin=1, L=3], weight [Cin=1, Cout=1, K=2], no bias, stride=1.
// L_out = (3-1)*1 + 2 = 4.
// Burn weight shape matches Coeus: [Cin, Cout, K].
// Tolerance: standard 1e-4 (f32 accumulation over short kernel).

#[test]
fn conv_transpose1d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::nn::conv::ConvTranspose1dConfig;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, cin, cout, l, k) = (1usize, 1, 1, 3, 2);
    let l_out = (l - 1) + k; // stride=1, no padding
    let data: Vec<f32> = vec![0.5, -0.3, 0.8];
    let w_vec: Vec<f32> = vec![0.7, -0.4];

    // Coeus: autograd forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, cin, l], &data),
        true,
    );
    let wv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![cin, cout, k], &w_vec),
        true,
    );
    let backend = SequentialBackend::new();
    let out_fwd = coeus_ops::conv_transpose1d(&xv.tensor, &wv.tensor, None, 1, 0, 0, 1, &backend);
    let out_c = coeus_autograd::conv_transpose1d(&xv, &wv, &None, out_fwd, 1, 0, 0, 1);
    assert_eq!(out_c.tensor.shape(), &[n, cout, l_out]);
    coeus_autograd::sum(&out_c).backward();

    // Burn autodiff reference.
    let xb: BurnTensor<AB, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, cin, l]), &device).require_grad();
    let mut conv_b = ConvTranspose1dConfig::new([cin, cout], k)
        .with_bias(false)
        .init::<AB>(&device);
    conv_b.weight =
        burn::module::Param::from_data(TensorData::new(w_vec.clone(), [cin, cout, k]), &device);
    let out_b = conv_b.forward(xb.clone());
    let grads = out_b.sum().backward();

    let to_vec = |t: BurnTensor<NdArray<f32>, 3>| t.into_data().to_vec::<f32>().unwrap();
    let to_vecw = |t: BurnTensor<NdArray<f32>, 3>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "ct1d_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &to_vec(xb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "ct1d_bwd_dw",
        wv.grad().unwrap().as_slice(),
        &to_vecw(conv_b.weight.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── ConvTranspose2d backward matches Burn autodiff ───────────────────────────
//
// Input [N=1, Cin=1, H=3, W=3], weight [Cin=1, Cout=1, Kh=2, Kw=2], no bias, stride=1.
// H_out = (3-1)*1 + 2 = 4, W_out = 4.
// Burn weight shape: [Cin, Cout, Kh, Kw] matches Coeus.
// Tolerance: 1e-4 from f32 accumulation.

#[test]
fn conv_transpose2d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::nn::conv::ConvTranspose2dConfig;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, cin, cout, h, w, kh, kw) = (1usize, 1, 1, 3, 3, 2, 2);
    let data: Vec<f32> = (0..n * cin * h * w).map(|x| x as f32 * 0.1 - 0.4).collect();
    let w_vec: Vec<f32> = vec![0.6, -0.2, 0.3, -0.5];

    // Coeus.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, cin, h, w], &data),
        true,
    );
    let wv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![cin, cout, kh, kw], &w_vec),
        true,
    );
    let backend = SequentialBackend::new();
    let out_fwd = coeus_ops::conv_transpose2d(&xv.tensor, &wv.tensor, None, 1, 0, 0, 1, &backend);
    let out_c = coeus_autograd::conv_transpose2d(&xv, &wv, &None, out_fwd, 1, 0, 0, 1);
    coeus_autograd::sum(&out_c).backward();

    // Burn autodiff reference.
    let xb: BurnTensor<AB, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, cin, h, w]), &device)
            .require_grad();
    let mut conv_b = ConvTranspose2dConfig::new([cin, cout], [kh, kw])
        .with_bias(false)
        .init::<AB>(&device);
    conv_b.weight = burn::module::Param::from_data(
        TensorData::new(w_vec.clone(), [cin, cout, kh, kw]),
        &device,
    );
    let out_b = conv_b.forward(xb.clone());
    let grads = out_b.sum().backward();

    let to_vec4 = |t: BurnTensor<NdArray<f32>, 4>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "ct2d_bwd_dx",
        xv.grad().unwrap().to_contiguous().as_slice(),
        &to_vec4(xb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "ct2d_bwd_dw",
        wv.grad().unwrap().to_contiguous().as_slice(),
        &to_vec4(conv_b.weight.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── AvgPool2d backward (Coeus vs. manual reference) ──────────────────────────

#[test]
fn avg_pool2d_backward_gradient_correctness() {
    // Input [1,1,4,4], kernel=2, stride=2 → output [1,1,2,2].
    // Backward seed = all-ones → each input element receives grad = 1/(ks*ks) = 0.25.
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        true,
    );
    let pool = coeus_nn::AvgPool2d::<f32, SequentialBackend>::with_params(2, 2, 0, 1);
    use coeus_nn::Module;
    let out = pool.forward(&xv);
    assert_eq!(out.tensor.shape(), &[1, 1, 2, 2]);

    let seed = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &[1.0; 4]);
    out.backward_with_seed(seed);

    let grad = xv.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 1, 4, 4], "avg_pool2d grad shape");
    // Each output element distributes its gradient equally over ks^2=4 input positions.
    for &g in grad.to_contiguous().as_slice() {
        assert!(
            (g - 0.25f32).abs() < 1e-5,
            "avg_pool2d grad element: expected 0.25, got {g}"
        );
    }
}

// ── MaxPool2d backward (Coeus vs. manual reference) ──────────────────────────

#[test]
fn max_pool2d_backward_gradient_correctness() {
    // Input [1,1,4,4]: row-major 0..15.
    // 2×2 non-overlapping blocks (stride=2):
    //   block(0,0): positions [(0,0),(0,1),(1,0),(1,1)] = [0,1,4,5]  → max=5
    //   block(0,1): positions [(0,2),(0,3),(1,2),(1,3)] = [2,3,6,7]  → max=7
    //   block(1,0): positions [(2,0),(2,1),(3,0),(3,1)] = [8,9,12,13]→ max=13
    //   block(1,1): positions [(2,2),(2,3),(3,2),(3,3)] = [10,11,14,15]→max=15
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        true,
    );
    let pool = coeus_nn::MaxPool2d::<f32, SequentialBackend>::with_params(2, 2, 0, 1);
    use coeus_nn::Module;
    let out = pool.forward(&xv);
    assert_eq!(out.tensor.shape(), &[1, 1, 2, 2]);
    let expected_fwd = [5.0f32, 7.0, 13.0, 15.0];
    assert_close_rel(
        "max_pool2d_bwd_fwd",
        out.tensor.as_slice(),
        &expected_fwd,
        1e-5,
    );

    let seed = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &[1.0; 4]);
    out.backward_with_seed(seed);

    let grad = xv.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 1, 4, 4], "max_pool2d grad shape");
    let gs = grad.to_contiguous();
    let gs = gs.as_slice();
    // Max positions (row-major flat index in 4×4): 5, 7, 13, 15
    for (i, &v) in gs.iter().enumerate() {
        let is_max_pos = matches!(i, 5 | 7 | 13 | 15);
        if is_max_pos {
            assert!(
                (v - 1.0f32).abs() < 1e-5,
                "max_pool2d grad[{i}]: expected 1.0, got {v}"
            );
        } else {
            assert!(
                v.abs() < 1e-5,
                "max_pool2d grad[{i}]: expected 0.0, got {v}"
            );
        }
    }
}

// ── GroupNorm forward (matches Burn NdArray) ──────────────────────────────────

#[test]
fn groupnorm_forward_matches_burn() {
    use burn::nn::GroupNormConfig;
    use coeus_nn::GroupNorm;

    // [N=2, C=4, L=3] with G=2 groups.  Weight=ones, bias=zeros (default init).
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.1 - 1.0).collect();
    let (n, c, l) = (2usize, 4, 3);

    // Coeus
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, l], &data),
        false,
    );
    let gn = GroupNorm::<f32, SequentialBackend, 2>::new(c, 1e-5);
    let out_c = gn.forward(&xv);

    // Burn
    let gn_b = GroupNormConfig::new(2, c).init::<BurnBackend>(&dev());
    let xb: BurnTensor<BurnBackend, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, l]), &dev());
    let out_b = gn_b.forward(xb);

    // Tolerance derivation: Coeus computes sqrt(var + eps) (eps before sqrt,
    // the standard PyTorch formula), while Burn 0.16 computes sqrt(var) + eps
    // (eps after sqrt).  The relative difference between the two divisors is
    // approximately eps / (2 * sqrt(var)) ≈ 1e-5 / (2 * 0.17) ≈ 2.9e-5 for the
    // smallest group variance (~0.03).  After normalization (amplification
    // ~1/sqrt(var) ≈ 5.8) and accumulation across C*L output elements, the
    // worst-case absolute error bound is ~6 * 5.8 * 2.9e-5 ≈ 1e-3.
    assert_close_rel(
        "groupnorm_fwd",
        out_c.tensor.to_contiguous().as_slice(),
        &bvec(out_b),
        1e-3,
    );
}

// ── GroupNorm forward + backward (matches Burn autodiff) ──────────────────────

#[test]
fn groupnorm_forward_backward_match_burn() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::GroupNorm;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    // [N=2, C=4, L=3] with G=2 groups.  Custom weight/bias to test affine grads.
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.1 - 1.0).collect();
    let w = vec![1.2_f32, 0.8, 1.0, 0.9];
    let b = vec![0.1_f32, -0.1, 0.2, 0.0];
    let (n, c, l, g) = (2usize, 4, 3, 2);
    let eps = 1e-5_f32;
    let c_per_g = c / g;
    let hidden = c_per_g * l; // elements per group per sample

    // ── Coeus forward + backward ──
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, l], &data),
        true,
    );
    let mut gn = GroupNorm::<f32, SequentialBackend, 2>::new(c, eps as f64);
    gn.weight = Var::new(CoeusTensor::from_slice(vec![c], &w), true);
    gn.bias = Var::new(CoeusTensor::from_slice(vec![c], &b), true);
    let out_c = coeus_autograd::sum(&gn.forward(&xv));
    out_c.backward();

    // ── Burn forward + backward (manual GroupNorm formula) ──
    // Reshape [N, C, L] → [N, G, C/G * L] and normalize over dim 2, then
    // reshape back and apply per-channel affine — matching Burn's group_norm.
    let xb: BurnTensor<AB, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, l]), &device).require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(w, [c]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(b, [c]), &device).require_grad();

    let x_flat = xb.clone().reshape([n, g, hidden]);
    let mean = x_flat.clone().sum_dim(2) / hidden as f32;
    let xc = x_flat.sub(mean);
    // Use Coeus's formula sqrt(var + eps), NOT Burn's sqrt(var) + eps,
    // so the reference gradient matches Coeus's forward computation.
    let var = xc.clone().powf_scalar(2.0).sum_dim(2) / hidden as f32;
    let normed = xc.div(var.add_scalar(eps).sqrt());
    let normed_3d = normed.reshape([n, c, l]);
    let mut aff_shape = [1usize; 3];
    aff_shape[1] = c;
    let out_b = normed_3d.mul(wb.clone().reshape(aff_shape)) + bk.clone().reshape(aff_shape);
    let grads = out_b.sum().backward();

    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();
    let to_vec3 = |t: BurnTensor<NdArray<f32>, 3>| t.into_data().to_vec::<f32>().unwrap();

    // Backward parity: input gradient
    assert_close_rel(
        "groupnorm_bwd_dx",
        xv.grad().unwrap().to_contiguous().as_slice(),
        &to_vec3(xb.grad(&grads).unwrap()),
        1e-4,
    );

    // Backward parity: weight gradient
    assert_close_rel(
        "groupnorm_bwd_dw",
        gn.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-3,
    );

    // Backward parity: bias gradient
    assert_close_rel(
        "groupnorm_bwd_db",
        gn.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
}

#[test]
fn batchnorm1d_eval_uses_running_stats_without_update() {
    use coeus_nn::BatchNorm1d;

    let (n, c, l) = (2usize, 3, 2);
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, l], &data),
        false,
    );
    let running_mean = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![c], &[1.0, 2.0, 3.0]);
    let running_var = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![c], &[4.0, 9.0, 16.0]);
    let weight = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![c], &[1.0, 1.0, 1.0]),
        true,
    );
    let bias = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![c], &[0.0, 0.0, 0.0]),
        true,
    );
    let mut bn = BatchNorm1d::<f32, SequentialBackend>::from_parts(
        c,
        weight,
        bias,
        1e-5,
        0.1,
        running_mean.clone(),
        running_var.clone(),
    );
    bn.set_training(false);

    let out = bn.forward(&x);
    assert_eq!(out.tensor.shape(), &[n, c, l]);
    assert_eq!(bn.running_mean.borrow().as_slice(), running_mean.as_slice());
    assert_eq!(bn.running_var.borrow().as_slice(), running_var.as_slice());

    let expected0 = (data[0] - 1.0) / (4.0_f32 + 1e-5).sqrt();
    let got0 = out.tensor.as_slice()[0];
    assert!(
        (got0 - expected0).abs() < 1e-6,
        "batchnorm eval value: got {got0}, expected {expected0}"
    );
}

// ── BatchNorm1d eval-mode forward matches Burn NdArray ───────────────────────

#[test]
fn batchnorm1d_eval_forward_matches_burn() {
    use burn::nn::BatchNormConfig;
    use coeus_nn::BatchNorm1d;

    // Burn's non-autodiff NdArray backend runs BatchNorm in inference (eval)
    // mode, using running statistics rather than batch statistics.  Both
    // implementations default to running_mean=0 and running_var=1 so the
    // eval outputs are directly comparable.
    let data: Vec<f32> = (0..12).map(|x| x as f32 - 5.5).collect();
    let (n, c, l) = (2usize, 2, 3);

    // Coeus in eval mode: running_mean=0, running_var=1, weight=1, bias=0.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, l], &data),
        false,
    );
    let mut bn = BatchNorm1d::<f32, SequentialBackend>::new(c, 1e-5, 0.1);
    bn.set_training(false);
    let out_c = bn.forward(&xv);

    // Burn (NdArray = eval mode by default): running_mean=0, running_var=1.
    let bn_b: burn::nn::BatchNorm<BurnBackend, 1> =
        BatchNormConfig::new(c).init::<BurnBackend, 1>(&dev());
    let xb: BurnTensor<BurnBackend, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, l]), &dev());
    let out_b = bvec(bn_b.forward(xb));

    // Tolerance derivation: Burn 0.16 uses sqrt(var) + eps (eps after sqrt),
    // coeus uses sqrt(var + eps).  With running_var=1 and eps=1e-5 the
    // denominators are sqrt(1+1e-5) ≈ 1.000005 vs (1+1e-5) = 1.00001 —
    // difference ≈ 5e-6.  For |x| ≤ 5.5 the per-element error ≤ 2.8e-5.
    // tol = 1e-4 covers this plus f32 rounding.
    assert_close_rel(
        "batchnorm1d_eval_vs_burn",
        out_c.tensor.as_slice(),
        &out_b,
        1e-4,
    );
}

// ── BatchNorm2d eval-mode forward matches Burn NdArray ───────────────────────

#[test]
fn batchnorm2d_eval_forward_matches_burn() {
    use burn::nn::BatchNormConfig;
    use coeus_nn::BatchNorm2d;

    // Input [N=2, C=2, H=3, W=3] = 36 elements.
    // running_mean=0, running_var=1 on both sides; eval mode throughout.
    // Tolerance derivation identical to BatchNorm1d: |err| ≤ 2.8e-5 for |x|≤8.5,
    // plus f32 rounding → tol = 1e-4.
    let data: Vec<f32> = (0..36).map(|x| x as f32 - 17.5).collect();
    let (n, c, h, w) = (2usize, 2, 3, 3);

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, h, w], &data),
        false,
    );
    let mut bn = BatchNorm2d::<f32, SequentialBackend>::new(c, 1e-5, 0.1);
    bn.set_training(false);
    let out_c = bn.forward(&xv);

    let bn_b: burn::nn::BatchNorm<BurnBackend, 2> =
        BatchNormConfig::new(c).init::<BurnBackend, 2>(&dev());
    let xb: BurnTensor<BurnBackend, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, h, w]), &dev());
    let out_b = bvec(bn_b.forward(xb));

    assert_close_rel(
        "batchnorm2d_eval_vs_burn",
        out_c.tensor.as_slice(),
        &out_b,
        1e-4,
    );
}

// ── InstanceNorm forward (matches Burn NdArray) ──────────────────────────────

#[test]
fn instancenorm_forward_matches_burn() {
    use burn::nn::InstanceNormConfig;
    use coeus_nn::InstanceNorm1d;

    // [N=2, C=3, L=4] with weight=ones, bias=zeros (default init).
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.1 - 0.5).collect();
    let (n, c, l) = (2usize, 3, 4);

    // Coeus
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, l], &data),
        false,
    );
    let inorm = InstanceNorm1d::<f32, SequentialBackend>::new(c, 1e-5);
    let out_c = inorm.forward(&xv);

    // Burn
    let in_b = InstanceNormConfig::new(c).init::<BurnBackend>(&dev());
    let xb: BurnTensor<BurnBackend, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, l]), &dev());
    let out_b = in_b.forward(xb);

    // Tolerance: same sqrt(var+eps) vs sqrt(var)+eps formula difference as
    // GroupNorm.  See groupnorm_forward_matches_burn for the derivation.
    assert_close_rel(
        "instancenorm_fwd",
        out_c.tensor.to_contiguous().as_slice(),
        &bvec(out_b),
        1e-3,
    );
}

// ── Embedding forward (matches Burn NdArray) ───────────────────────────────────

#[test]
fn embedding_forward_matches_burn() {
    use burn::tensor::Int;
    use coeus_nn::Embedding;

    // Weight [num_embeddings=5, embedding_dim=3] with known values.
    let weights: Vec<f32> = (0..15).map(|x| x as f32 * 0.1).collect();
    let (n_emb, d_model) = (5usize, 3);

    // Indices [batch=2, seq=3]: token ids into the embedding table.
    let indices: [[i32; 3]; 2] = [[0, 2, 4], [1, 3, 0]];

    // ── Coeus ──
    let w_tensor =
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n_emb, d_model], &weights);
    let weight_var = Var::new(w_tensor, false);
    let mut emb = Embedding::<f32, SequentialBackend>::new(n_emb, d_model);
    // Override the default ones weight with our known values.
    emb.weight = weight_var;

    // Coeus embedding expects float indices (same Scalar trait as weights).
    let idx_flat: Vec<f32> = indices
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v as f32)
        .collect();
    let idx_tensor = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &idx_flat);
    let out_c = emb.forward_indices(&idx_tensor);

    // ── Burn ──
    let wb: BurnTensor<BurnBackend, 2> =
        BurnTensor::from_data(TensorData::new(weights.clone(), [n_emb, d_model]), &dev());
    let ib: BurnTensor<BurnBackend, 2, Int> = BurnTensor::from_ints(indices, &dev());
    let out_b = burn::tensor::module::embedding(wb, ib);

    assert_close(
        "embedding_fwd",
        out_c.tensor.to_contiguous().as_slice(),
        &bvec(out_b),
    );
}

// ── Embedding forward + backward (matches Burn autodiff) ───────────────────────

#[test]
fn embedding_forward_backward_match_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::Int;
    use coeus_nn::Embedding;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    // Weight [num_embeddings=4, embedding_dim=2] with known values.
    let weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let (n_emb, d_model) = (4usize, 2);

    // Indices [batch=2, seq=2].
    let indices: [[i32; 2]; 2] = [[0, 2], [3, 1]];

    // ── Coeus forward + backward ──
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n_emb, d_model], &weights),
        true,
    );
    let mut emb = Embedding::<f32, SequentialBackend>::new(n_emb, d_model);
    emb.weight = xv.clone();

    let idx_flat: Vec<f32> = indices
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v as f32)
        .collect();
    let idx_tensor = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &idx_flat);
    let out_c = coeus_autograd::sum(&emb.forward_indices(&idx_tensor));
    out_c.backward();

    // ── Burn forward + backward ──
    let wb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(weights.clone(), [n_emb, d_model]), &device)
            .require_grad();
    let ib: BurnTensor<AB, 2, Int> = BurnTensor::from_ints(indices, &device);
    let out_b = burn::tensor::module::embedding(wb.clone(), ib);
    let grads = out_b.sum().backward();

    // Forward parity
    let out_b_fwd: Vec<f32> = {
        // Re-compute forward without autodiff for value comparison.
        let wb_fwd: BurnTensor<BurnBackend, 2> =
            BurnTensor::from_data(TensorData::new(weights.clone(), [n_emb, d_model]), &dev());
        let ib_fwd: BurnTensor<BurnBackend, 2, Int> = BurnTensor::from_ints(indices, &dev());
        bvec(burn::tensor::module::embedding(wb_fwd, ib_fwd))
    };
    assert_close(
        "embedding_fwd_custom",
        emb.forward_indices(&idx_tensor)
            .tensor
            .to_contiguous()
            .as_slice(),
        &out_b_fwd,
    );

    // Backward parity: weight gradient
    // Embedding backward scatters the output gradient into the weight rows
    // indexed by the input.  With sum loss, each indexed row gets +1 per
    // occurrence.
    let to_vec = |t: BurnTensor<NdArray<f32>, 2>| t.into_data().to_vec::<f32>().unwrap();
    assert_close(
        "embedding_bwd_dw",
        xv.grad().unwrap().to_contiguous().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
    );
}

// ── Embedding backward: gradient accumulation for repeated indices ────────────

#[test]
fn embedding_backward_accumulates_grad_for_repeated_indices() {
    // Input indices [0, 1, 0] — index 0 appears twice.
    // With sum loss, weight.grad[0] should be 2 × grad_out_row and weight.grad[1] = 1 × grad_out_row.
    let weights = vec![1.0f32, 2.0, 3.0, 4.0]; // [vocab=2, dim=2]
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &weights),
        true,
    );

    let idx = vec![0i32, 1, 0];
    let idx_tensor = CoeusTensor::<i32, SequentialBackend>::from_slice(vec![3], &idx);

    let out = coeus_autograd::embedding(&xv, &idx_tensor);
    assert_eq!(out.tensor.shape(), &[3, 2]);
    // out[0] = weights[0] = [1, 2], out[1] = weights[1] = [3, 4], out[2] = weights[0] = [1, 2]

    // Backward with all-ones grad → accumulation should double grad for row 0.
    coeus_autograd::sum(&out).backward();

    let gw = xv.grad().unwrap();
    // row 0 was accessed twice → grad = 2.0 for each element
    assert!(
        (gw.as_slice()[0] - 2.0).abs() < 1e-6,
        "grad[0,0]={}",
        gw.as_slice()[0]
    );
    assert!(
        (gw.as_slice()[1] - 2.0).abs() < 1e-6,
        "grad[0,1]={}",
        gw.as_slice()[1]
    );
    // row 1 was accessed once → grad = 1.0 for each element
    assert!(
        (gw.as_slice()[2] - 1.0).abs() < 1e-6,
        "grad[1,0]={}",
        gw.as_slice()[2]
    );
    assert!(
        (gw.as_slice()[3] - 1.0).abs() < 1e-6,
        "grad[1,1]={}",
        gw.as_slice()[3]
    );
}

#[test]
fn embedding_padding_idx_zeroes_grad_for_pad_token() {
    let emb = coeus_nn::Embedding::<f32, SequentialBackend>::with_padding_idx(4, 3, 0);

    let w_slice = emb.weight.tensor.as_slice();
    assert_eq!(
        &w_slice[0..3],
        &[0.0, 0.0, 0.0],
        "padding_idx row must be zero-initialized"
    );

    let idx = CoeusTensor::<i32, SequentialBackend>::from_slice(vec![4], &[0, 1, 2, 0]);
    let out = emb.forward_indices(&idx);
    assert_eq!(out.tensor.shape(), &[4, 3]);

    coeus_autograd::sum(&out).backward();
    let gw = emb.weight.grad().expect("embedding weight must have grad");
    let gw_s = gw.as_slice();

    assert_eq!(
        &gw_s[0..3],
        &[0.0, 0.0, 0.0],
        "padding_idx gradient row must stay zero"
    );
    assert_eq!(&gw_s[3..6], &[1.0, 1.0, 1.0]);
    assert_eq!(&gw_s[6..9], &[1.0, 1.0, 1.0]);
    assert_eq!(&gw_s[9..12], &[0.0, 0.0, 0.0]);
}

// ── New activation parity tests (sigmoid, tanh, silu, log_softmax, ──────────
// ── leaky_relu, softplus, mish) against Burn 0.16 NdArray reference ─────────
//
// Each test uses a small [2, 4] deterministic tensor that contains negative,
// zero, and positive values so all activation branches are exercised.
// Tolerance: 512 * f32::EPSILON * (1 + |ref|)  — same as the file-global
// `assert_close`, which accounts for up to 512 ULP of accumulated rounding.

fn act_input() -> Vec<f32> {
    vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
}

fn burn_act_tensor() -> BurnTensor<BurnBackend, 2> {
    BurnTensor::from_data(TensorData::new(act_input(), [2, 4]), &dev())
}

fn coeus_act_tensor() -> CoeusTensor<f32, SequentialBackend> {
    CoeusTensor::<f32, SequentialBackend>::from_slice([2, 4], &act_input())
}

#[test]
fn sigmoid_matches_burn() {
    let s = SequentialBackend::new();
    let coeus_out = sigmoid(&coeus_act_tensor(), &s);
    let burn_out: Vec<f32> = bvec(burn_act::sigmoid(burn_act_tensor()));
    assert_close("sigmoid", coeus_out.as_slice(), &burn_out);
}

#[test]
fn tanh_matches_burn() {
    let s = SequentialBackend::new();
    let coeus_out = tanh(&coeus_act_tensor(), &s);
    let burn_out: Vec<f32> = bvec(burn_act::tanh(burn_act_tensor()));
    assert_close("tanh", coeus_out.as_slice(), &burn_out);
}

#[test]
fn silu_matches_burn() {
    // Burn: silu(x) = x * sigmoid(x)
    let s = SequentialBackend::new();
    let coeus_out = silu(&coeus_act_tensor(), &s);
    let burn_out: Vec<f32> = bvec(burn_act::silu(burn_act_tensor()));
    assert_close("silu", coeus_out.as_slice(), &burn_out);
}

#[test]
fn log_softmax_matches_burn() {
    // Burn log_softmax along dim=1 (last axis of a [2,4] tensor).
    let s = SequentialBackend::new();
    let coeus_out = log_softmax_axis(&coeus_act_tensor(), 1, &s);
    let burn_out: Vec<f32> = bvec(burn_act::log_softmax(burn_act_tensor(), 1));
    assert_close("log_softmax", coeus_out.as_slice(), &burn_out);
}

#[test]
fn leaky_relu_matches_burn() {
    // Burn leaky_relu with negative_slope = 0.01.
    let s = SequentialBackend::new();
    let coeus_out = leaky_relu(&coeus_act_tensor(), &s, 0.01);
    let burn_out: Vec<f32> = bvec(burn_act::leaky_relu(burn_act_tensor(), 0.01));
    assert_close("leaky_relu", coeus_out.as_slice(), &burn_out);
}

#[test]
fn softplus_matches_burn() {
    // Burn softplus with beta = 1.0.
    // Coeus: log(1 + exp(beta * x)) / beta where beta = 1.
    let s = SequentialBackend::new();
    let coeus_out = softplus(&coeus_act_tensor(), &s);
    // Burn signature: softplus(tensor, beta: f64)
    let burn_out: Vec<f32> = bvec(burn_act::softplus(burn_act_tensor(), 1.0));
    assert_close("softplus", coeus_out.as_slice(), &burn_out);
}

#[test]
fn mish_matches_burn() {
    // Burn: mish(x) = x * tanh(softplus(x, 1.0))
    // Coeus mirrors this composition so results are within rounding.
    let s = SequentialBackend::new();
    let coeus_out = mish(&coeus_act_tensor(), &s);
    let burn_out: Vec<f32> = bvec(burn_act::mish(burn_act_tensor()));
    assert_close("mish", coeus_out.as_slice(), &burn_out);
}

// ── cat backward: gradient routing ────────────────────────────────────────────

#[test]
fn cat_backward_routes_grad_to_each_input() {
    // cat([x, y], dim=0) then backward-with-ones should give x.grad = ones[:x_len]
    // and y.grad = ones[x_len:].
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &[1.0f32; 6]),
        true,
    );
    let y = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![3, 3], &[2.0f32; 9]),
        true,
    );
    let out = coeus_autograd::cat(&[&x, &y], 0);
    assert_eq!(out.tensor.shape(), &[5, 3]);

    let seed = CoeusTensor::<f32, SequentialBackend>::ones(vec![5, 3]);
    out.backward_with_seed(seed);

    // x.grad and y.grad should both be all-ones (identity backward through cat).
    let gx = x.grad().expect("x.grad must be set");
    let gy = y.grad().expect("y.grad must be set");
    for &v in gx.as_slice() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "cat_bwd x.grad: expected 1.0 got {v}"
        );
    }
    for &v in gy.as_slice() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "cat_bwd y.grad: expected 1.0 got {v}"
        );
    }
}

// ── where_cond backward: gradient to true/false branches ─────────────────────

#[test]
fn where_cond_backward_routes_grad_correctly() {
    // where(cond, on_true, on_false) — already tested via burn parity; extra check
    // that gradient is exactly zero at masked positions.
    let cond = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0, 0.0, 1.0, 0.0]),
        false,
    );
    let on_true = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[10.0, 11.0, 12.0, 13.0]),
        true,
    );
    let on_false = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[20.0, 21.0, 22.0, 23.0]),
        true,
    );
    let out = coeus_autograd::where_cond(&cond, &on_true, &on_false);
    assert_eq!(out.tensor.as_slice(), &[10.0, 21.0, 12.0, 23.0]);

    let seed = CoeusTensor::<f32, SequentialBackend>::ones(vec![4]);
    out.backward_with_seed(seed);

    let gt = on_true.grad().expect("on_true grad must be set");
    let gf = on_false.grad().expect("on_false grad must be set");
    // Gradient flows where cond==1 for on_true, where cond==0 for on_false.
    assert_close("where_true_grad", gt.as_slice(), &[1.0, 0.0, 1.0, 0.0]);
    assert_close("where_false_grad", gf.as_slice(), &[0.0, 1.0, 0.0, 1.0]);
}

// ── Dropout backward: gradient masked correctly ───────────────────────────────

#[test]
fn dropout_backward_masks_gradient() {
    // Use p=0 (no dropout) to verify identity, then check that gradient
    // is zero at dropped positions when p > 0 (seed-deterministic).
    use coeus_nn::{Dropout, Module};

    // p=0: identity pass-through.
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![4], &[1.0, 2.0, 3.0, 4.0]),
        true,
    );
    let drop0 = Dropout::new(0.0);
    let out0 = drop0.forward(&x);
    assert_eq!(
        out0.tensor.as_slice(),
        x.tensor.as_slice(),
        "p=0 should be identity"
    );

    // Backward with p=0 passes gradient unchanged.
    coeus_autograd::sum(&out0).backward();
    let gx0 = x.grad().unwrap();
    for &v in gx0.as_slice() {
        assert!((v - 1.0).abs() < 1e-6, "p=0 grad should be 1.0, got {v}");
    }

    // p > 0 training: output is scaled (some elements may be zero).
    // We only check the mask consistency: grad is non-negative (mask is 0 or 1/(1-p)).
    let x2 = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![100], &vec![1.0f32; 100]),
        true,
    );
    let drop_p = Dropout::new(0.5);
    let out_p = drop_p.forward(&x2);
    coeus_autograd::sum(&out_p).backward();
    let gx2 = x2.grad().unwrap();
    // All gradient values must be either 0 or ≥ 1 (scale = 2.0 for p=0.5).
    for &v in gx2.as_slice() {
        assert!(
            v == 0.0 || v >= 1.0,
            "dropout backward: unexpected grad {v}"
        );
    }
}

// ── BatchNorm3d eval-mode forward matches Burn NdArray ───────────────────────

#[test]
fn batchnorm3d_eval_forward_matches_burn() {
    use burn::nn::BatchNormConfig;
    use coeus_nn::BatchNorm3d;

    // Input [N=2, C=2, D=2, H=3, W=3] = 72 elements.
    // running_mean=0, running_var=1 on both sides; eval mode throughout.
    // Tolerance derivation: identical to BatchNorm1d/2d; |err| ≤ 2.8e-5 for
    // |x| ≤ 8.5 from the sqrt(var+eps) vs sqrt(var)+eps formula difference.
    // tol = 1e-4 accounts for f32 rounding.
    let data: Vec<f32> = (0..72).map(|x| x as f32 - 35.5).collect();
    let (n, c, d, h, w) = (2usize, 2, 2, 3, 3);

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, d, h, w], &data),
        false,
    );
    let mut bn = BatchNorm3d::<f32, SequentialBackend>::new(c, 1e-5, 0.1);
    bn.set_training(false);
    let out_c = bn.forward(&xv);

    let bn_b: burn::nn::BatchNorm<BurnBackend, 3> =
        BatchNormConfig::new(c).init::<BurnBackend, 3>(&dev());
    let xb: BurnTensor<BurnBackend, 5> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, d, h, w]), &dev());
    let out_b = bvec(bn_b.forward(xb));

    assert_close_rel(
        "batchnorm3d_eval_vs_burn",
        out_c.tensor.as_slice(),
        &out_b,
        1e-4,
    );
}

// ── Conv1d forward matches Burn NdArray (ones kernel) ────────────────────────

#[test]
fn conv1d_forward_matches_burn() {
    use burn::nn::conv::Conv1dConfig;
    use burn::nn::PaddingConfig1d;
    use coeus_nn::Conv1d;

    // C_in=2, C_out=1, K=3, no bias, ones weight → output[j] = sum(input[:, j:j+3]).
    // Input [N=1, C_in=2, L=6]; valid conv → output [N=1, C_out=1, L=4].
    // Tolerance: 512 * f32::EPSILON * (1 + max_output).
    let data: Vec<f32> = (0..12).map(|x| x as f32 * 0.1).collect();
    let (n, ic, oc, l, k) = (1usize, 2, 1, 6, 3);
    let out_len = l - k + 1;
    let w_vec = vec![1.0f32; oc * ic * k];

    // Coeus: weight=ones, no bias.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, l], &data),
        false,
    );
    let mut conv_c = Conv1d::<f32, SequentialBackend>::new(ic, oc, k, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k], &w_vec), true);
    let out_c = conv_c.forward(&xv);

    // Burn: identical weight.
    let xb: BurnTensor<BurnBackend, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, l]), &dev());
    let mut conv_b = Conv1dConfig::new(ic, oc, k)
        .with_bias(false)
        .with_padding(PaddingConfig1d::Valid)
        .init::<BurnBackend>(&dev());
    conv_b.weight =
        burn::module::Param::from_data(TensorData::new(w_vec.clone(), [oc, ic, k]), &dev());
    let out_b = bvec(conv_b.forward(xb));

    assert_eq!(out_c.tensor.shape(), &[n, oc, out_len]);
    assert_close("conv1d_vs_burn", out_c.tensor.as_slice(), &out_b);
}

// ── Conv2d forward matches Burn NdArray (ones kernel) ────────────────────────

#[test]
fn conv2d_forward_matches_burn() {
    use burn::nn::conv::Conv2dConfig;
    use burn::nn::PaddingConfig2d;
    use coeus_nn::Conv2d;

    // C_in=2, C_out=1, K=3×3, no bias, ones weight.
    // Input [N=1, C_in=2, H=5, W=5]; valid conv → output [N=1, C_out=1, H=3, W=3].
    let (n, ic, oc, h, w, k) = (1usize, 2, 1, 5, 5, 3);
    let out_h = h - k + 1;
    let out_w = w - k + 1;
    let data: Vec<f32> = (0..n * ic * h * w).map(|x| x as f32 * 0.05 - 1.0).collect();
    let w_vec = vec![1.0f32; oc * ic * k * k];

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, h, w], &data),
        false,
    );
    let mut conv_c = Conv2d::<f32, SequentialBackend>::new(ic, oc, k, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k, k], &w_vec), true);
    let out_c = conv_c.forward(&xv);

    let xb: BurnTensor<BurnBackend, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, h, w]), &dev());
    let mut conv_b = Conv2dConfig::new([ic, oc], [k, k])
        .with_bias(false)
        .with_padding(PaddingConfig2d::Valid)
        .init::<BurnBackend>(&dev());
    conv_b.weight =
        burn::module::Param::from_data(TensorData::new(w_vec.clone(), [oc, ic, k, k]), &dev());
    let out_b = bvec(conv_b.forward(xb));

    assert_eq!(out_c.tensor.shape(), &[n, oc, out_h, out_w]);
    assert_close("conv2d_vs_burn", out_c.tensor.as_slice(), &out_b);
}

// ── Conv3d forward matches Burn NdArray (ones kernel) ────────────────────────

#[test]
fn conv3d_forward_matches_burn() {
    use burn::nn::conv::Conv3dConfig;
    use burn::nn::PaddingConfig3d;
    use coeus_nn::Conv3d;

    // C_in=2, C_out=1, K=2×2×2, no bias, ones weight.
    // Input [N=1, C_in=2, D=3, H=3, W=3]; valid conv → output [N=1, C_out=1, D=2, H=2, W=2].
    let (n, ic, oc, d, h, w, k) = (1usize, 2, 1, 3, 3, 3, 2);
    let out_d = d - k + 1;
    let out_h = h - k + 1;
    let out_w = w - k + 1;
    let data: Vec<f32> = (0..n * ic * d * h * w)
        .map(|x| x as f32 * 0.05 - 1.25)
        .collect();
    let w_vec = vec![1.0f32; oc * ic * k * k * k];

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, d, h, w], &data),
        false,
    );
    let mut conv_c = Conv3d::<f32, SequentialBackend>::new(ic, oc, k, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k, k, k], &w_vec), true);
    let out_c = conv_c.forward(&xv);

    let xb: BurnTensor<BurnBackend, 5> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, d, h, w]), &dev());
    let mut conv_b = Conv3dConfig::new([ic, oc], [k, k, k])
        .with_bias(false)
        .with_padding(PaddingConfig3d::Valid)
        .init::<BurnBackend>(&dev());
    conv_b.weight =
        burn::module::Param::from_data(TensorData::new(w_vec.clone(), [oc, ic, k, k, k]), &dev());
    let out_b = bvec(conv_b.forward(xb));

    assert_eq!(out_c.tensor.shape(), &[n, oc, out_d, out_h, out_w]);
    assert_close("conv3d_vs_burn", out_c.tensor.as_slice(), &out_b);
}

// ── InstanceNorm2d forward matches Burn NdArray ──────────────────────────────

#[test]
fn instancenorm2d_forward_matches_burn() {
    use burn::nn::InstanceNormConfig;
    use coeus_nn::InstanceNorm2d;

    // [N=2, C=2, H=3, W=3] — Burn InstanceNorm uses affine=false by default (weight=1, bias=0).
    // Coeus InstanceNorm2d::new also defaults to no affine.
    // Both normalize over [H, W] = 9 elements independently per (N, C) pair.
    let (n, c, h, w) = (2usize, 2, 3, 3);
    let data: Vec<f32> = (0..n * c * h * w).map(|x| x as f32 * 0.25 - 4.5).collect();

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, h, w], &data),
        false,
    );
    let inorm = InstanceNorm2d::<f32, SequentialBackend>::new(c, 1e-5);
    let out_c = inorm.forward(&xv);

    let in_b = InstanceNormConfig::new(c).init::<BurnBackend>(&dev());
    let xb: BurnTensor<BurnBackend, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, h, w]), &dev());
    let out_b = bvec(in_b.forward(xb));

    // Tolerance: same sqrt(var+eps) vs sqrt(var)+eps gap as GroupNorm; tol=1e-4.
    assert_close_rel(
        "instancenorm2d_vs_burn",
        out_c.tensor.as_slice(),
        &out_b,
        1e-4,
    );
}

// ── RMSProp step (analytical reference) ──────────────────────────────────────
//
// RMSProp update rule (first step, no momentum):
//   v_t = α * v_{t-1} + (1 - α) * g_t²   (v_0 = 0)
//   θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)

#[test]
fn rmsprop_step_matches_analytical_reference() {
    use coeus_optim::traits::Optimizer;
    use coeus_optim::RMSProp;
    let lr = 0.01f64;
    let alpha = 0.9f64;
    let eps = 1e-8f64;
    let params_data = vec![1.0f64, -0.5, 2.0, 0.3];
    let grads_data = vec![0.5f64, 1.0, -0.2, 0.8];

    let p = Var::new(
        CoeusTensor::<f64, SequentialBackend>::from_slice(vec![4], &params_data),
        true,
    );
    if let Some(ref g) = p.grad {
        *g.write() = CoeusTensor::from_slice(vec![4], &grads_data);
    }

    let mut opt = RMSProp::new(vec![p.clone()], lr, alpha, eps);
    opt.step();

    // Closed-form expected for t=1 (v_0 = 0):
    //   v_1 = (1 - α) * g²
    //   θ_1 = θ - lr * g / (√v_1 + ε)
    let expected: Vec<f64> = params_data
        .iter()
        .zip(grads_data.iter())
        .map(|(&theta, &g)| {
            let v1 = (1.0 - alpha) * g * g;
            theta - lr * g / (v1.sqrt() + eps)
        })
        .collect();
    let actual = opt.params[0].tensor.as_slice().to_vec();
    assert_close_rel(
        "rmsprop_step",
        &actual.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &expected.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        1e-6,
    );
}

// ── AdaGrad step (analytical reference) ──────────────────────────────────────
//
// AdaGrad update rule (first step):
//   G_t = G_{t-1} + g_t²   (G_0 = 0)
//   θ_t = θ_{t-1} - lr * g_t / (√G_t + ε)

#[test]
fn adagrad_step_matches_analytical_reference() {
    use coeus_optim::traits::Optimizer;
    use coeus_optim::AdaGrad;
    let lr = 0.1f64;
    let eps = 1e-8f64;
    let params_data = vec![0.5f64, -1.0, 1.5, -0.3];
    let grads_data = vec![0.2f64, 0.8, -0.4, 1.0];

    let p = Var::new(
        CoeusTensor::<f64, SequentialBackend>::from_slice(vec![4], &params_data),
        true,
    );
    if let Some(ref g) = p.grad {
        *g.write() = CoeusTensor::from_slice(vec![4], &grads_data);
    }

    let mut opt = AdaGrad::new(vec![p.clone()], lr, eps);
    opt.step();

    // Closed-form expected for t=1 (G_0 = 0):
    //   G_1 = g²
    //   θ_1 = θ - lr * g / (√G_1 + ε)
    let expected: Vec<f64> = params_data
        .iter()
        .zip(grads_data.iter())
        .map(|(&theta, &g)| {
            let g1 = g * g;
            theta - lr * g / (g1.sqrt() + eps)
        })
        .collect();
    let actual = opt.params[0].tensor.as_slice().to_vec();
    assert_close_rel(
        "adagrad_step",
        &actual.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &expected.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        1e-6,
    );
}

// ── AdamW step (analytical reference) ────────────────────────────────────────
//
// AdamW update (decoupled weight decay, first step t=1):
//   m_t = β1 * 0 + (1 - β1) * g = (1 - β1) * g
//   v_t = β2 * 0 + (1 - β2) * g² = (1 - β2) * g²
//   m̂ = m_t / (1 - β1^1) = m_t / (1 - β1)
//   v̂ = v_t / (1 - β2^1) = v_t / (1 - β2)
//   θ_t = θ_{t-1} - lr * (m̂ / (√v̂ + ε) + wd * θ_{t-1})

#[test]
fn adamw_step_matches_analytical_reference() {
    use coeus_optim::traits::Optimizer;
    use coeus_optim::AdamW;
    let lr = 0.001f64;
    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let eps = 1e-8f64;
    let wd = 0.01f64;
    let params_data = vec![1.0f64, -0.5, 2.0, 0.3];
    let grads_data = vec![0.5f64, 1.0, -0.2, 0.8];

    let p = Var::new(
        CoeusTensor::<f64, SequentialBackend>::from_slice(vec![4], &params_data),
        true,
    );
    if let Some(ref g) = p.grad {
        *g.write() = CoeusTensor::from_slice(vec![4], &grads_data);
    }

    let mut opt = AdamW::new(vec![p.clone()], lr, beta1, beta2, eps, wd);
    opt.step();

    let expected: Vec<f64> = params_data
        .iter()
        .zip(grads_data.iter())
        .map(|(&theta, &g)| {
            let m1 = (1.0 - beta1) * g;
            let v1 = (1.0 - beta2) * g * g;
            let m_hat = m1 / (1.0 - beta1);
            let v_hat = v1 / (1.0 - beta2);
            theta - lr * (m_hat / (v_hat.sqrt() + eps) + wd * theta)
        })
        .collect();
    let actual = opt.params[0].tensor.as_slice().to_vec();
    assert_close_rel(
        "adamw_step",
        &actual.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &expected.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        1e-5,
    );
}

// ── Conv3d with stride and padding matches Burn NdArray ───────────────────────

#[test]
fn conv3d_stride_padding_matches_burn() {
    use burn::nn::conv::Conv3dConfig;
    use burn::nn::PaddingConfig3d;
    use coeus_nn::Conv3d;

    // C_in=1, C_out=1, K=2×2×2, stride=2, padding=1, no bias, ones weight.
    // Input [N=1, C_in=1, D=4, H=4, W=4].
    // Output D_out = (4 + 2*1 - 2) / 2 + 1 = 3.
    let (n, ic, oc, d, h, w, k, stride, pad) = (1usize, 1, 1, 4, 4, 4, 2, 2, 1);
    let out_d = (d + 2 * pad - k) / stride + 1;
    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;
    let data: Vec<f32> = (0..n * ic * d * h * w)
        .map(|x| x as f32 * 0.1 - 2.0)
        .collect();
    let w_vec = vec![1.0f32; oc * ic * k * k * k];

    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, d, h, w], &data),
        false,
    );
    let mut conv_c =
        Conv3d::<f32, SequentialBackend>::with_params(ic, oc, k, stride, pad, 1, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k, k, k], &w_vec), true);
    let out_c = conv_c.forward(&xv);

    let xb: BurnTensor<BurnBackend, 5> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, d, h, w]), &dev());
    let mut conv_b = Conv3dConfig::new([ic, oc], [k, k, k])
        .with_stride([stride, stride, stride])
        .with_bias(false)
        .with_padding(PaddingConfig3d::Explicit(pad, pad, pad))
        .init::<BurnBackend>(&dev());
    conv_b.weight =
        burn::module::Param::from_data(TensorData::new(w_vec.clone(), [oc, ic, k, k, k]), &dev());
    let out_b = bvec(conv_b.forward(xb));

    assert_eq!(out_c.tensor.shape(), &[n, oc, out_d, out_h, out_w]);
    assert_close(
        "conv3d_stride_padding_vs_burn",
        out_c.tensor.as_slice(),
        &out_b,
    );
}

// ── Conv3d backward (dx, dw) matches Burn autodiff ────────────────────────────
//
// Valid (no-padding) convolution so the backward formula is exact with stride=1.
// Weight shape [oc, ic, kd, kh, kw]; no bias.
// Uses `burn::tensor::module::conv3d` free function with tracked input/weight
// tensors (no std-gated GradientsParams needed).

#[test]
fn conv3d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::module::conv3d as burn_conv3d;
    use burn::tensor::ops::ConvOptions;
    use coeus_nn::Conv3d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, ic, oc, d, h, w, k) = (1usize, 2, 1, 5, 5, 5, 3);
    let data: Vec<f32> = (0..n * ic * d * h * w)
        .map(|x| x as f32 * 0.03 - 2.0)
        .collect();
    let w_vec: Vec<f32> = (0..oc * ic * k * k * k)
        .map(|x| (x as f32 + 1.0) * 0.1 - 0.5)
        .collect();

    // Coeus: forward + backward (valid conv, stride=1, pad=0).
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, d, h, w], &data),
        true,
    );
    let mut conv_c = Conv3d::<f32, SequentialBackend>::new(ic, oc, k, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k, k, k], &w_vec), true);
    coeus_autograd::sum(&conv_c.forward(&xv)).backward();
    let dx_c = xv.grad().unwrap();
    let dw_c = conv_c.weight.grad().unwrap();

    // Burn: free-function conv3d with tracked input and weight.
    let xb: BurnTensor<AB, 5> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, d, h, w]), &device)
            .require_grad();
    let wb: BurnTensor<AB, 5> =
        BurnTensor::from_data(TensorData::new(w_vec.clone(), [oc, ic, k, k, k]), &device)
            .require_grad();
    let opts = ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    let grads = burn_conv3d(xb.clone(), wb.clone(), None, opts)
        .sum()
        .backward();

    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    let dw_b: Vec<f32> = wb.grad(&grads).unwrap().into_data().to_vec().unwrap();

    assert_close("conv3d_bwd_dx", dx_c.as_slice(), &dx_b);
    assert_close("conv3d_bwd_dw", dw_c.as_slice(), &dw_b);
}

// ── Transpose backward (gradient routing) matches Burn autodiff ──────────────

#[test]
fn transpose_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::TensorData;
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    // 2×3 matrix transposed → 3×2; gradient of sum flows back unchanged
    // through the transpose (transposing the grad tensor).
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2usize, 3usize];

    // Coeus: forward transpose, backward = transpose of upstream grad.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(shape.to_vec(), &data),
        true,
    );
    let yt = coeus_autograd::transpose(&xv, 0, 1);
    let loss = coeus_autograd::sum(&yt);
    loss.backward();
    let coeus_grad = xv.grad().unwrap();

    // Burn: same computation via autodiff.
    let xb: BurnTensor<AB, 2> =
        BurnTensor::from_data(TensorData::new(data.clone(), shape), &device).require_grad();
    let yt_b = xb.clone().transpose();
    let loss_b = yt_b.sum();
    let grads = loss_b.backward();
    let burn_grad: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();

    // Gradient of sum(transpose(x)) = ones(2,3) — same shape and values as input.
    assert_close("transpose_backward", coeus_grad.as_slice(), &burn_grad);
}

// ── Conv1d backward: dx and dw match Burn autodiff ───────────────────────────
//
// Uses burn::tensor::module::conv1d with tracked weight tensor so both dx and
// dw can be extracted from the same GradientsParams without needing std-gated
// burn::optim.  Non-trivial weights ensure the gradients are non-constant.
//
// Backward equations:
//   dx  = full-correlation(upstream, flipped-weight)
//   dw  = valid-correlation(input, upstream)

#[test]
fn conv1d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::module::conv1d as burn_conv1d;
    use burn::tensor::ops::ConvOptions;
    use coeus_nn::Conv1d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, ic, oc, l, k) = (1usize, 2, 1, 6, 3);
    // Non-trivial inputs and weights — constant values would mask wrong gradients.
    let data: Vec<f32> = (0..n * ic * l).map(|x| x as f32 * 0.1 - 0.5).collect();
    let w_vec: Vec<f32> = (0..oc * ic * k)
        .map(|x| (x as f32 + 1.0) * 0.2 - 0.3)
        .collect();

    // Coeus: forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, l], &data),
        true,
    );
    let mut conv_c = Conv1d::<f32, SequentialBackend>::new(ic, oc, k, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k], &w_vec), true);
    coeus_autograd::sum(&conv_c.forward(&xv)).backward();
    let dx_c = xv.grad().unwrap();
    let dw_c = conv_c.weight.grad().unwrap();

    // Burn: raw tensor conv1d via autodiff so weight is a tracked tensor.
    let xb: BurnTensor<AB, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, l]), &device).require_grad();
    let wb: BurnTensor<AB, 3> =
        BurnTensor::from_data(TensorData::new(w_vec.clone(), [oc, ic, k]), &device).require_grad();
    // stride=1, padding=0 (valid), dilation=1, groups=1.
    let opts = ConvOptions::new([1], [0], [1], 1);
    let grads = burn_conv1d(xb.clone(), wb.clone(), None, opts)
        .sum()
        .backward();

    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    let dw_b: Vec<f32> = wb.grad(&grads).unwrap().into_data().to_vec().unwrap();

    assert_close("conv1d_bwd_dx", dx_c.as_slice(), &dx_b);
    assert_close("conv1d_bwd_dw", dw_c.as_slice(), &dw_b);
}

// ── Conv2d backward: dx and dw match Burn autodiff ───────────────────────────

#[test]
fn conv2d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::tensor::module::conv2d as burn_conv2d;
    use burn::tensor::ops::ConvOptions;
    use coeus_nn::Conv2d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, ic, oc, h, w, k) = (1usize, 2, 1, 5, 5, 3);
    let data: Vec<f32> = (0..n * ic * h * w).map(|x| x as f32 * 0.05 - 1.0).collect();
    let w_vec: Vec<f32> = (0..oc * ic * k * k)
        .map(|x| (x as f32 + 1.0) * 0.1 - 0.45)
        .collect();

    // Coeus backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, ic, h, w], &data),
        true,
    );
    let mut conv_c = Conv2d::<f32, SequentialBackend>::new(ic, oc, k, false);
    conv_c.weight = Var::new(CoeusTensor::from_slice(vec![oc, ic, k, k], &w_vec), true);
    coeus_autograd::sum(&conv_c.forward(&xv)).backward();
    let dx_c = xv.grad().unwrap();
    let dw_c = conv_c.weight.grad().unwrap();

    // Burn: raw tensor conv2d via autodiff so weight is a tracked tensor.
    let xb: BurnTensor<AB, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, ic, h, w]), &device).require_grad();
    let wb: BurnTensor<AB, 4> =
        BurnTensor::from_data(TensorData::new(w_vec.clone(), [oc, ic, k, k]), &device)
            .require_grad();
    let opts = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let grads = burn_conv2d(xb.clone(), wb.clone(), None, opts)
        .sum()
        .backward();

    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    let dw_b: Vec<f32> = wb.grad(&grads).unwrap().into_data().to_vec().unwrap();

    assert_close("conv2d_bwd_dx", dx_c.as_slice(), &dx_b);
    assert_close("conv2d_bwd_dw", dw_c.as_slice(), &dw_b);
}

// ── BatchNorm2d training-mode backward: dx, dw, db match Burn autodiff ────────
//
// Implements the same training-mode formula manually in Burn autodiff tensors:
//   view [N,C,H,W] as [M=N*H*W, C] (NHWC permutation, matching Coeus layout),
//   mean[1,C] over M, var[1,C] (population, /M), x_hat, gamma*x_hat + beta.
// Bessel correction is NOT applied — Coeus uses population variance.

#[test]
fn batchnorm2d_training_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::BatchNorm2d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, c, h, w) = (2usize, 2, 3, 3);
    let m = (n * h * w) as f32; // = 18
    let eps = 1e-5_f32;
    let data: Vec<f32> = (0..n * c * h * w).map(|x| x as f32 * 0.05 - 1.0).collect();
    let gamma = vec![1.2f32, 0.8];
    let beta = vec![0.1f32, -0.1];

    // Coeus: training-mode forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, h, w], &data),
        true,
    );
    let mut bn = BatchNorm2d::<f32, SequentialBackend>::new(c, eps as f64, 0.1);
    bn.weight = Var::new(CoeusTensor::from_slice(vec![c], &gamma), true);
    bn.bias = Var::new(CoeusTensor::from_slice(vec![c], &beta), true);
    coeus_autograd::sum(&bn.forward(&xv)).backward();

    // Burn autodiff: manual BN2d formula matching Coeus NHWC layout.
    // [N,C,H,W] → permute [N,H,W,C] → reshape [M, C].
    let xb: BurnTensor<AB, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, h, w]), &device).require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(gamma.clone(), [c]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(beta.clone(), [c]), &device).require_grad();

    let flat: BurnTensor<AB, 2> = xb
        .clone()
        .permute([0, 2, 3, 1]) // [N,H,W,C]
        .reshape([n * h * w, c]); // [M, C]
    let mean = flat.clone().sum_dim(0) / m; // [1, C]
    let xmu = flat.sub(mean);
    let var = xmu.clone().powf_scalar(2.0).sum_dim(0) / m; // [1, C] population
    let xhat = xmu / (var.add_scalar(eps)).sqrt(); // [M, C]
    let out_b = xhat.mul(wb.clone().reshape([1, c])) + bk.clone().reshape([1, c]);
    let grads = out_b.sum().backward();

    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();
    let to_vec4 = |t: BurnTensor<NdArray<f32>, 4>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "batchnorm2d_bwd_dw",
        bn.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "batchnorm2d_bwd_db",
        bn.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
    // dx comparison: Coeus returns [N,C,H,W]; Burn reference is in [M,C] layout
    // converted back via the same NHWC permutation.
    let dx_b_flat = to_vec4(xb.grad(&grads).unwrap());
    let dx_c_flat = xv.grad().unwrap();
    assert_close_rel("batchnorm2d_bwd_dx", dx_c_flat.as_slice(), &dx_b_flat, 1e-4);
}

// ── InstanceNorm1d backward (dx) matches Burn autodiff ────────────────────────
//
// InstanceNorm normalizes each (sample, channel) slice over the spatial dim L.
// Formula: reshape [N,C,L] → [N*C, L], population variance (÷L), then
//   y[nc, l] = gamma[nc%C] * x_hat[nc, l] + beta[nc%C].
// gamma/beta are tiled [C] → [N*C, 1] via repeat_dim(0, N).

#[test]
fn instancenorm1d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::InstanceNorm1d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, c, l) = (2usize, 2, 4);
    let nc = n * c;
    let eps = 1e-5_f32;
    let data: Vec<f32> = (0..n * c * l).map(|x| x as f32 * 0.1 - 1.0).collect();
    let gamma = vec![1.2f32, 0.8];
    let beta = vec![0.1f32, -0.1];

    // Coeus forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, l], &data),
        true,
    );
    let mut in1 = InstanceNorm1d::<f32, SequentialBackend>::new(c, eps as f64);
    in1.weight = Var::new(CoeusTensor::from_slice(vec![c], &gamma), true);
    in1.bias = Var::new(CoeusTensor::from_slice(vec![c], &beta), true);
    coeus_autograd::sum(&in1.forward(&xv)).backward();

    // Burn autodiff: manual InstanceNorm1d formula.
    // gamma/beta: [C] → repeat N times dim-0 → [N*C] → [N*C, 1] for broadcast.
    let xb: BurnTensor<AB, 3> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, l]), &device).require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(gamma.clone(), [c]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(beta.clone(), [c]), &device).require_grad();

    let flat: BurnTensor<AB, 2> = xb.clone().reshape([nc, l]);
    let mean = flat.clone().sum_dim(1) / (l as f32); // [N*C, 1]
    let xmu = flat.sub(mean);
    let var = xmu.clone().powf_scalar(2.0).sum_dim(1) / (l as f32);
    let xhat = xmu / (var.add_scalar(eps)).sqrt(); // [N*C, L]
    let g2 = wb.clone().reshape([1, c]).repeat_dim(0, n).reshape([nc, 1]);
    let b2 = bk.clone().reshape([1, c]).repeat_dim(0, n).reshape([nc, 1]);
    let grads = (xhat.mul(g2) + b2).sum().backward();

    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "instancenorm1d_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &dx_b,
        1e-4,
    );
    assert_close_rel(
        "instancenorm1d_bwd_dw",
        in1.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "instancenorm1d_bwd_db",
        in1.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── InstanceNorm2d backward (dx) matches Burn autodiff ────────────────────────
//
// Same as InstanceNorm1d but spatial = H*W (normalized over each HxW slice).
// Reshape [N,C,H,W] → [N*C, H*W], same population-variance normalize, then
//   y[nc, hw] = gamma[nc%C] * x_hat[nc, hw] + beta[nc%C].

#[test]
fn instancenorm2d_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::InstanceNorm2d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, c, h, w) = (2usize, 2, 3, 3);
    let nc = n * c;
    let hw = h * w;
    let eps = 1e-5_f32;
    let data: Vec<f32> = (0..n * c * h * w).map(|x| x as f32 * 0.05 - 1.0).collect();
    let gamma = vec![1.2f32, 0.8];
    let beta = vec![0.1f32, -0.1];

    // Coeus forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, h, w], &data),
        true,
    );
    let mut in2 = InstanceNorm2d::<f32, SequentialBackend>::new(c, eps as f64);
    in2.weight = Var::new(CoeusTensor::from_slice(vec![c], &gamma), true);
    in2.bias = Var::new(CoeusTensor::from_slice(vec![c], &beta), true);
    coeus_autograd::sum(&in2.forward(&xv)).backward();

    // Burn autodiff: manual InstanceNorm2d formula.
    // Reshape [N,C,H,W] → [N*C, H*W] for spatial normalization.
    let xb: BurnTensor<AB, 4> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, h, w]), &device).require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(gamma.clone(), [c]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(beta.clone(), [c]), &device).require_grad();

    let flat: BurnTensor<AB, 2> = xb.clone().reshape([nc, hw]);
    let mean = flat.clone().sum_dim(1) / (hw as f32); // [N*C, 1]
    let xmu = flat.sub(mean);
    let var = xmu.clone().powf_scalar(2.0).sum_dim(1) / (hw as f32);
    let xhat = xmu / (var.add_scalar(eps)).sqrt(); // [N*C, H*W]
    let g2 = wb.clone().reshape([1, c]).repeat_dim(0, n).reshape([nc, 1]);
    let b2 = bk.clone().reshape([1, c]).repeat_dim(0, n).reshape([nc, 1]);
    let grads = (xhat.mul(g2) + b2).sum().backward();

    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "instancenorm2d_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &dx_b,
        1e-4,
    );
    assert_close_rel(
        "instancenorm2d_bwd_dw",
        in2.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "instancenorm2d_bwd_db",
        in2.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── InstanceNorm3d forward matches Burn ───────────────────────────────────────
//
// InstanceNorm3d does not exist in Burn's nn module; use a manual reference:
// reshape [N,C,D,H,W] → [N*C, D*H*W], normalize over D*H*W (population var),
// apply per-channel gamma/beta via repeat_dim. Verifies both forward values
// and backward dx/dw/db.

#[test]
fn instancenorm3d_forward_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::InstanceNorm3d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, c, d, h, w) = (1usize, 2, 3, 3, 3);
    let nc = n * c;
    let dhw = d * h * w;
    let eps = 1e-5_f32;
    let data: Vec<f32> = (0..n * c * d * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let gamma = vec![1.2f32, 0.8];
    let beta = vec![0.1f32, -0.1];

    // Coeus forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, d, h, w], &data),
        true,
    );
    let mut in3 = InstanceNorm3d::<f32, SequentialBackend>::new(c, eps as f64);
    in3.weight = Var::new(CoeusTensor::from_slice(vec![c], &gamma), true);
    in3.bias = Var::new(CoeusTensor::from_slice(vec![c], &beta), true);
    let out_c = in3.forward(&xv);
    coeus_autograd::sum(&out_c).backward();

    // Burn autodiff: manual InstanceNorm3d formula.
    // [N,C,D,H,W] → reshape [N*C, D*H*W], normalize over D*H*W per (sample,channel).
    let xb: BurnTensor<AB, 5> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, d, h, w]), &device)
            .require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(gamma.clone(), [c]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(beta.clone(), [c]), &device).require_grad();

    let flat: BurnTensor<AB, 2> = xb.clone().reshape([nc, dhw]);
    let mean = flat.clone().sum_dim(1) / (dhw as f32);
    let xmu = flat.sub(mean);
    let var = xmu.clone().powf_scalar(2.0).sum_dim(1) / (dhw as f32);
    let xhat = xmu / (var.add_scalar(eps)).sqrt();
    let g2 = wb.clone().reshape([1, c]).repeat_dim(0, n).reshape([nc, 1]);
    let b2 = bk.clone().reshape([1, c]).repeat_dim(0, n).reshape([nc, 1]);
    let out_ref = xhat.mul(g2) + b2;

    // Forward value comparison.
    let out_c_flat: Vec<f32> = out_c.tensor.as_slice().to_vec();
    let out_b_flat: Vec<f32> = out_ref.clone().into_data().to_vec().unwrap();
    assert_close_rel("instancenorm3d_fwd", &out_c_flat, &out_b_flat, 1e-4);

    let grads = out_ref.sum().backward();

    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();
    let dx_b: Vec<f32> = xb.grad(&grads).unwrap().into_data().to_vec().unwrap();
    assert_close_rel(
        "instancenorm3d_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &dx_b,
        1e-4,
    );
    assert_close_rel(
        "instancenorm3d_bwd_dw",
        in3.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "instancenorm3d_bwd_db",
        in3.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
}

// ── BatchNorm3d training-mode forward + backward matches Burn autodiff ────────
//
// Input [N=2, C=2, D=2, H=2, W=2] = 32 elements.
// Training mode: batch mean/var computed over the N*D*H*W = 16 samples per channel.
// Burn reference: permute [N,C,D,H,W] → [N,D,H,W,C] → reshape [M=N*D*H*W, C],
//   population-variance BN (÷M), then affine transform.
// Tolerance: standard 1e-4 derived from f32 rounding in reciprocal-sqrt.

#[test]
fn batchnorm3d_training_backward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use coeus_nn::BatchNorm3d;

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (n, c, d, h, w) = (2usize, 2, 2, 2, 2);
    let m = (n * d * h * w) as f32; // 16
    let eps = 1e-5_f32;
    let data: Vec<f32> = (0..n * c * d * h * w)
        .map(|x| x as f32 * 0.05 - 0.75)
        .collect();
    let gamma = vec![1.1f32, 0.9];
    let beta = vec![0.05f32, -0.05];

    // Coeus: training-mode forward + backward.
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![n, c, d, h, w], &data),
        true,
    );
    let mut bn = BatchNorm3d::<f32, SequentialBackend>::new(c, eps as f64, 0.1);
    bn.weight = Var::new(CoeusTensor::from_slice(vec![c], &gamma), true);
    bn.bias = Var::new(CoeusTensor::from_slice(vec![c], &beta), true);
    coeus_autograd::sum(&bn.forward(&xv)).backward();

    // Burn autodiff reference: manual BN3d formula matching Coeus NDHWC layout.
    // Permute [N,C,D,H,W] → [N,D,H,W,C] → reshape [M, C].
    let xb: BurnTensor<AB, 5> =
        BurnTensor::from_data(TensorData::new(data.clone(), [n, c, d, h, w]), &device)
            .require_grad();
    let wb: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(gamma.clone(), [c]), &device).require_grad();
    let bk: BurnTensor<AB, 1> =
        BurnTensor::from_data(TensorData::new(beta.clone(), [c]), &device).require_grad();

    let flat: BurnTensor<AB, 2> = xb
        .clone()
        .permute([0, 2, 3, 4, 1]) // [N,D,H,W,C]
        .reshape([n * d * h * w, c]); // [M, C]
    let mean = flat.clone().sum_dim(0) / m; // [1, C]
    let xmu = flat.sub(mean);
    let var = xmu.clone().powf_scalar(2.0).sum_dim(0) / m; // [1, C] population
    let xhat = xmu / (var.add_scalar(eps)).sqrt(); // [M, C]
    let out_b = xhat.mul(wb.clone().reshape([1, c])) + bk.clone().reshape([1, c]);
    let grads = out_b.sum().backward();

    let to_vec = |t: BurnTensor<NdArray<f32>, 1>| t.into_data().to_vec::<f32>().unwrap();
    let to_vec5 = |t: BurnTensor<NdArray<f32>, 5>| t.into_data().to_vec::<f32>().unwrap();

    assert_close_rel(
        "batchnorm3d_bwd_dw",
        bn.weight.grad().unwrap().as_slice(),
        &to_vec(wb.grad(&grads).unwrap()),
        1e-4,
    );
    assert_close_rel(
        "batchnorm3d_bwd_db",
        bn.bias.grad().unwrap().as_slice(),
        &to_vec(bk.grad(&grads).unwrap()),
        1e-4,
    );
    // dx: Coeus [N,C,D,H,W]; Burn computes in [M,C] → grads flow back to [N,C,D,H,W] via permute
    let dx_b = to_vec5(xb.grad(&grads).unwrap());
    let dx_c = xv.grad().unwrap();
    assert_close_rel("batchnorm3d_bwd_dx", dx_c.as_slice(), &dx_b, 1e-4);
}

// ── TransformerEncoderLayer Pre-LN forward matches Burn components ────────────
//
// Coeus Pre-LN: x₁ = x + Attn(LN1(x)), x₂ = x₁ + FFN(LN2(x₁)).
//
// Burn norm_first=true uses the same Pre-LN order but with reversed norm naming:
//   norm_2 is applied before MHA, norm_1 is applied before FFN.
//
// We assemble individual Burn components (MHA + LN×2 + PWFF) with Coeus weights
// (transposed where Linear convention differs) and manually compose the Pre-LN
// forward to sidestep the private fields on TransformerEncoderLayer.
//
// Configuration: d_model=4, H=2, d_ff=8, batch=1, seq=3, dropout=0, no bias.
// Tolerance: 2e-4 (f32 summation over 4-dimensional inner products).

#[test]
fn transformer_encoder_layer_forward_matches_burn() {
    use burn::nn::{
        attention::{MhaInput, MultiHeadAttentionConfig},
        transformer::PositionWiseFeedForwardConfig,
        LayerNormConfig,
    };
    use burn::{backend::autodiff::Autodiff, module::Param, tensor::TensorData};
    use coeus_autograd::Var;
    use coeus_nn::{Module, NullMask, TransformerEncoderLayer};

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();
    let (batch, seq, d_model, d_ff) = (1usize, 3, 4, 8);

    // Utility: transpose a matrix stored row-major [rows × cols] → [cols × rows].
    let transpose = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        let mut t = vec![0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = w[i * cols + j];
            }
        }
        t
    };

    // ── Coeus layer (H=2, dropout=0, bias=false for MHA to simplify) ──
    let mut layer =
        TransformerEncoderLayer::<f32, SequentialBackend, 2, NullMask>::new(d_model, d_ff, 0.0);
    // Set MHA biases to None (Coeus initializes them to zeros anyway;
    // Burn MHA will have no bias, making both equivalent since 0 + x = x).
    layer.self_attn.b_q = None;
    layer.self_attn.b_k = None;
    layer.self_attn.b_v = None;
    layer.self_attn.b_o = None;

    // Input: [batch=1, seq=3, d_model=4].
    let input_data: Vec<f32> = (0..batch * seq * d_model)
        .map(|i| (i as f32) * 0.05 - 0.3)
        .collect();
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &input_data),
        false,
    );
    let out_c = layer.forward(&xv);

    // Extract Coeus weights as slices for Burn setup.
    let wq = layer.self_attn.w_q.tensor.as_slice().to_vec();
    let wk = layer.self_attn.w_k.tensor.as_slice().to_vec();
    let wv = layer.self_attn.w_v.tensor.as_slice().to_vec();
    let wo = layer.self_attn.w_o.tensor.as_slice().to_vec();
    let gamma1 = layer.norm1.weight.tensor.as_slice().to_vec();
    let beta1 = layer.norm1.bias.tensor.as_slice().to_vec();
    let gamma2 = layer.norm2.weight.tensor.as_slice().to_vec();
    let beta2 = layer.norm2.bias.tensor.as_slice().to_vec();
    let wff1 = layer.ffn.linear1.weight.tensor.as_slice().to_vec(); // [d_ff, d_model]
    let wff2 = layer.ffn.linear2.weight.tensor.as_slice().to_vec(); // [d_model, d_ff]

    // ── Burn components: manually compose Pre-LN encoder forward ──
    // Coeus stores W [out, in], computes x @ W^T.
    // Burn Linear stores W [in, out], computes x @ W.
    // So burn_weight = transpose(coeus_weight).

    let mut mha_b = MultiHeadAttentionConfig::new(d_model, 2)
        .with_dropout(0.0)
        .with_quiet_softmax(false)
        .init::<AB>(&device);
    mha_b.query.weight = Param::from_data(
        TensorData::new(transpose(&wq, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.key.weight = Param::from_data(
        TensorData::new(transpose(&wk, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.value.weight = Param::from_data(
        TensorData::new(transpose(&wv, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.output.weight = Param::from_data(
        TensorData::new(transpose(&wo, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.query.bias = None;
    mha_b.key.bias = None;
    mha_b.value.bias = None;
    mha_b.output.bias = None;

    // Burn norm_mha ← Coeus norm1 (MHA pre-norm).
    let mut norm_mha_b = LayerNormConfig::new(d_model).init::<AB>(&device);
    norm_mha_b.gamma = Param::from_data(TensorData::new(gamma1.clone(), [d_model]), &device);
    norm_mha_b.beta = Param::from_data(TensorData::new(beta1.clone(), [d_model]), &device);

    // Burn norm_ffn ← Coeus norm2 (FFN pre-norm).
    let mut norm_ffn_b = LayerNormConfig::new(d_model).init::<AB>(&device);
    norm_ffn_b.gamma = Param::from_data(TensorData::new(gamma2.clone(), [d_model]), &device);
    norm_ffn_b.beta = Param::from_data(TensorData::new(beta2.clone(), [d_model]), &device);

    // Burn PWFF: linear_inner [d_model, d_ff] ← transpose(coeus_wff1 [d_ff, d_model]).
    //            linear_outer [d_ff, d_model]  ← transpose(coeus_wff2 [d_model, d_ff]).
    let mut pwff_b = PositionWiseFeedForwardConfig::new(d_model, d_ff)
        .with_dropout(0.0)
        .init::<AB>(&device);
    pwff_b.linear_inner.weight = Param::from_data(
        TensorData::new(transpose(&wff1, d_ff, d_model), [d_model, d_ff]),
        &device,
    );
    pwff_b.linear_outer.weight = Param::from_data(
        TensorData::new(transpose(&wff2, d_model, d_ff), [d_ff, d_model]),
        &device,
    );
    // Coeus FFN biases are zeros; set Burn biases to zero vectors.
    let zeros_d = vec![0f32; d_model];
    let zeros_ff = vec![0f32; d_ff];
    let bff1 = layer
        .ffn
        .linear1
        .bias
        .as_ref()
        .map(|b| b.tensor.as_slice().to_vec());
    let bff2 = layer
        .ffn
        .linear2
        .bias
        .as_ref()
        .map(|b| b.tensor.as_slice().to_vec());
    pwff_b.linear_inner.bias = Some(Param::from_data(
        TensorData::new(bff1.unwrap_or(zeros_ff.clone()), [d_ff]),
        &device,
    ));
    pwff_b.linear_outer.bias = Some(Param::from_data(
        TensorData::new(bff2.unwrap_or(zeros_d.clone()), [d_model]),
        &device,
    ));

    // ── Manual Pre-LN forward (mirrors Burn norm_first=true behaviour) ──
    //   x₁ = x + MHA(norm_mha(x))
    //   out = x₁ + FFN(norm_ffn(x₁))
    let xb: BurnTensor<AB, 3> = BurnTensor::from_data(
        TensorData::new(input_data.clone(), [batch, seq, d_model]),
        &device,
    );
    let normed_mha = norm_mha_b.forward(xb.clone());
    let attn_out = mha_b.forward(MhaInput::self_attn(normed_mha)).context;
    let x_b = xb.clone() + attn_out;
    let normed_ffn = norm_ffn_b.forward(x_b.clone());
    let ffn_out = pwff_b.forward(normed_ffn);
    let out_b = x_b + ffn_out;

    assert_close_rel(
        "encoder_layer_fwd",
        out_c.tensor.as_slice(),
        &out_b.into_data().to_vec::<f32>().unwrap(),
        2e-4,
    );
}

// ── TransformerEncoderLayer Pre-LN backward (dx) matches Burn ────────────────
//
// Same setup as the forward test, but wraps both paths with autodiff and
// compares the input gradient dx after loss = sum(output).
//
// Only dx is asserted here because weight gradients span multiple separately-
// assembled Burn components that lack a flat gather API; dx is sufficient to
// verify that the backward graph through LN→MHA→residual→LN→FFN→residual
// is correctly wired in Coeus.

#[test]
fn transformer_encoder_layer_backward_matches_burn() {
    use burn::nn::{
        attention::{MhaInput, MultiHeadAttentionConfig},
        transformer::PositionWiseFeedForwardConfig,
        LayerNormConfig,
    };
    use burn::{backend::autodiff::Autodiff, module::Param, tensor::TensorData};
    use coeus_nn::{Module, NullMask, TransformerEncoderLayer};

    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();
    let (batch, seq, d_model, d_ff) = (1usize, 3, 4, 8);

    let transpose = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        let mut t = vec![0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = w[i * cols + j];
            }
        }
        t
    };

    // Coeus layer — biases zeroed / removed for simplicity.
    let mut layer =
        TransformerEncoderLayer::<f32, SequentialBackend, 2, NullMask>::new(d_model, d_ff, 0.0);
    layer.self_attn.b_q = None;
    layer.self_attn.b_k = None;
    layer.self_attn.b_v = None;
    layer.self_attn.b_o = None;

    let input_data: Vec<f32> = (0..batch * seq * d_model)
        .map(|i| (i as f32) * 0.05 - 0.3)
        .collect();
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &input_data),
        true,
    );
    let out_c = layer.forward(&xv);
    coeus_autograd::sum(&out_c).backward();

    // Extract Coeus weights.
    let wq = layer.self_attn.w_q.tensor.as_slice().to_vec();
    let wk = layer.self_attn.w_k.tensor.as_slice().to_vec();
    let wv = layer.self_attn.w_v.tensor.as_slice().to_vec();
    let wo = layer.self_attn.w_o.tensor.as_slice().to_vec();
    let gamma1 = layer.norm1.weight.tensor.as_slice().to_vec();
    let beta1 = layer.norm1.bias.tensor.as_slice().to_vec();
    let gamma2 = layer.norm2.weight.tensor.as_slice().to_vec();
    let beta2 = layer.norm2.bias.tensor.as_slice().to_vec();
    let wff1 = layer.ffn.linear1.weight.tensor.as_slice().to_vec();
    let wff2 = layer.ffn.linear2.weight.tensor.as_slice().to_vec();
    let bff1 = layer
        .ffn
        .linear1
        .bias
        .as_ref()
        .map(|b| b.tensor.as_slice().to_vec())
        .unwrap_or_else(|| vec![0f32; d_ff]);
    let bff2 = layer
        .ffn
        .linear2
        .bias
        .as_ref()
        .map(|b| b.tensor.as_slice().to_vec())
        .unwrap_or_else(|| vec![0f32; d_model]);

    // Burn components with matching weights.
    let mut mha_b = MultiHeadAttentionConfig::new(d_model, 2)
        .with_dropout(0.0)
        .with_quiet_softmax(false)
        .init::<AB>(&device);
    mha_b.query.weight = Param::from_data(
        TensorData::new(transpose(&wq, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.key.weight = Param::from_data(
        TensorData::new(transpose(&wk, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.value.weight = Param::from_data(
        TensorData::new(transpose(&wv, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.output.weight = Param::from_data(
        TensorData::new(transpose(&wo, d_model, d_model), [d_model, d_model]),
        &device,
    );
    mha_b.query.bias = None;
    mha_b.key.bias = None;
    mha_b.value.bias = None;
    mha_b.output.bias = None;

    let mut norm_mha_b = LayerNormConfig::new(d_model).init::<AB>(&device);
    norm_mha_b.gamma = Param::from_data(TensorData::new(gamma1, [d_model]), &device);
    norm_mha_b.beta = Param::from_data(TensorData::new(beta1, [d_model]), &device);

    let mut norm_ffn_b = LayerNormConfig::new(d_model).init::<AB>(&device);
    norm_ffn_b.gamma = Param::from_data(TensorData::new(gamma2, [d_model]), &device);
    norm_ffn_b.beta = Param::from_data(TensorData::new(beta2, [d_model]), &device);

    let mut pwff_b = PositionWiseFeedForwardConfig::new(d_model, d_ff)
        .with_dropout(0.0)
        .init::<AB>(&device);
    pwff_b.linear_inner.weight = Param::from_data(
        TensorData::new(transpose(&wff1, d_ff, d_model), [d_model, d_ff]),
        &device,
    );
    pwff_b.linear_outer.weight = Param::from_data(
        TensorData::new(transpose(&wff2, d_model, d_ff), [d_ff, d_model]),
        &device,
    );
    pwff_b.linear_inner.bias = Some(Param::from_data(TensorData::new(bff1, [d_ff]), &device));
    pwff_b.linear_outer.bias = Some(Param::from_data(TensorData::new(bff2, [d_model]), &device));

    // Burn Pre-LN forward with autodiff.
    let xb: BurnTensor<AB, 3> =
        BurnTensor::from_data(TensorData::new(input_data, [batch, seq, d_model]), &device)
            .require_grad();

    let normed_mha = norm_mha_b.forward(xb.clone());
    let attn_out = mha_b.forward(MhaInput::self_attn(normed_mha)).context;
    let x_b = xb.clone() + attn_out;
    let normed_ffn = norm_ffn_b.forward(x_b.clone());
    let ffn_out = pwff_b.forward(normed_ffn);
    let out_b = x_b + ffn_out;
    let grads = out_b.sum().backward();

    // Compare dx.
    assert_close_rel(
        "encoder_bwd_dx",
        xv.grad().unwrap().as_slice(),
        &xb.grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        2e-4,
    );
}

// ── TransformerEncoder N-layer self-consistency ───────────────────────────────
//
// Evidence tier: empirical self-consistency (structural).
// Invariant: `TransformerEncoder<H=2,N=2>::forward(x)` equals manually chaining
// the two `TransformerEncoderLayer` forwards on the same backend.
// Per-layer Burn differential correctness already established by
// `transformer_encoder_layer_forward_matches_burn`.

#[test]
fn transformer_encoder_stack_2layer_self_consistent() {
    use coeus_autograd::NullMask;
    use coeus_nn::{transformer::encoder::TransformerEncoder, Module};

    let (batch, seq, d_model, d_ff) = (1usize, 3, 4, 8);
    let enc = TransformerEncoder::<f32, SequentialBackend, 2, 2, NullMask>::new(d_model, d_ff, 0.0);

    let data: Vec<f32> = (0..batch * seq * d_model)
        .map(|i| i as f32 * 0.1 - 0.3)
        .collect();
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &data),
        false,
    );

    let out_stack = enc.forward(&x);

    let mid = enc.layers[0].forward(&x);
    let out_manual = enc.layers[1].forward(&mid);

    assert_close_rel(
        "encoder_stack_self_consistency",
        out_stack.tensor.as_slice(),
        out_manual.tensor.as_slice(),
        f32::EPSILON * 32.0,
    );
}

// ── TransformerEncoder 2-layer forward vs Burn ────────────────────────────────
//
// Evidence tier: differential (Burn autodiff NdArray reference).
// Two Burn Pre-LN encoder layers assembled from Coeus weights, chained
// sequentially, produce the same output as the Coeus 2-layer stack.

#[test]
fn transformer_encoder_stack_2layer_forward_matches_burn() {
    use burn::backend::autodiff::Autodiff;
    use burn::nn::{
        attention::{MhaInput, MultiHeadAttentionConfig},
        transformer::PositionWiseFeedForwardConfig,
        LayerNormConfig,
    };
    use burn::{module::Param, tensor::TensorData};
    use coeus_autograd::NullMask;
    use coeus_nn::{transformer::encoder::TransformerEncoder, Module};
    type AB = Autodiff<NdArray<f32>>;
    let device: NdArrayDevice = Default::default();

    let (batch, seq, d_model, d_ff, heads) = (1usize, 3, 4, 8, 2usize);

    // Coeus 2-layer stack with fixed (random) weights.
    let mut enc =
        TransformerEncoder::<f32, SequentialBackend, 2, 2, NullMask>::new(d_model, d_ff, 0.0);
    // Drop biases from MHA so Burn (which defaults to bias=None) is equivalent.
    for layer in enc.layers.iter_mut() {
        layer.self_attn.b_q = None;
        layer.self_attn.b_k = None;
        layer.self_attn.b_v = None;
        layer.self_attn.b_o = None;
    }

    let data: Vec<f32> = (0..batch * seq * d_model)
        .map(|i| i as f32 * 0.05 - 0.3)
        .collect();
    let xv = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![batch, seq, d_model], &data),
        false,
    );
    // Coeus 2-layer output.
    let out_coeus = enc.forward(&xv);

    // Utility: Coeus [out,in] row-major → Burn [in,out] row-major.
    let transpose = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        let mut t = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = w[r * cols + c];
            }
        }
        t
    };

    let xb: BurnTensor<AB, 3> = BurnTensor::from_data(
        TensorData::new(data.clone(), [batch, seq, d_model]),
        &device,
    );

    // Chain two Burn Pre-LN layers using Coeus weights.
    let mut x_b = xb;
    for layer in enc.layers.iter() {
        let wq = transpose(layer.self_attn.w_q.tensor.as_slice(), d_model, d_model);
        let wk = transpose(layer.self_attn.w_k.tensor.as_slice(), d_model, d_model);
        let wv = transpose(layer.self_attn.w_v.tensor.as_slice(), d_model, d_model);
        let wo = transpose(layer.self_attn.w_o.tensor.as_slice(), d_model, d_model);

        let mut mha_b = MultiHeadAttentionConfig::new(d_model, heads)
            .with_dropout(0.0)
            .init::<AB>(&device);
        mha_b.query.weight = Param::from_data(TensorData::new(wq, [d_model, d_model]), &device);
        mha_b.query.bias = None;
        mha_b.key.weight = Param::from_data(TensorData::new(wk, [d_model, d_model]), &device);
        mha_b.key.bias = None;
        mha_b.value.weight = Param::from_data(TensorData::new(wv, [d_model, d_model]), &device);
        mha_b.value.bias = None;
        mha_b.output.weight = Param::from_data(TensorData::new(wo, [d_model, d_model]), &device);
        mha_b.output.bias = None;

        let mut norm1_b = LayerNormConfig::new(d_model).init::<AB>(&device);
        norm1_b.gamma = Param::from_data(
            TensorData::new(layer.norm1.weight.tensor.as_slice().to_vec(), [d_model]),
            &device,
        );
        norm1_b.beta = Param::from_data(
            TensorData::new(layer.norm1.bias.tensor.as_slice().to_vec(), [d_model]),
            &device,
        );

        let mut norm2_b = LayerNormConfig::new(d_model).init::<AB>(&device);
        norm2_b.gamma = Param::from_data(
            TensorData::new(layer.norm2.weight.tensor.as_slice().to_vec(), [d_model]),
            &device,
        );
        norm2_b.beta = Param::from_data(
            TensorData::new(layer.norm2.bias.tensor.as_slice().to_vec(), [d_model]),
            &device,
        );

        // linear1: Coeus [d_ff, d_model] → Burn linear_inner [d_model, d_ff]
        let w1 = transpose(layer.ffn.linear1.weight.tensor.as_slice(), d_ff, d_model);
        // linear2: Coeus [d_model, d_ff] → Burn linear_outer [d_ff, d_model]
        let w2 = transpose(layer.ffn.linear2.weight.tensor.as_slice(), d_model, d_ff);
        let b1 = layer
            .ffn
            .linear1
            .bias
            .as_ref()
            .map(|v| v.tensor.as_slice().to_vec())
            .unwrap_or_else(|| vec![0.0f32; d_ff]);
        let b2 = layer
            .ffn
            .linear2
            .bias
            .as_ref()
            .map(|v| v.tensor.as_slice().to_vec())
            .unwrap_or_else(|| vec![0.0f32; d_model]);

        let mut pwff_b = PositionWiseFeedForwardConfig::new(d_model, d_ff)
            .with_dropout(0.0)
            .init::<AB>(&device);
        pwff_b.linear_inner.weight =
            Param::from_data(TensorData::new(w1, [d_model, d_ff]), &device);
        pwff_b.linear_inner.bias = Some(Param::from_data(TensorData::new(b1, [d_ff]), &device));
        pwff_b.linear_outer.weight =
            Param::from_data(TensorData::new(w2, [d_ff, d_model]), &device);
        pwff_b.linear_outer.bias = Some(Param::from_data(TensorData::new(b2, [d_model]), &device));

        // Pre-LN: x1 = x + Attn(LN1(x)),  out = x1 + FFN(LN2(x1))
        let normed_mha = norm1_b.forward(x_b.clone());
        let attn_out = mha_b.forward(MhaInput::self_attn(normed_mha)).context;
        let x1_b = x_b + attn_out;
        let normed_ffn = norm2_b.forward(x1_b.clone());
        let ffn_out = pwff_b.forward(normed_ffn);
        x_b = x1_b + ffn_out;
    }

    let out_b: Vec<f32> = x_b.into_data().to_vec::<f32>().unwrap();
    assert_close_rel(
        "encoder_stack_2layer_fwd",
        out_coeus.tensor.as_slice(),
        &out_b,
        2e-4,
    );
}

// ── TransformerDecoder structural tests ─────────────────────────────────────

#[test]
fn transformer_decoder_layer_forward_is_deterministic() {
    use coeus_autograd::CausalMask;
    use coeus_nn::{NullMask, TransformerDecoderLayer};
    const H: usize = 2;
    let d_model = 4;
    let d_ff = 8;
    let dec = TransformerDecoderLayer::<f32, SequentialBackend, H, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    );
    let tgt_data: Vec<f32> = (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect();
    let mem_data: Vec<f32> = (0..12).map(|i| 0.05 * (i as f32)).collect();
    let tgt = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 3, d_model], &tgt_data),
        false,
    );
    let memory = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 3, d_model], &mem_data),
        false,
    );
    let out1 = dec.forward_decoder(&tgt, &memory);
    let out2 = dec.forward_decoder(&tgt, &memory);
    let v1 = out1.tensor.as_slice().to_vec();
    let v2 = out2.tensor.as_slice().to_vec();
    assert_close_rel("dec_layer_deterministic", &v1, &v2, f32::EPSILON);
}

#[test]
fn transformer_decoder_stack_2layer_self_consistent() {
    use coeus_autograd::CausalMask;
    use coeus_nn::{NullMask, TransformerDecoder};
    const H: usize = 2;
    const N: usize = 2;
    let d_model = 4;
    let d_ff = 8;
    let dec = TransformerDecoder::<f32, SequentialBackend, H, N, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    );
    let tgt_data: Vec<f32> = (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect();
    let mem_data: Vec<f32> = (0..12).map(|i| 0.05 * (i as f32)).collect();
    let tgt = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 3, d_model], &tgt_data),
        false,
    );
    let memory = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 3, d_model], &mem_data),
        false,
    );

    // Stack forward
    let stack_out = dec.forward_decoder(&tgt, &memory);

    // Manual layer chaining
    let mid = dec.layers[0].forward_decoder(&tgt, &memory);
    let manual_out = dec.layers[1].forward_decoder(&mid, &memory);

    let v_stack = stack_out.tensor.as_slice().to_vec();
    let v_manual = manual_out.tensor.as_slice().to_vec();
    assert_close_rel(
        "decoder_stack_2layer_self_consistent",
        &v_stack,
        &v_manual,
        f32::EPSILON * 32.0,
    );
}

#[test]
fn transformer_decoder_forward_uses_self_as_memory() {
    // dec.forward(x) is defined as dec.forward_decoder(x, x).
    use coeus_autograd::CausalMask;
    use coeus_nn::{Module, NullMask, TransformerDecoderLayer};
    const H: usize = 2;
    let d_model = 4;
    let d_ff = 8;
    let dec = TransformerDecoderLayer::<f32, SequentialBackend, H, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    );
    let data: Vec<f32> = (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect();
    let x = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1, 3, d_model], &data),
        false,
    );
    let v_fwd = dec.forward(&x).tensor.as_slice().to_vec();
    let v_cross = dec.forward_decoder(&x, &x).tensor.as_slice().to_vec();
    assert_close_rel(
        "decoder_forward_vs_forward_decoder_self",
        &v_fwd,
        &v_cross,
        f32::EPSILON * 2.0,
    );
}
