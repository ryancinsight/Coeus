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
