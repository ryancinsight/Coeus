// CudaBackend vs CPU parity differential tests.
//
// Each test runs the same operation on both `CudaBackend` and
// `SequentialBackend` (the verified CPU reference) with identical inputs and
// asserts element-wise output agreement within a derived tolerance. This
// mirrors the WgpuBackend parity audit (`coeus-wgpu/tests/wgpu/parity.rs`) so
// both GPU backends are held to the same CPU reference contract.
//
// The CUDA backend routes each op family to an on-device kernel where one is
// implemented and to a CPU fallback otherwise; either way the observable result
// must match the CPU reference. Every test is skipped unless a live CUDA
// device, driver, and context are all present.

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::{ConvOps, OptimizerOps, PoolOps};
use coeus_tensor::Tensor;

/// Element-wise tolerance for direct (non-accumulating) ops. f32 transcendental
/// kernels (`expf`/`tanhf`/...) agree with the CPU libm path well within this.
const CUDA_TOL: f32 = 1e-4;
/// Tolerance for accumulating ops (matmul / conv). Reduction order differs
/// between the tiled GPU kernel and the sequential CPU triple loop, so the
/// bound is the f32 rounding growth over the contraction dimension.
const CUDA_ACC_TOL: f32 = 1e-3;

/// Acquire CPU + CUDA backends, or `None` when no usable CUDA device/context is
/// available (test is then skipped rather than failed).
fn backends() -> Option<(SequentialBackend, CudaBackend)> {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return None;
    }
    let cuda_b = CudaBackend::new();
    if coeus_cuda::CudaDriver::get().is_none() || coeus_cuda::get_cuda_context().is_none() {
        return None;
    }
    Some((SequentialBackend::new(), cuda_b))
}

fn to_gpu(
    t: &Tensor<f32, SequentialBackend>,
    s: &SequentialBackend,
    c: &CudaBackend,
) -> Tensor<f32, CudaBackend> {
    t.to_backend_on(s, c)
}

fn to_cpu(
    t: &Tensor<f32, CudaBackend>,
    c: &CudaBackend,
    s: &SequentialBackend,
) -> Tensor<f32, SequentialBackend> {
    t.to_backend_on(c, s)
}

fn assert_parity_tol(label: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e} tol={tol:.0e}"
        );
    }
}

// Elementwise binary.

macro_rules! binary_parity {
    ($name:ident, $op:expr, $a:expr, $b:expr) => {
        #[test]
        fn $name() {
            let Some((s, c)) = backends() else {
                return;
            };
            let a = Tensor::from_slice(vec![4, 4], &$a);
            let b = Tensor::from_slice(vec![4, 4], &$b);
            let cpu = $op(&a, &b, &s);
            let gpu = to_cpu(&$op(&to_gpu(&a, &s, &c), &to_gpu(&b, &s, &c), &c), &c, &s);
            assert_parity_tol(stringify!($name), cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
        }
    };
}

binary_parity!(
    test_cuda_parity_add,
    coeus_ops::add,
    (0..16).map(|x| x as f32).collect::<Vec<_>>(),
    (0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>()
);
binary_parity!(
    test_cuda_parity_sub,
    coeus_ops::sub,
    (0..16).map(|x| x as f32).collect::<Vec<_>>(),
    (0..16).map(|x| x as f32 * 0.5).collect::<Vec<_>>()
);
binary_parity!(
    test_cuda_parity_mul,
    coeus_ops::mul,
    (0..16).map(|x| x as f32 * 0.1 + 0.5).collect::<Vec<_>>(),
    (0..16).map(|x| x as f32 * 0.2 - 1.0).collect::<Vec<_>>()
);
binary_parity!(
    test_cuda_parity_div,
    coeus_ops::div,
    (0..16).map(|x| (x as f32 + 1.0) * 0.5).collect::<Vec<_>>(),
    (0..16).map(|x| (x as f32 + 1.0) * 0.25).collect::<Vec<_>>()
);

// Unary activations.

macro_rules! unary_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let Some((s, c)) = backends() else {
                return;
            };
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data);
            let cpu = $op(&x, &s);
            let gpu = to_cpu(&$op(&to_gpu(&x, &s, &c), &c), &c, &s);
            assert_parity_tol(stringify!($name), cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
        }
    };
}

unary_parity!(
    test_cuda_parity_relu,
    coeus_ops::relu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
unary_parity!(
    test_cuda_parity_sigmoid,
    coeus_ops::sigmoid,
    vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 0.5, 2.0]
);
unary_parity!(
    test_cuda_parity_tanh,
    coeus_ops::tanh,
    vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.5, 1.5]
);
unary_parity!(
    test_cuda_parity_gelu,
    coeus_ops::gelu,
    // Include |x| ~ 2.3, where the tanh-approximation of GELU diverges most
    // from the exact-erf contract, so this test genuinely guards the contract.
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
unary_parity!(
    test_cuda_parity_silu,
    coeus_ops::silu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
unary_parity!(
    test_cuda_parity_mish,
    coeus_ops::mish,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
unary_parity!(
    test_cuda_parity_exp,
    coeus_ops::exp,
    vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, 2.0]
);
unary_parity!(
    test_cuda_parity_log,
    coeus_ops::log,
    vec![0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 0.25, 16.0]
);
unary_parity!(
    test_cuda_parity_sqrt,
    coeus_ops::sqrt,
    vec![0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 0.5, 25.0]
);
unary_parity!(
    test_cuda_parity_neg,
    coeus_ops::neg,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
unary_parity!(
    test_cuda_parity_abs,
    coeus_ops::abs,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
unary_parity!(
    test_cuda_parity_cos,
    coeus_ops::cos,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);
unary_parity!(
    test_cuda_parity_sin,
    coeus_ops::sin,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);

// Unary activation gradients (the `*Grad` kernel variants). These exercise the
// device kernels driven by autograd backward, including the exact-erf
// `GeluGrad` kernel, against the CPU `eval_unary` reference.

macro_rules! unary_grad_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let Some((s, c)) = backends() else {
                return;
            };
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data);
            let cpu = coeus_ops::elementwise_unary(&x, &s, $op);
            let gpu = to_cpu(
                &coeus_ops::elementwise_unary(&to_gpu(&x, &s, &c), &c, $op),
                &c,
                &s,
            );
            assert_parity_tol(stringify!($name), cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
        }
    };
}

unary_grad_parity!(
    test_cuda_parity_relu_grad,
    coeus_ops::UnaryOp::ReluGrad,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
unary_grad_parity!(
    test_cuda_parity_sigmoid_grad,
    coeus_ops::UnaryOp::SigmoidGrad,
    vec![0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.3]
);
unary_grad_parity!(
    test_cuda_parity_tanh_grad,
    coeus_ops::UnaryOp::TanhGrad,
    vec![-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.3]
);
unary_grad_parity!(
    test_cuda_parity_gelu_grad,
    coeus_ops::UnaryOp::GeluGrad,
    // Span the region where the tanh-approx gradient diverges from exact erf,
    // so this guards the exact-erf GeluGrad kernel contract.
    vec![-3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0]
);
unary_grad_parity!(
    test_cuda_parity_silu_grad,
    coeus_ops::UnaryOp::SiluGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);
unary_grad_parity!(
    test_cuda_parity_mish_grad,
    coeus_ops::UnaryOp::MishGrad,
    vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 1.5]
);

// Reductions.

#[test]
fn test_cuda_parity_sum_axis0() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s);
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x, &s, &c), 0, &c), &c, &s);
    assert_parity_tol("sum_axis0", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_sum_axis1() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x, &s, &c), 1, &c), &c, &s);
    assert_parity_tol("sum_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_mean_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::mean_axis(&to_gpu(&x, &s, &c), 1, &c), &c, &s);
    assert_parity_tol("mean_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_max_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::max_axis(&to_gpu(&x, &s, &c), 1, &c), &c, &s);
    assert_parity_tol("max_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_min_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s);
    let gpu = to_cpu(&coeus_ops::min_axis(&to_gpu(&x, &s, &c), 0, &c), &c, &s);
    assert_parity_tol("min_axis0", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

// Matmul.

#[test]
fn test_cuda_parity_matmul_2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (m, k, n) = (16, 20, 12);
    let a: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.02 - 0.5).collect();
    let at = Tensor::from_slice(vec![m, k], &a);
    let bt = Tensor::from_slice(vec![k, n], &b);
    let cpu = coeus_ops::matmul(&at, &bt, &s);
    let gpu = to_cpu(
        &coeus_ops::matmul(&to_gpu(&at, &s, &c), &to_gpu(&bt, &s, &c), &c),
        &c,
        &s,
    );
    assert_parity_tol("matmul_2d", cpu.as_slice(), gpu.as_slice(), CUDA_ACC_TOL);
}

#[test]
fn test_cuda_parity_batched_matmul() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (b_sz, m, k, n) = (3, 8, 10, 6);
    let a: Vec<f32> = (0..b_sz * m * k).map(|x| x as f32 * 0.01).collect();
    let b: Vec<f32> = (0..b_sz * k * n).map(|x| x as f32 * 0.02 - 0.3).collect();
    let at = Tensor::from_slice(vec![b_sz, m, k], &a);
    let bt = Tensor::from_slice(vec![b_sz, k, n], &b);
    let cpu = coeus_ops::matmul(&at, &bt, &s);
    let gpu = to_cpu(
        &coeus_ops::matmul(&to_gpu(&at, &s, &c), &to_gpu(&bt, &s, &c), &c),
        &c,
        &s,
    );
    assert_parity_tol(
        "batched_matmul",
        cpu.as_slice(),
        gpu.as_slice(),
        CUDA_ACC_TOL,
    );
}

// Convolutions.

#[test]
fn test_cuda_parity_conv1d_forward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (batch, in_c, len, out_c, ksize) = (2, 3, 8, 4, 3);
    let input: Vec<f32> = (0..batch * in_c * len)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * ksize)
        .map(|x| x as f32 * 0.1 - 1.8)
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|x| x as f32 * 0.2 - 0.3).collect();

    let in_t = Tensor::from_slice(vec![batch, in_c, len], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, ksize], &weight);
    let b_t = Tensor::from_slice(vec![out_c], &bias);
    let out_len = len - ksize + 1;

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![batch, out_c, out_len]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.conv1d(
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(b_t.storage()),
        1,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    );

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![batch, out_c, out_len], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.conv1d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        1,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    );

    assert_parity_tol(
        "conv1d_fwd",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv2d_forward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (batch, in_c, h, ww, out_c, kh, kw) = (2, 2, 5, 5, 3, 3, 3);
    let input: Vec<f32> = (0..batch * in_c * h * ww)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|x| x as f32 * 0.2 - 0.1).collect();

    let in_t = Tensor::from_slice(vec![batch, in_c, h, ww], &input);
    let wt = Tensor::from_slice(vec![out_c, in_c, kh, kw], &weight);
    let bt = Tensor::from_slice(vec![out_c], &bias);
    let oh = h - kh + 1;
    let ow = ww - kw + 1;

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![batch, out_c, oh, ow]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.conv2d(
        in_t.storage(),
        in_t.layout(),
        wt.storage(),
        wt.layout(),
        Some(bt.storage()),
        1,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    );

    let in_g = to_gpu(&in_t, &s, &c);
    let wg = to_gpu(&wt, &s, &c);
    let bg = to_gpu(&bt, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![batch, out_c, oh, ow], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.conv2d(
        in_g.storage(),
        in_g.layout(),
        wg.storage(),
        wg.layout(),
        Some(bg.storage()),
        1,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    );

    assert_parity_tol(
        "conv2d_fwd",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (n, in_c, h, w, out_c, kh, kw) = (2, 2, 5, 5, 3, 3, 3);
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input: Vec<f32> = (0..n * in_c * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let grad_out: Vec<f32> = (0..n * out_c * oh * ow)
        .map(|x| x as f32 * 0.03 - 0.4)
        .collect();

    let in_t = Tensor::from_slice(vec![n, in_c, h, w], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, kh, kw], &weight);
    let go_t = Tensor::from_slice(vec![n, out_c, oh, ow], &grad_out);

    // CPU reference gradients.
    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![n, in_c, h, w]);
    let mut gw_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c, in_c, kh, kw]);
    let mut gb_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c]);
    let gi_l = gi_c.layout().clone();
    let gw_l = gw_c.layout().clone();
    s.conv2d_backward(
        go_t.storage(),
        go_t.layout(),
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(gi_c.storage_mut()),
        &gi_l,
        Some(gw_c.storage_mut()),
        &gw_l,
        Some(gb_c.storage_mut()),
        1,
        0,
        1,
    );

    // CUDA gradients.
    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let go_g = to_gpu(&go_t, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, in_c, h, w], &c);
    let mut gw_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c, in_c, kh, kw], &c);
    let mut gb_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c], &c);
    c.conv2d_backward(
        go_g.storage(),
        go_g.layout(),
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(gi_g.storage_mut()),
        &gi_l,
        Some(gw_g.storage_mut()),
        &gw_l,
        Some(gb_g.storage_mut()),
        1,
        0,
        1,
    );

    assert_parity_tol(
        "conv2d_bwd_grad_input",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv2d_bwd_grad_weight",
        gw_c.as_slice(),
        to_cpu(&gw_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv2d_bwd_grad_bias",
        gb_c.as_slice(),
        to_cpu(&gb_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv3d_forward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (n, in_c, d, h, w, out_c, kd, kh, kw) = (2, 2, 4, 4, 4, 3, 2, 2, 2);
    let od = d - kd + 1;
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input: Vec<f32> = (0..n * in_c * d * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kd * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|x| x as f32 * 0.2 - 0.3).collect();

    let in_t = Tensor::from_slice(vec![n, in_c, d, h, w], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, kd, kh, kw], &weight);
    let b_t = Tensor::from_slice(vec![out_c], &bias);

    let mut out_s = Tensor::<f32, SequentialBackend>::zeros(vec![n, out_c, od, oh, ow]);
    let out_l = out_s.layout().clone();
    s.conv3d(
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(b_t.storage()),
        1,
        0,
        1,
        out_s.storage_mut(),
        &out_l,
    );

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut out_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, out_c, od, oh, ow], &c);
    c.conv3d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        1,
        0,
        1,
        out_g.storage_mut(),
        &out_l,
    );

    assert_parity_tol(
        "conv3d_fwd",
        out_s.as_slice(),
        to_cpu(&out_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv3d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    let (n, in_c, d, h, w, out_c, kd, kh, kw) = (2, 2, 4, 4, 4, 3, 2, 2, 2);
    let od = d - kd + 1;
    let oh = h - kh + 1;
    let ow = w - kw + 1;

    let input: Vec<f32> = (0..n * in_c * d * h * w)
        .map(|x| x as f32 * 0.05 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..out_c * in_c * kd * kh * kw)
        .map(|x| x as f32 * 0.1 - 1.5)
        .collect();
    let grad_out: Vec<f32> = (0..n * out_c * od * oh * ow)
        .map(|x| x as f32 * 0.03 - 0.4)
        .collect();

    let in_t = Tensor::from_slice(vec![n, in_c, d, h, w], &input);
    let w_t = Tensor::from_slice(vec![out_c, in_c, kd, kh, kw], &weight);
    let go_t = Tensor::from_slice(vec![n, out_c, od, oh, ow], &grad_out);

    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![n, in_c, d, h, w]);
    let mut gw_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c, in_c, kd, kh, kw]);
    let mut gb_c = Tensor::<f32, SequentialBackend>::zeros(vec![out_c]);
    let gi_l = gi_c.layout().clone();
    let gw_l = gw_c.layout().clone();
    s.conv3d_backward(
        go_t.storage(),
        go_t.layout(),
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(gi_c.storage_mut()),
        &gi_l,
        Some(gw_c.storage_mut()),
        &gw_l,
        Some(gb_c.storage_mut()),
        1,
        0,
        1,
    );

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let go_g = to_gpu(&go_t, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, in_c, d, h, w], &c);
    let mut gw_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c, in_c, kd, kh, kw], &c);
    let mut gb_g = Tensor::<f32, CudaBackend>::zeros_on(vec![out_c], &c);
    c.conv3d_backward(
        go_g.storage(),
        go_g.layout(),
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(gi_g.storage_mut()),
        &gi_l,
        Some(gw_g.storage_mut()),
        &gw_l,
        Some(gb_g.storage_mut()),
        1,
        0,
        1,
    );

    assert_parity_tol(
        "conv3d_bwd_grad_input",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv3d_bwd_grad_weight",
        gw_c.as_slice(),
        to_cpu(&gw_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv3d_bwd_grad_bias",
        gb_c.as_slice(),
        to_cpu(&gb_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

// Pooling.

#[test]
fn test_cuda_parity_max_pool2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data: Vec<f32> = (0..2 * 2 * 4 * 4).map(|x| x as f32 * 0.1).collect();
    let x = Tensor::from_slice(vec![2, 2, 4, 4], &data);

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 2, 2]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.max_pool2d(
        x.storage(),
        x.layout(),
        2,
        2,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    );

    let xg = to_gpu(&x, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 2, 2], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.max_pool2d(
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    );

    assert_parity_tol(
        "max_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_avg_pool2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data: Vec<f32> = (0..2 * 2 * 4 * 4).map(|x| x as f32 * 0.1).collect();
    let x = Tensor::from_slice(vec![2, 2, 4, 4], &data);

    let mut cpu_out = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 2, 2]);
    let cpu_out_layout = cpu_out.layout().clone();
    s.avg_pool2d(
        x.storage(),
        x.layout(),
        2,
        2,
        0,
        1,
        cpu_out.storage_mut(),
        &cpu_out_layout,
    );

    let xg = to_gpu(&x, &s, &c);
    let mut gpu_out = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 2, 2], &c);
    let gpu_out_layout = gpu_out.layout().clone();
    c.avg_pool2d(
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    );

    assert_parity_tol(
        "avg_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_max_pool2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    // Non-monotonic data so the argmax routing of the gradient is exercised.
    let data: Vec<f32> = (0..2 * 2 * 4 * 4)
        .map(|i| ((i * 7 + 3) % 13) as f32)
        .collect();
    let x = Tensor::from_slice(vec![2, 2, 4, 4], &data);
    let grad_out: Vec<f32> = (0..2 * 2 * 2 * 2).map(|i| i as f32 * 0.5 + 1.0).collect();
    let go = Tensor::from_slice(vec![2, 2, 2, 2], &grad_out);

    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 4, 4]);
    let gi_l = gi_c.layout().clone();
    s.max_pool2d_backward(
        go.storage(),
        go.layout(),
        x.storage(),
        x.layout(),
        2,
        2,
        0,
        1,
        gi_c.storage_mut(),
        &gi_l,
    );

    let xg = to_gpu(&x, &s, &c);
    let gog = to_gpu(&go, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 4, 4], &c);
    c.max_pool2d_backward(
        gog.storage(),
        gog.layout(),
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gi_g.storage_mut(),
        &gi_l,
    );

    assert_parity_tol(
        "max_pool2d_bwd",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_avg_pool2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };
    // avg-pool backward distributes grad_out uniformly over each window and
    // needs no input values, so only grad_out is supplied.
    let grad_out: Vec<f32> = (0..2 * 2 * 2 * 2).map(|i| i as f32 * 0.5 + 1.0).collect();
    let go = Tensor::from_slice(vec![2, 2, 2, 2], &grad_out);

    let mut gi_c = Tensor::<f32, SequentialBackend>::zeros(vec![2, 2, 4, 4]);
    let gi_l = gi_c.layout().clone();
    s.avg_pool2d_backward(
        go.storage(),
        go.layout(),
        2,
        2,
        0,
        1,
        gi_c.storage_mut(),
        &gi_l,
    );

    let gog = to_gpu(&go, &s, &c);
    let mut gi_g = Tensor::<f32, CudaBackend>::zeros_on(vec![2, 2, 4, 4], &c);
    c.avg_pool2d_backward(
        gog.storage(),
        gog.layout(),
        2,
        2,
        0,
        1,
        gi_g.storage_mut(),
        &gi_l,
    );

    assert_parity_tol(
        "avg_pool2d_bwd",
        gi_c.as_slice(),
        to_cpu(&gi_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

// Optimizer step (AdamW).

#[test]
fn test_cuda_parity_adamw_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let m1_init: Vec<f32> = vec![0.0; n];
    let m2_init: Vec<f32> = vec![0.0; n];

    let p_c = Tensor::from_slice(vec![n], &param);
    let g_c = Tensor::from_slice(vec![n], &grad);
    let mut m1_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m1_init);
    let mut m2_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m2_init);
    let mut p_c_mut = p_c.clone();
    let p_c_layout = p_c_mut.layout().clone();
    let g_c_layout = g_c.layout().clone();
    let m1_c_layout = m1_c.layout().clone();
    let m2_c_layout = m2_c.layout().clone();
    s.adamw_step(
        p_c_mut.storage_mut(),
        &p_c_layout,
        g_c.storage(),
        &g_c_layout,
        m1_c.storage_mut(),
        &m1_c_layout,
        m2_c.storage_mut(),
        &m2_c_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );

    let p_g = to_gpu(&p_c, &s, &c);
    let g_g = to_gpu(&g_c, &s, &c);
    let mut m1_g = Tensor::from_slice_on(vec![n], &m1_init, &c);
    let mut m2_g = Tensor::from_slice_on(vec![n], &m2_init, &c);
    let mut p_g_mut = p_g.clone();
    let p_g_layout = p_g_mut.layout().clone();
    let g_g_layout = g_g.layout().clone();
    let m1_g_layout = m1_g.layout().clone();
    let m2_g_layout = m2_g.layout().clone();
    c.adamw_step(
        p_g_mut.storage_mut(),
        &p_g_layout,
        g_g.storage(),
        &g_g_layout,
        m1_g.storage_mut(),
        &m1_g_layout,
        m2_g.storage_mut(),
        &m2_g_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );

    assert_parity_tol(
        "adamw_step",
        p_c_mut.as_slice(),
        to_cpu(&p_g_mut, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

// Full round-trip: CPU to GPU to CPU identity.

#[test]
fn test_cuda_parity_roundtrip_identity() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data: Vec<f32> = (0..100).map(|x| x as f32 * 0.123 - 6.15).collect();
    let x = Tensor::<f32, SequentialBackend>::from_slice(vec![10, 10], &data);
    let back = to_gpu(&x, &s, &c).to_backend_on(&c, &s);
    assert_parity_tol("roundtrip", x.as_slice(), back.as_slice(), CUDA_TOL);
}

// ── Fused optimizer step parity (sgd / adam / rmsprop / adagrad) ──
//
// adamw is covered by `test_cuda_parity_adamw_step` above. These cover the
// remaining four on-device optimizer kernels, checking both the updated
// parameter and the optimizer state against the CPU reference.

#[test]
fn test_cuda_parity_sgd_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let vel: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, momentum) = (0.05f32, 0.9f32);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut vel_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &vel);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let vl = vel_c.layout().clone();
    s.sgd_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        vel_c.storage_mut(),
        &vl,
        lr,
        momentum,
    );

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut vel_g = Tensor::from_slice_on(vec![n], &vel, &c);
    c.sgd_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        vel_g.storage_mut(),
        &vl,
        lr,
        momentum,
    );

    assert_parity_tol(
        "sgd_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "sgd_velocity",
        vel_c.as_slice(),
        to_cpu(&vel_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_adam_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let m_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.001).collect();
    let v_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, beta1, beta2, eps, t) = (0.05f32, 0.9f32, 0.99f32, 1e-6f32, 3usize);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut m_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &m_init);
    let mut v_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &v_init);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let ml = m_c.layout().clone();
    let vl = v_c.layout().clone();
    s.adam_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        m_c.storage_mut(),
        &ml,
        v_c.storage_mut(),
        &vl,
        lr,
        beta1,
        beta2,
        eps,
        t,
    );

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut m_g = Tensor::from_slice_on(vec![n], &m_init, &c);
    let mut v_g = Tensor::from_slice_on(vec![n], &v_init, &c);
    c.adam_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        m_g.storage_mut(),
        &ml,
        v_g.storage_mut(),
        &vl,
        lr,
        beta1,
        beta2,
        eps,
        t,
    );

    assert_parity_tol(
        "adam_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "adam_m",
        m_c.as_slice(),
        to_cpu(&m_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "adam_v",
        v_c.as_slice(),
        to_cpu(&v_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_rmsprop_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let v_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, alpha, eps) = (0.05f32, 0.99f32, 1e-6f32);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut v_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &v_init);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let vl = v_c.layout().clone();
    s.rmsprop_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        v_c.storage_mut(),
        &vl,
        lr,
        alpha,
        eps,
    );

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut v_g = Tensor::from_slice_on(vec![n], &v_init, &c);
    c.rmsprop_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        v_g.storage_mut(),
        &vl,
        lr,
        alpha,
        eps,
    );

    assert_parity_tol(
        "rmsprop_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "rmsprop_v",
        v_c.as_slice(),
        to_cpu(&v_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

#[test]
fn test_cuda_parity_adagrad_step() {
    let Some((s, c)) = backends() else {
        return;
    };
    let n = 16;
    let param: Vec<f32> = (0..n).map(|x| x as f32 * 0.01).collect();
    let grad: Vec<f32> = (0..n).map(|x| -(x as f32 * 0.05 - 0.4)).collect();
    let h_init: Vec<f32> = (0..n).map(|x| x as f32 * 0.002).collect();
    let (lr, eps) = (0.05f32, 1e-6f32);

    let g_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &grad);
    let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &param);
    let mut h_c = Tensor::<f32, SequentialBackend>::from_slice(vec![n], &h_init);
    let pl = p_c.layout().clone();
    let gl = g_c.layout().clone();
    let hl = h_c.layout().clone();
    s.adagrad_step(
        p_c.storage_mut(),
        &pl,
        g_c.storage(),
        &gl,
        h_c.storage_mut(),
        &hl,
        lr,
        eps,
    );

    let g_g = to_gpu(&g_c, &s, &c);
    let mut p_g = Tensor::from_slice_on(vec![n], &param, &c);
    let mut h_g = Tensor::from_slice_on(vec![n], &h_init, &c);
    c.adagrad_step(
        p_g.storage_mut(),
        &pl,
        g_g.storage(),
        &gl,
        h_g.storage_mut(),
        &hl,
        lr,
        eps,
    );

    assert_parity_tol(
        "adagrad_p",
        p_c.as_slice(),
        to_cpu(&p_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "adagrad_history",
        h_c.as_slice(),
        to_cpu(&h_g, &c, &s).as_slice(),
        CUDA_TOL,
    );
}

// ── Transposed convolution forward parity (on-device gather vs CPU scatter) ──

#[test]
fn test_cuda_parity_conv_transpose1d() {
    let Some((s, c)) = backends() else {
        return;
    };
    // input [n, c_in, l], weight [c_in, c_out, k]
    let (n, c_in, l, c_out, k) = (2, 3, 5, 4, 3);
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 0usize, 1usize);
    let l_out = (l - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;

    let input: Vec<f32> = (0..n * c_in * l).map(|x| x as f32 * 0.05 - 0.7).collect();
    let weight: Vec<f32> = (0..c_in * c_out * k)
        .map(|x| x as f32 * 0.1 - 0.6)
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|x| x as f32 * 0.25 - 0.3).collect();

    let in_t = Tensor::from_slice(vec![n, c_in, l], &input);
    let w_t = Tensor::from_slice(vec![c_in, c_out, k], &weight);
    let b_t = Tensor::from_slice(vec![c_out], &bias);

    let mut out_s = Tensor::<f32, SequentialBackend>::zeros(vec![n, c_out, l_out]);
    let out_l = out_s.layout().clone();
    s.conv_transpose1d(
        in_t.storage(),
        in_t.layout(),
        w_t.storage(),
        w_t.layout(),
        Some(b_t.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_s.storage_mut(),
        &out_l,
    );

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&w_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut out_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, c_out, l_out], &c);
    c.conv_transpose1d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_g.storage_mut(),
        &out_l,
    );

    assert_parity_tol(
        "conv_transpose1d",
        out_s.as_slice(),
        to_cpu(&out_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv_transpose2d() {
    let Some((s, c)) = backends() else {
        return;
    };
    // input [n, c_in, h, w], weight [c_in, c_out, kh, kw]
    let (n, c_in, h, w, c_out, kh, kw) = (2, 2, 4, 4, 3, 3, 3);
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 1usize, 1usize);
    let h_out = (h - 1) * stride - 2 * padding + dilation * (kh - 1) + output_padding + 1;
    let w_out = (w - 1) * stride - 2 * padding + dilation * (kw - 1) + output_padding + 1;

    let input: Vec<f32> = (0..n * c_in * h * w)
        .map(|x| x as f32 * 0.03 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..c_in * c_out * kh * kw)
        .map(|x| x as f32 * 0.07 - 0.4)
        .collect();
    let bias: Vec<f32> = (0..c_out).map(|x| x as f32 * 0.2 - 0.2).collect();

    let in_t = Tensor::from_slice(vec![n, c_in, h, w], &input);
    let wt_t = Tensor::from_slice(vec![c_in, c_out, kh, kw], &weight);
    let b_t = Tensor::from_slice(vec![c_out], &bias);

    let mut out_s = Tensor::<f32, SequentialBackend>::zeros(vec![n, c_out, h_out, w_out]);
    let out_l = out_s.layout().clone();
    s.conv_transpose2d(
        in_t.storage(),
        in_t.layout(),
        wt_t.storage(),
        wt_t.layout(),
        Some(b_t.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_s.storage_mut(),
        &out_l,
    );

    let in_g = to_gpu(&in_t, &s, &c);
    let w_g = to_gpu(&wt_t, &s, &c);
    let b_g = to_gpu(&b_t, &s, &c);
    let mut out_g = Tensor::<f32, CudaBackend>::zeros_on(vec![n, c_out, h_out, w_out], &c);
    c.conv_transpose2d(
        in_g.storage(),
        in_g.layout(),
        w_g.storage(),
        w_g.layout(),
        Some(b_g.storage()),
        stride,
        padding,
        output_padding,
        dilation,
        out_g.storage_mut(),
        &out_l,
    );

    assert_parity_tol(
        "conv_transpose2d",
        out_s.as_slice(),
        to_cpu(&out_g, &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv_transpose1d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };

    let input = [0.5f32, -0.25, 0.75];
    let weight = [0.7f32, -0.4];
    let seed = [1.0f32, -0.5, 0.25, 2.0];

    let input_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 3], &input),
        true,
    );
    let weight_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 2], &weight),
        true,
    );
    let out_cpu =
        coeus_ops::conv_transpose1d(&input_cpu.tensor, &weight_cpu.tensor, None, 1, 0, 0, 1, &s);
    let tracked_cpu =
        coeus_autograd::conv_transpose1d(&input_cpu, &weight_cpu, &None, out_cpu, 1, 0, 0, 1);
    tracked_cpu.backward_with_seed(Tensor::<f32, SequentialBackend>::from_slice(
        [1, 1, 4],
        &seed,
    ));

    let input_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 3], &input, &c),
        true,
    );
    let weight_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 2], &weight, &c),
        true,
    );
    let out_gpu =
        coeus_ops::conv_transpose1d(&input_gpu.tensor, &weight_gpu.tensor, None, 1, 0, 0, 1, &c);
    let tracked_gpu =
        coeus_autograd::conv_transpose1d(&input_gpu, &weight_gpu, &None, out_gpu, 1, 0, 0, 1);
    tracked_gpu.backward_with_seed(Tensor::<f32, CudaBackend>::from_slice_on(
        [1, 1, 4],
        &seed,
        &c,
    ));

    assert_parity_tol(
        "conv_transpose1d_backward_input",
        input_cpu.grad().unwrap().as_slice(),
        to_cpu(&input_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv_transpose1d_backward_weight",
        weight_cpu.grad().unwrap().as_slice(),
        to_cpu(&weight_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}

#[test]
fn test_cuda_parity_conv_transpose2d_backward() {
    let Some((s, c)) = backends() else {
        return;
    };

    let input = [0.5f32, -0.25, 0.75, 1.25];
    let weight = [0.6f32, -0.2, 0.3, -0.5];
    let seed: Vec<f32> = (0..9).map(|x| x as f32 * 0.2 - 0.7).collect();

    let input_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 2, 2], &input),
        true,
    );
    let weight_cpu = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice([1, 1, 2, 2], &weight),
        true,
    );
    let out_cpu =
        coeus_ops::conv_transpose2d(&input_cpu.tensor, &weight_cpu.tensor, None, 1, 0, 0, 1, &s);
    let tracked_cpu =
        coeus_autograd::conv_transpose2d(&input_cpu, &weight_cpu, &None, out_cpu, 1, 0, 0, 1);
    tracked_cpu.backward_with_seed(Tensor::<f32, SequentialBackend>::from_slice(
        [1, 1, 3, 3],
        &seed,
    ));

    let input_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 2, 2], &input, &c),
        true,
    );
    let weight_gpu = Var::new(
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 2, 2], &weight, &c),
        true,
    );
    let out_gpu =
        coeus_ops::conv_transpose2d(&input_gpu.tensor, &weight_gpu.tensor, None, 1, 0, 0, 1, &c);
    let tracked_gpu =
        coeus_autograd::conv_transpose2d(&input_gpu, &weight_gpu, &None, out_gpu, 1, 0, 0, 1);
    tracked_gpu.backward_with_seed(Tensor::<f32, CudaBackend>::from_slice_on(
        [1, 1, 3, 3],
        &seed,
        &c,
    ));

    assert_parity_tol(
        "conv_transpose2d_backward_input",
        input_cpu.grad().unwrap().as_slice(),
        to_cpu(&input_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
    assert_parity_tol(
        "conv_transpose2d_backward_weight",
        weight_cpu.grad().unwrap().as_slice(),
        to_cpu(&weight_gpu.grad().unwrap(), &c, &s).as_slice(),
        CUDA_ACC_TOL,
    );
}
