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

use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
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
    use coeus_ops::BackendOps;
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
    use coeus_ops::BackendOps;
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

// Pooling.

#[test]
fn test_cuda_parity_max_pool2d() {
    use coeus_ops::BackendOps;
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
    use coeus_ops::BackendOps;
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

// Optimizer step (AdamW).

#[test]
fn test_cuda_parity_adamw_step() {
    use coeus_ops::BackendOps;
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
