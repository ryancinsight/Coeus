// ── WgpuBackend vs CPU parity differential tests ──
//
// Each test runs the same operation on both WgpuBackend and SequentialBackend
// (the CPU reference) with identical inputs and asserts element-wise output
// agreement within `WGPU_TOL`.  This verifies that every shader kernel
// matches the verified CPU path for all supported op families.

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

const WGPU_TOL: f32 = 1e-4;

fn seq() -> SequentialBackend {
    SequentialBackend::new()
}
fn wgpu() -> WgpuBackend {
    WgpuBackend::new()
}

/// Transfer a CPU tensor to the WgpuBackend.
fn to_gpu(t: &Tensor<f32, SequentialBackend>) -> Tensor<f32, WgpuBackend> {
    t.to_backend_on(&seq(), &wgpu())
}
/// Transfer a WgpuBackend tensor back to CPU.
fn to_cpu(t: &Tensor<f32, WgpuBackend>) -> Tensor<f32, SequentialBackend> {
    t.to_backend_on(&wgpu(), &seq())
}

fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < WGPU_TOL,
            "{label}[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e}"
        );
    }
}

// ── Elementwise binary ───────────────────────────────────────────────────

#[test]
fn test_wgpu_parity_add() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>());
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5 - 4.0).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::add(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::add(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("add", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sub() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>());
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.5).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::sub(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::sub(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("sub", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mul() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.1 + 0.5).collect::<Vec<_>>(),
    );
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| x as f32 * 0.2 - 1.0).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::mul(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::mul(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("mul", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_div() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| (x as f32 + 1.0) * 0.5).collect::<Vec<_>>(),
    );
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0..16).map(|x| (x as f32 + 1.0) * 0.25).collect::<Vec<_>>(),
    );
    let cpu = coeus_ops::div(&a, &b, &s);
    let gpu = to_cpu(&coeus_ops::div(&to_gpu(&a), &to_gpu(&b), &wgpu()));
    assert_parity("div", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_aliasing_unary_neg_matches_cpu() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
    let data = vec![-4.0f32, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let x_cpu = Tensor::from_slice(vec![data.len()], &data);
    let x_gpu = to_gpu(&x_cpu);

    // Clone shares storage; output aliases input and must use non-hephaestus fallback.
    let mut out_gpu = x_gpu.clone();
    let out_layout = out_gpu.layout().clone();
    w.elementwise_unary(
        coeus_ops::UnaryOp::Neg,
        x_gpu.storage(),
        x_gpu.layout(),
        out_gpu.storage_mut(),
        &out_layout,
    );

    let expected = coeus_ops::neg(&x_cpu, &s);
    let got = to_cpu(&out_gpu);
    assert_parity("aliasing_unary_neg", expected.as_slice(), got.as_slice());
}

#[test]
fn test_wgpu_aliasing_binary_add_matches_cpu() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
    let a_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0..16).map(|x| x as f32 * 0.1 + 0.5).collect();

    let a_cpu = Tensor::from_slice(vec![4, 4], &a_data);
    let b_cpu = Tensor::from_slice(vec![4, 4], &b_data);
    let a_gpu = to_gpu(&a_cpu);
    let b_gpu = to_gpu(&b_cpu);

    // Clone shares storage; output aliases left input and must use non-hephaestus fallback.
    let mut out_gpu = a_gpu.clone();
    let out_layout = out_gpu.layout().clone();
    w.elementwise_binary(
        coeus_ops::BinaryOp::Add,
        a_gpu.storage(),
        a_gpu.layout(),
        b_gpu.storage(),
        b_gpu.layout(),
        out_gpu.storage_mut(),
        &out_layout,
    );

    let expected = coeus_ops::add(&a_cpu, &b_cpu, &s);
    let got = to_cpu(&out_gpu);
    assert_parity("aliasing_binary_add", expected.as_slice(), got.as_slice());
}

// ── Unary activations ────────────────────────────────────────────────────

macro_rules! test_unary_parity {
    ($name:ident, $op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let s = seq();
            let w = wgpu();
            let data: Vec<f32> = $data;
            let x = Tensor::from_slice(vec![data.len()], &data);
            let cpu = $op(&x, &s);
            let gpu = to_cpu(&$op(&to_gpu(&x), &w));
            assert_parity(stringify!($name), cpu.as_slice(), gpu.as_slice());
        }
    };
}

test_unary_parity!(
    test_wgpu_parity_relu,
    coeus_ops::relu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_sigmoid,
    coeus_ops::sigmoid,
    vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 0.5, 2.0]
);
test_unary_parity!(
    test_wgpu_parity_tanh,
    coeus_ops::tanh,
    vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_gelu,
    coeus_ops::gelu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_silu,
    coeus_ops::silu,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_mish,
    coeus_ops::mish,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_softplus,
    coeus_ops::softplus,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5]
);
test_unary_parity!(
    test_wgpu_parity_exp,
    coeus_ops::exp,
    vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, 2.0]
);
test_unary_parity!(
    test_wgpu_parity_log,
    coeus_ops::log,
    vec![0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 0.25, 16.0]
);
test_unary_parity!(
    test_wgpu_parity_sqrt,
    coeus_ops::sqrt,
    vec![0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 0.5, 25.0]
);
test_unary_parity!(
    test_wgpu_parity_neg,
    coeus_ops::neg,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
test_unary_parity!(
    test_wgpu_parity_abs,
    coeus_ops::abs,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -3.0]
);
test_unary_parity!(
    test_wgpu_parity_cos,
    coeus_ops::cos,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);
test_unary_parity!(
    test_wgpu_parity_sin,
    coeus_ops::sin,
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, core::f32::consts::PI]
);
test_unary_parity!(
    test_wgpu_parity_recip,
    coeus_ops::recip,
    vec![-4.0, -2.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0]
);
test_unary_parity!(
    test_wgpu_parity_sign,
    coeus_ops::sign,
    vec![-4.0, -0.25, 0.0, 0.25, 1.0, -1.0, 3.0, -3.0]
);
test_unary_parity!(
    test_wgpu_parity_floor,
    coeus_ops::floor,
    vec![-2.7, -1.2, -0.1, 0.0, 0.1, 1.2, 2.7, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_ceil,
    coeus_ops::ceil,
    vec![-2.7, -1.2, -0.1, 0.0, 0.1, 1.2, 2.7, 3.0]
);
test_unary_parity!(
    test_wgpu_parity_round,
    coeus_ops::round,
    vec![-2.7, -1.6, -1.2, -0.1, 0.1, 1.2, 1.6, 2.7]
);
test_unary_parity!(
    test_wgpu_parity_trunc,
    coeus_ops::trunc,
    vec![-2.7, -1.5, -1.2, -0.1, 0.1, 1.2, 1.5, 2.7]
);

// ── Reductions ────────────────────────────────────────────────────────────

#[test]
fn test_wgpu_parity_sum_axis0() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s);
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 0, &wgpu()));
    assert_parity("sum_axis0", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sum_axis1() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 1, &wgpu()));
    assert_parity("sum_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mean_axis() {
    let s = seq();
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::mean_axis(&to_gpu(&x), 1, &wgpu()));
    assert_parity("mean_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_max_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s);
    let gpu = to_cpu(&coeus_ops::max_axis(&to_gpu(&x), 1, &wgpu()));
    assert_parity("max_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_min_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s);
    let gpu = to_cpu(&coeus_ops::min_axis(&to_gpu(&x), 0, &wgpu()));
    assert_parity("min_axis0", cpu.as_slice(), gpu.as_slice());
}

// ── Matmul ────────────────────────────────────────────────────────────────

#[test]
fn test_wgpu_parity_matmul_2d() {
    let s = seq();
    let m = 16;
    let k = 20;
    let n = 12;
    let a: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.02 - 0.5).collect();

    let at = Tensor::from_slice(vec![m, k], &a);
    let bt = Tensor::from_slice(vec![k, n], &b);
    let cpu = coeus_ops::matmul(&at, &bt, &s);
    let gpu = to_cpu(&coeus_ops::matmul(&to_gpu(&at), &to_gpu(&bt), &wgpu()));

    let cs = cpu.as_slice();
    let gs = gpu.as_slice();
    assert_eq!(cs.len(), gs.len(), "matmul_2d: length");
    for (i, (&c, &g)) in cs.iter().zip(gs.iter()).enumerate() {
        let diff = (c - g).abs();
        // Accumulated f32 matmul: use 1e-3 tolerance
        assert!(
            diff < 1e-3,
            "matmul_2d[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e}"
        );
    }
}

#[test]
fn test_wgpu_parity_batched_matmul() {
    let s = seq();
    let (b_sz, m, k, n) = (3, 8, 10, 6);
    let a: Vec<f32> = (0..b_sz * m * k).map(|x| x as f32 * 0.01).collect();
    let b: Vec<f32> = (0..b_sz * k * n).map(|x| x as f32 * 0.02 - 0.3).collect();

    let at = Tensor::from_slice(vec![b_sz, m, k], &a);
    let bt = Tensor::from_slice(vec![b_sz, k, n], &b);
    let cpu = coeus_ops::matmul(&at, &bt, &s);
    let gpu = to_cpu(&coeus_ops::matmul(&to_gpu(&at), &to_gpu(&bt), &wgpu()));

    let cs = cpu.as_slice();
    let gs = gpu.as_slice();
    assert_eq!(cs.len(), gs.len(), "batched_matmul: length");
    for (i, (&c, &g)) in cs.iter().zip(gs.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < 1e-3,
            "batched_matmul[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e}"
        );
    }
}

// ── Convolutions ──────────────────────────────────────────────────────────

#[test]
fn test_wgpu_parity_conv1d_forward() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
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

    let in_g = to_gpu(&in_t);
    let w_g = to_gpu(&w_t);
    let b_g = to_gpu(&b_t);
    let mut gpu_out = Tensor::<f32, WgpuBackend>::zeros_on(vec![batch, out_c, out_len], &w);
    let gpu_out_layout = gpu_out.layout().clone();
    w.conv1d(
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

    let gpu_cpu = to_cpu(&gpu_out);
    let cs = cpu_out.as_slice();
    let gs = gpu_cpu.as_slice();
    assert_eq!(cs.len(), gs.len(), "conv1d_fwd: length");
    for (i, (&c, &g)) in cs.iter().zip(gs.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < 1e-3,
            "conv1d_fwd[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e}"
        );
    }
}

#[test]
fn test_wgpu_parity_conv2d_forward() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
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

    let in_g = to_gpu(&in_t);
    let wg = to_gpu(&wt);
    let bg = to_gpu(&bt);
    let mut gpu_out = Tensor::<f32, WgpuBackend>::zeros_on(vec![batch, out_c, oh, ow], &w);
    let gpu_out_layout = gpu_out.layout().clone();
    w.conv2d(
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

    let gpu_cpu = to_cpu(&gpu_out);
    let cs = cpu_out.as_slice();
    let gs = gpu_cpu.as_slice();
    assert_eq!(cs.len(), gs.len(), "conv2d_fwd: length");
    for (i, (&c, &g)) in cs.iter().zip(gs.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < 1e-3,
            "conv2d_fwd[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e}"
        );
    }
}

// ── Pooling ───────────────────────────────────────────────────────────────

#[test]
fn test_wgpu_parity_max_pool2d() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
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

    let xg = to_gpu(&x);
    let mut gpu_out = Tensor::<f32, WgpuBackend>::zeros_on(vec![2, 2, 2, 2], &w);
    let gpu_out_layout = gpu_out.layout().clone();
    w.max_pool2d(
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    );

    assert_parity(
        "max_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out).as_slice(),
    );
}

#[test]
fn test_wgpu_parity_avg_pool2d() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
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

    let xg = to_gpu(&x);
    let mut gpu_out = Tensor::<f32, WgpuBackend>::zeros_on(vec![2, 2, 2, 2], &w);
    let gpu_out_layout = gpu_out.layout().clone();
    w.avg_pool2d(
        xg.storage(),
        xg.layout(),
        2,
        2,
        0,
        1,
        gpu_out.storage_mut(),
        &gpu_out_layout,
    );

    assert_parity(
        "avg_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out).as_slice(),
    );
}

// ── Optimizer step (AdamW) ────────────────────────────────────────────────

#[test]
fn test_wgpu_parity_adamw_step() {
    use coeus_ops::BackendOps;

    let s = seq();
    let w = wgpu();
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

    let p_g = to_gpu(&p_c);
    let g_g = to_gpu(&g_c);
    let mut m1_g = Tensor::from_slice_on(vec![n], &m1_init, &w);
    let mut m2_g = Tensor::from_slice_on(vec![n], &m2_init, &w);
    let mut p_g_mut = p_g.clone();
    let p_g_layout = p_g_mut.layout().clone();
    let g_g_layout = g_g.layout().clone();
    let m1_g_layout = m1_g.layout().clone();
    let m2_g_layout = m2_g.layout().clone();
    w.adamw_step(
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

    assert_parity(
        "adamw_step",
        p_c_mut.as_slice(),
        to_cpu(&p_g_mut).as_slice(),
    );
}

// ── Full round-trip: CPU→GPU→CPU identity ────────────────────────────────

#[test]
fn test_wgpu_parity_roundtrip_identity() {
    let s = seq();
    let data: Vec<f32> = (0..100).map(|x| x as f32 * 0.123 - 6.15).collect();
    let x = Tensor::<f32, SequentialBackend>::from_slice(vec![10, 10], &data);
    let back = to_gpu(&x).to_backend_on(&wgpu(), &s);
    assert_parity("roundtrip", x.as_slice(), back.as_slice());
}
