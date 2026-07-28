use coeus_core::SequentialBackend;
use coeus_ops::{ConvOps, PoolOps};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_conv1d_forward() {
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
    )
    .expect("invariant: validated CPU max_pool2d dispatch must succeed");

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
    )
    .expect("invariant: validated WGPU max_pool2d dispatch must succeed");

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
    )
    .expect("invariant: validated CPU avg_pool2d dispatch must succeed");

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
    )
    .expect("invariant: validated WGPU avg_pool2d dispatch must succeed");

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

#[test]
fn test_wgpu_parity_max_pool2d() {
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
    )
    .expect("invariant: validated CPU max_pool2d dispatch must succeed");

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
    )
    .expect("invariant: validated WGPU max_pool2d dispatch must succeed");

    assert_parity(
        "max_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out).as_slice(),
    );
}

#[test]
fn test_wgpu_parity_avg_pool2d() {
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
    )
    .expect("invariant: validated CPU avg_pool2d dispatch must succeed");

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
    )
    .expect("invariant: validated WGPU avg_pool2d dispatch must succeed");

    assert_parity(
        "avg_pool2d",
        cpu_out.as_slice(),
        to_cpu(&gpu_out).as_slice(),
    );
}
