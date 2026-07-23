use coeus_tensor::Tensor;

use super::{seq, to_cpu, to_gpu, wgpu};

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
