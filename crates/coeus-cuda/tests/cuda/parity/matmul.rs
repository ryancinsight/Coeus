use super::*;

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
