use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_strided_add_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    // [3, 4] matrix, then transpose to [4, 3] non-contiguous view.
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let at = a.t();
    let bt = b.t();
    let cpu_out = coeus_ops::add(&at, &bt, &s);
    let a_gpu = to_gpu(&a).t();
    let b_gpu = to_gpu(&b).t();
    let gpu_out = to_cpu(&coeus_ops::add(&a_gpu, &b_gpu, &w));
    assert_parity(
        "strided_add_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_mul_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.5).collect();
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let at = a.t();
    let bt = b.t();
    let cpu_out = coeus_ops::mul(&at, &bt, &s);
    let a_gpu = to_gpu(&a).t();
    let b_gpu = to_gpu(&b).t();
    let gpu_out = to_cpu(&coeus_ops::mul(&a_gpu, &b_gpu, &w));
    assert_parity(
        "strided_mul_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_exp_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    // Small values to avoid inf in exp.
    let data: Vec<f32> = (0..12).map(|x| (x as f32 - 5.5) * 0.1).collect();
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let at = a.t();
    let cpu_out = coeus_ops::elementwise_unary(&at, &s, coeus_ops::UnaryOp::Exp)
        .expect("valid CPU strided exponential input");
    let a_gpu = to_gpu(&a).t();
    let gpu_out = to_cpu(
        &coeus_ops::elementwise_unary(&a_gpu, &w, coeus_ops::UnaryOp::Exp)
            .expect("valid WGPU strided exponential input"),
    );
    assert_parity(
        "strided_exp_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_neg_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let data: Vec<f32> = (0..12).map(|x| x as f32 - 6.0).collect();
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![4, 3], &data);
    let at = a.t();
    let cpu_out = coeus_ops::elementwise_unary(&at, &s, coeus_ops::UnaryOp::Neg)
        .expect("valid CPU strided negation input");
    let a_gpu = to_gpu(&a).t();
    let gpu_out = to_cpu(
        &coeus_ops::elementwise_unary(&a_gpu, &w, coeus_ops::UnaryOp::Neg)
            .expect("valid WGPU strided negation input"),
    );
    assert_parity(
        "strided_neg_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_sqrt_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let at = a.t();
    let cpu_out = coeus_ops::elementwise_unary(&at, &s, coeus_ops::UnaryOp::Sqrt)
        .expect("valid CPU strided square-root input");
    let a_gpu = to_gpu(&a).t();
    let gpu_out = to_cpu(
        &coeus_ops::elementwise_unary(&a_gpu, &w, coeus_ops::UnaryOp::Sqrt)
            .expect("valid WGPU strided square-root input"),
    );
    assert_parity(
        "strided_sqrt_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_elu_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let data = [
        -3.0f32, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.0, -0.5, 0.5, 1.5,
    ];
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let at = a.t();
    let cpu_out = coeus_ops::elementwise_unary(&at, &s, coeus_ops::UnaryOp::Elu)
        .expect("valid CPU strided ELU input");
    let a_gpu = to_gpu(&a).t();
    let gpu_out = to_cpu(
        &coeus_ops::elementwise_unary(&a_gpu, &w, coeus_ops::UnaryOp::Elu)
            .expect("valid WGPU Hephaestus strided ELU input"),
    );
    assert_parity(
        "strided_elu_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_elu_gradient_transposed_matches_cpu() {
    let s = seq();
    let w = wgpu();
    let data = [
        -3.0f32, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.0, -0.5, 0.5, 1.5,
    ];
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data);
    let at = a.t();
    let cpu_out = coeus_ops::elementwise_unary(&at, &s, coeus_ops::UnaryOp::EluGrad)
        .expect("valid CPU strided ELU gradient input");
    let a_gpu = to_gpu(&a).t();
    let gpu_out = to_cpu(
        &coeus_ops::elementwise_unary(&a_gpu, &w, coeus_ops::UnaryOp::EluGrad)
            .expect("valid WGPU Hephaestus strided ELU gradient input"),
    );
    assert_parity(
        "strided_elu_gradient_transposed",
        cpu_out.as_slice(),
        gpu_out.as_slice(),
    );
}

#[test]
fn test_wgpu_strided_rank3_binary_matches_cpu() {
    let s = seq();
    let w = wgpu();
    // Rank-3: [2, 3, 4], then permute to [4, 2, 3] (non-contiguous).
    let data: Vec<f32> = (0..24).map(|x| x as f32 * 0.25 - 3.0).collect();
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 4], &data);
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 4], &data);
    let ap = a.permute(&[2, 0, 1]);
    let bp = b.permute(&[2, 0, 1]);
    let cpu_out = coeus_ops::add(&ap, &bp, &s);
    let ap_gpu = to_gpu(&a).permute(&[2, 0, 1]);
    let bp_gpu = to_gpu(&b).permute(&[2, 0, 1]);
    let gpu_out = to_cpu(&coeus_ops::add(&ap_gpu, &bp_gpu, &w));
    assert_parity("strided_rank3_add", cpu_out.as_slice(), gpu_out.as_slice());
}

#[test]
fn test_wgpu_parity_roundtrip_identity() {
    let s = seq();
    let data: Vec<f32> = (0..100).map(|x| x as f32 * 0.123 - 6.15).collect();
    let x = Tensor::<f32, SequentialBackend>::from_slice(vec![10, 10], &data);
    let back = to_gpu(&x).to_backend_on(&wgpu(), &s);
    assert_parity("roundtrip", x.as_slice(), back.as_slice());
}
