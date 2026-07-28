#![allow(clippy::excessive_precision)]

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;

fn assert_tensor_eq<B: coeus_core::ComputeBackend>(
    coeus: &Tensor<f32, B>,
    expected: &[f32],
    tol: f32,
) where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let coeus_slice = coeus.as_slice();
    assert_eq!(coeus_slice.len(), expected.len());
    for (i, (&c, &b)) in coeus_slice.iter().zip(expected.iter()).enumerate() {
        if c.is_infinite() && b.is_infinite() && c.is_sign_positive() == b.is_sign_positive() {
            continue;
        }
        if c.is_nan() && b.is_nan() {
            continue;
        }
        let diff = (c - b).abs();
        assert!(
            diff < tol,
            "Mismatch at index {i}: coeus = {c}, expected = {b} (diff = {diff}, tolerance = {tol})"
        );
    }
}

#[test]
fn test_provider_parity_elementwise_arithmetic() {
    let backend = SequentialBackend::new();

    let shape = vec![3, 4];
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b_data = vec![
        2.0f32, 0.5, 1.5, 3.0, -1.0, 0.0, 2.5, 4.0, 1.0, -2.0, 0.5, 3.0,
    ];

    let a_coeus = Tensor::from_slice(shape.clone(), &a_data).expect("construct tensor");
    let b_coeus = Tensor::from_slice(shape.clone(), &b_data).expect("construct tensor");

    // 1. Add
    let c_coeus = coeus_ops::add(&a_coeus, &b_coeus, &backend).expect("run addition");
    let expected_add = vec![
        3.0, 2.5, 4.5, 7.0, 4.0, 6.0, 9.5, 12.0, 10.0, 8.0, 11.5, 15.0,
    ];
    assert_tensor_eq(&c_coeus, &expected_add, 1e-4);

    // 2. Sub
    let c_coeus = coeus_ops::sub(&a_coeus, &b_coeus, &backend).expect("run subtraction");
    let expected_sub = vec![
        -1.0, 1.5, 1.5, 1.0, 6.0, 6.0, 4.5, 4.0, 8.0, 12.0, 10.5, 9.0,
    ];
    assert_tensor_eq(&c_coeus, &expected_sub, 1e-4);

    // 3. Mul
    let c_coeus = coeus_ops::mul(&a_coeus, &b_coeus, &backend).expect("run multiplication");
    let expected_mul = vec![
        2.0, 1.0, 4.5, 12.0, -5.0, 0.0, 17.5, 32.0, 9.0, -20.0, 5.5, 36.0,
    ];
    assert_tensor_eq(&c_coeus, &expected_mul, 1e-4);

    // 4. Div
    let c_coeus = coeus_ops::div(&a_coeus, &b_coeus, &backend).expect("run division");
    let expected_div = vec![
        0.5,
        4.0,
        2.0,
        1.3333333,
        -5.0,
        f32::INFINITY,
        2.8,
        2.0,
        9.0,
        -5.0,
        22.0,
        4.0,
    ];
    assert_tensor_eq(&c_coeus, &expected_div, 1e-4);
}

#[test]
fn test_provider_parity_activations() {
    let backend = MoiraiBackend::new();

    let shape = vec![2, 3];
    let data = vec![-1.5f32, 2.0, -0.5, 0.0, 1.0, -3.0];

    let x_coeus = Tensor::from_slice(shape, &data).expect("construct tensor");

    // 1. Relu
    let out_coeus = coeus_ops::relu(&x_coeus, &backend).expect("run ReLU");
    let expected_relu = vec![0.0, 2.0, 0.0, 0.0, 1.0, 0.0];
    assert_tensor_eq(&out_coeus, &expected_relu, 1e-4);

    // 2. Sigmoid
    let out_coeus = coeus_ops::sigmoid(&x_coeus, &backend).expect("run sigmoid");
    let expected_sigmoid = vec![
        0.18242552,
        0.880797,
        0.37754068,
        0.5,
        0.7310586,
        0.047425873,
    ];
    assert_tensor_eq(&out_coeus, &expected_sigmoid, 1e-4);

    // 3. Tanh
    let out_coeus = coeus_ops::tanh(&x_coeus, &backend).expect("run tanh");
    let expected_tanh = vec![
        -0.9051482,
        0.9640276,
        -0.46211717,
        0.0,
        0.761_594_2,
        -0.9950547,
    ];
    assert_tensor_eq(&out_coeus, &expected_tanh, 1e-4);

    // 4. Gelu
    let out_coeus = coeus_ops::gelu(&x_coeus, &backend).expect("run GELU");
    let expected_gelu = vec![-0.100227, 1.954598, -0.154269, 0.0, 0.841345, -0.004072];
    assert_tensor_eq(&out_coeus, &expected_gelu, 1e-3);

    // 5. Silu
    let out_coeus = coeus_ops::silu(&x_coeus, &backend).expect("run SiLU");
    let expected_silu = vec![
        -0.27363828,
        1.761594,
        -0.18877034,
        0.0,
        0.7310586,
        -0.14227763,
    ];
    assert_tensor_eq(&out_coeus, &expected_silu, 1e-4);
}

#[test]
fn test_provider_parity_reductions() {
    let backend = SequentialBackend::new();

    let shape = vec![2, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let x_coeus = Tensor::from_slice(shape, &data).expect("construct tensor");

    // 1. Sum along axis 0
    let out_coeus = coeus_ops::sum_axis(&x_coeus, 0, &backend).expect("valid sum axis");
    let expected_sum0 = vec![5.0, 7.0, 9.0];
    assert_tensor_eq(&out_coeus, &expected_sum0, 1e-4);

    // 2. Sum along axis 1
    let out_coeus = coeus_ops::sum_axis(&x_coeus, 1, &backend).expect("valid sum axis");
    let expected_sum1 = vec![6.0, 15.0];
    assert_tensor_eq(&out_coeus, &expected_sum1, 1e-4);

    // 3. Mean along axis 0
    let out_coeus = coeus_ops::mean_axis(&x_coeus, 0, &backend).expect("valid mean axis");
    let expected_mean0 = vec![2.5, 3.5, 4.5];
    assert_tensor_eq(&out_coeus, &expected_mean0, 1e-4);

    // 4. Mean along axis 1
    let out_coeus = coeus_ops::mean_axis(&x_coeus, 1, &backend).expect("valid mean axis");
    let expected_mean1 = vec![2.0, 5.0];
    assert_tensor_eq(&out_coeus, &expected_mean1, 1e-4);
}

#[test]
fn test_provider_parity_matmul() {
    let backend = MoiraiBackend::new();

    // 2D Matmul
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

    let a_coeus = Tensor::from_slice(vec![2, 3], &a_data).expect("construct tensor");
    let b_coeus = Tensor::from_slice(vec![3, 2], &b_data).expect("construct tensor");

    let c_coeus = coeus_ops::matmul(&a_coeus, &b_coeus, &backend).expect("run matrix multiplication");
    let expected_matmul = vec![58.0, 64.0, 139.0, 154.0];
    assert_tensor_eq(&c_coeus, &expected_matmul, 1e-4);
}

#[test]
fn test_provider_parity_batched_matmul() {
    let backend = MoiraiBackend::new();

    // 3D Matmul: [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5,
    ];
    let b_data = vec![
        7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5,
    ];

    let a_coeus = Tensor::from_slice(vec![2, 2, 3], &a_data).expect("construct tensor");
    let b_coeus = Tensor::from_slice(vec![2, 3, 2], &b_data).expect("construct tensor");

    let c_coeus = coeus_ops::matmul(&a_coeus, &b_coeus, &backend).expect("run batched matrix multiplication");
    let expected_batched = vec![58.0, 64.0, 139.0, 154.0, 22.75, 30.25, 45.25, 61.75];
    assert_tensor_eq(&c_coeus, &expected_batched, 1e-4);
}

#[test]
fn test_mnemosyne_huge_pool() {
    let initial_stats = mnemosyne::memory_stats();

    // Allocate and drop a large tensor (e.g. 256KB = 32768 f64s) multiple times
    for _ in 0..10 {
        let t = Tensor::<f64, MoiraiBackend>::zeros([128, 256]).expect("construct tensor");
        drop(t);
    }

    let final_stats = mnemosyne::memory_stats();
    let map_diff = final_stats
        .map_calls
        .saturating_sub(initial_stats.map_calls);
    println!(
        "Initial map_calls: {}, Final map_calls: {}, Diff: {}",
        initial_stats.map_calls, final_stats.map_calls, map_diff
    );
    // Since it caches, diff should be at most 1 (the first allocation mapped from OS, subsequent ones hit cache)
    assert!(
        map_diff <= 1,
        "Mnemosyne huge pool cache miss: map_diff = {map_diff}"
    );
}
