use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;

use burn::backend::NdArray as BurnNdArray;
use burn::tensor::{Tensor as BurnTensor, TensorData};

type BurnCpu = BurnNdArray<f32>;

fn assert_tensor_eq<B: coeus_core::ComputeBackend, const D: usize>(
    coeus: &Tensor<f32, B>,
    burn: &BurnTensor<BurnCpu, D>,
    tol: f32,
) where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let coeus_slice = coeus.as_slice();
    let burn_data = burn.clone().into_data();
    let burn_slice = burn_data.iter::<f32>().collect::<Vec<_>>();
    assert_eq!(coeus_slice.len(), burn_slice.len());
    for (i, (&c, &b)) in coeus_slice.iter().zip(burn_slice.iter()).enumerate() {
        if c.is_infinite() && b.is_infinite() && c.is_sign_positive() == b.is_sign_positive() {
            continue;
        }
        if c.is_nan() && b.is_nan() {
            continue;
        }
        let diff = (c - b).abs();
        assert!(
            diff < tol,
            "Mismatch at index {i}: coeus = {c}, burn = {b} (diff = {diff}, tolerance = {tol})"
        );
    }
}

#[test]
fn test_burn_parity_elementwise_arithmetic() {
    let backend = SequentialBackend::new();
    let device = Default::default();

    let shape = vec![3, 4];
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b_data = vec![
        2.0f32, 0.5, 1.5, 3.0, -1.0, 0.0, 2.5, 4.0, 1.0, -2.0, 0.5, 3.0,
    ];

    let a_coeus = Tensor::from_slice(shape.clone(), &a_data);
    let b_coeus = Tensor::from_slice(shape.clone(), &b_data);

    let a_burn =
        BurnTensor::<BurnCpu, 2>::from_data(TensorData::new(a_data.clone(), [3, 4]), &device);
    let b_burn =
        BurnTensor::<BurnCpu, 2>::from_data(TensorData::new(b_data.clone(), [3, 4]), &device);

    // 1. Add
    let c_coeus = coeus_ops::add(&a_coeus, &b_coeus, &backend);
    let c_burn = a_burn.clone() + b_burn.clone();
    assert_tensor_eq(&c_coeus, &c_burn, 1e-4);

    // 2. Sub
    let c_coeus = coeus_ops::sub(&a_coeus, &b_coeus, &backend);
    let c_burn = a_burn.clone() - b_burn.clone();
    assert_tensor_eq(&c_coeus, &c_burn, 1e-4);

    // 3. Mul
    let c_coeus = coeus_ops::mul(&a_coeus, &b_coeus, &backend);
    let c_burn = a_burn.clone() * b_burn.clone();
    assert_tensor_eq(&c_coeus, &c_burn, 1e-4);

    // 4. Div
    let c_coeus = coeus_ops::div(&a_coeus, &b_coeus, &backend);
    let c_burn = a_burn.clone() / b_burn.clone();
    assert_tensor_eq(&c_coeus, &c_burn, 1e-4);
}

#[test]
fn test_burn_parity_activations() {
    let backend = MoiraiBackend::new();
    let device = Default::default();

    let shape = vec![2, 3];
    let data = vec![-1.5f32, 2.0, -0.5, 0.0, 1.0, -3.0];

    let x_coeus = Tensor::from_slice(shape, &data);
    let x_burn = BurnTensor::<BurnCpu, 2>::from_data(TensorData::new(data, [2, 3]), &device);

    // 1. Relu
    let out_coeus = coeus_ops::relu(&x_coeus, &backend);
    let out_burn = burn::tensor::activation::relu(x_burn.clone());
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);

    // 2. Sigmoid
    let out_coeus = coeus_ops::sigmoid(&x_coeus, &backend);
    let out_burn = burn::tensor::activation::sigmoid(x_burn.clone());
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);

    // 3. Tanh
    let out_coeus = coeus_ops::tanh(&x_coeus, &backend);
    let out_burn = x_burn.clone().tanh();
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);

    // 4. Gelu (tanh-based approximation in coeus, compare with slightly wider tolerance)
    let out_coeus = coeus_ops::gelu(&x_coeus, &backend);
    let out_burn = burn::tensor::activation::gelu(x_burn.clone());
    assert_tensor_eq(&out_coeus, &out_burn, 1e-3);

    // 5. Silu
    let out_coeus = coeus_ops::silu(&x_coeus, &backend);
    let out_burn = burn::tensor::activation::silu(x_burn.clone());
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);
}

#[test]
fn test_burn_parity_reductions() {
    let backend = SequentialBackend::new();
    let device = Default::default();

    let shape = vec![2, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let x_coeus = Tensor::from_slice(shape, &data);
    let x_burn = BurnTensor::<BurnCpu, 2>::from_data(TensorData::new(data, [2, 3]), &device);

    // 1. Sum along axis 0
    let out_coeus = coeus_ops::sum_axis(&x_coeus, 0, &backend);
    let out_burn = x_burn.clone().sum_dim(0);
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);

    // 2. Sum along axis 1
    let out_coeus = coeus_ops::sum_axis(&x_coeus, 1, &backend);
    let out_burn = x_burn.clone().sum_dim(1);
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);

    // 3. Mean along axis 0
    let out_coeus = coeus_ops::mean_axis(&x_coeus, 0, &backend);
    let out_burn = x_burn.clone().mean_dim(0);
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);

    // 4. Mean along axis 1
    let out_coeus = coeus_ops::mean_axis(&x_coeus, 1, &backend);
    let out_burn = x_burn.clone().mean_dim(1);
    assert_tensor_eq(&out_coeus, &out_burn, 1e-4);
}

#[test]
fn test_burn_parity_matmul() {
    let backend = MoiraiBackend::new();
    let device = Default::default();

    // 2D Matmul
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

    let a_coeus = Tensor::from_slice(vec![2, 3], &a_data);
    let b_coeus = Tensor::from_slice(vec![3, 2], &b_data);

    let a_burn = BurnTensor::<BurnCpu, 2>::from_data(TensorData::new(a_data, [2, 3]), &device);
    let b_burn = BurnTensor::<BurnCpu, 2>::from_data(TensorData::new(b_data, [3, 2]), &device);

    let c_coeus = coeus_ops::matmul(&a_coeus, &b_coeus, &backend);
    let c_burn = a_burn.matmul(b_burn);
    assert_tensor_eq(&c_coeus, &c_burn, 1e-4);
}

#[test]
fn test_burn_parity_batched_matmul() {
    let backend = MoiraiBackend::new();
    let device = Default::default();

    // 3D Matmul: [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5,
    ];
    let b_data = vec![
        7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5,
    ];

    let a_coeus = Tensor::from_slice(vec![2, 2, 3], &a_data);
    let b_coeus = Tensor::from_slice(vec![2, 3, 2], &b_data);

    let a_burn = BurnTensor::<BurnCpu, 3>::from_data(TensorData::new(a_data, [2, 2, 3]), &device);
    let b_burn = BurnTensor::<BurnCpu, 3>::from_data(TensorData::new(b_data, [2, 3, 2]), &device);

    let c_coeus = coeus_ops::matmul(&a_coeus, &b_coeus, &backend);
    let c_burn = a_burn.matmul(b_burn);
    assert_tensor_eq(&c_coeus, &c_burn, 1e-4);
}
