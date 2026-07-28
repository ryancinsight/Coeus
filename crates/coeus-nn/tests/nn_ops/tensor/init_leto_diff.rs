//! Differential verification of neural-network initializers.
//!
//! `coeus_nn::init::{uniform_with_seed, normal_with_seed}` route through
//! `coeus-leto`, which dispatches dynamic-rank Coeus shapes to Leto's
//! monomorphized seeded random constructors. The oracle here is the direct
//! `coeus-leto` dispatch output for the same shape, scalar type, and seed.

use coeus_autograd::Var;
use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, Scalar,
    SequentialBackend,
};
use coeus_tensor::Tensor;

fn assert_values<T: Scalar>(got: &[T], expected: &[T], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert_eq!(actual, reference, "{context}: mismatch at index {index}");
    }
}

fn check_backend<T, B>(backend: &B)
where
    T: coeus_core::Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = [2usize, 3];
    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend).expect("construct tensor"), true).expect("construct variable");

    coeus_nn::init::uniform_with_seed(&mut weight, -2.0, 5.0, 42).expect("initialize parameters");
    let expected_uniform = coeus_leto::uniform_values(
        &shape,
        <T as coeus_core::Scalar>::from_f64(-2.0),
        <T as coeus_core::Scalar>::from_f64(5.0),
        42,
    )
    .unwrap();
    assert_values(
        weight.tensor.as_slice(),
        &expected_uniform,
        "uniform initializer",
    );

    coeus_nn::init::normal_with_seed(&mut weight, 1.0, 2.0, 11).expect("initialize parameters");
    let expected_normal = coeus_leto::normal_values(
        &shape,
        <T as coeus_core::Scalar>::from_f64(1.0),
        <T as coeus_core::Scalar>::from_f64(2.0),
        11,
    )
    .unwrap();
    assert_values(
        weight.tensor.as_slice(),
        &expected_normal,
        "normal initializer",
    );
}

// kaiming_uniform and kaiming_normal delegate to uniform/normal with analytically
// computed bounds. Verify by reproducing those bounds and comparing against direct
// coeus-leto dispatch.
//
// kaiming_uniform bound: limit = sqrt(6 / fan_in)
// kaiming_normal std_dev: sigma = sqrt(2 / fan_in)
fn check_kaiming<T, B>(backend: &B)
where
    T: coeus_core::Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = [3usize, 4];
    let fan_in = 4usize;
    let seed = 37u64;

    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend).expect("construct tensor"), true).expect("construct variable");
    coeus_nn::init::kaiming_uniform_with_seed(&mut weight, fan_in, seed).expect("initialize parameters");

    let limit = (6.0f64 / fan_in as f64).sqrt();
    let expected_uniform = coeus_leto::uniform_values(
        &shape,
        <T as Scalar>::from_f64(-limit),
        <T as Scalar>::from_f64(limit),
        seed,
    )
    .unwrap();
    assert_values(
        weight.tensor.as_slice(),
        &expected_uniform,
        "kaiming_uniform",
    );

    coeus_nn::init::kaiming_normal_with_seed(&mut weight, fan_in, seed).expect("initialize parameters");
    let std_dev = (2.0f64 / fan_in as f64).sqrt();
    let expected_normal = coeus_leto::normal_values(
        &shape,
        <T as Scalar>::from_f64(0.0),
        <T as Scalar>::from_f64(std_dev),
        seed,
    )
    .unwrap();
    assert_values(weight.tensor.as_slice(), &expected_normal, "kaiming_normal");
}

#[test]
fn sequential_initializers_match_leto_dispatch() {
    let backend = SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_initializers_match_leto_dispatch() {
    let backend = MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

// xavier_uniform bound: limit = sqrt(6 / (fan_in + fan_out))
// xavier_normal std_dev: sigma = sqrt(2 / (fan_in + fan_out))
fn check_xavier<T, B>(backend: &B)
where
    T: coeus_core::Float + coeus_leto::RandomScalar,
    B: coeus_ops::BackendOps<T> + ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = [3usize, 4];
    let fan_in = 3usize;
    let fan_out = 4usize;
    let seed = 53u64;

    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend).expect("construct tensor"), true).expect("construct variable");
    coeus_nn::init::xavier_uniform_with_seed(&mut weight, fan_in, fan_out, seed).expect("initialize parameters");

    let limit = (6.0f64 / (fan_in + fan_out) as f64).sqrt();
    let expected_uniform = coeus_leto::uniform_values(
        &shape,
        <T as Scalar>::from_f64(-limit),
        <T as Scalar>::from_f64(limit),
        seed,
    )
    .unwrap();
    assert_values(
        weight.tensor.as_slice(),
        &expected_uniform,
        "xavier_uniform",
    );

    coeus_nn::init::xavier_normal_with_seed(&mut weight, fan_in, fan_out, seed).expect("initialize parameters");
    let std_dev = (2.0f64 / (fan_in + fan_out) as f64).sqrt();
    let expected_normal = coeus_leto::normal_values(
        &shape,
        <T as Scalar>::from_f64(0.0),
        <T as Scalar>::from_f64(std_dev),
        seed,
    )
    .unwrap();
    assert_values(weight.tensor.as_slice(), &expected_normal, "xavier_normal");
}

#[test]
fn sequential_kaiming_matches_leto_dispatch() {
    let backend = SequentialBackend;
    check_kaiming::<f32, _>(&backend);
    check_kaiming::<f64, _>(&backend);
}

#[test]
fn moirai_kaiming_matches_leto_dispatch() {
    let backend = MoiraiBackend;
    check_kaiming::<f32, _>(&backend);
    check_kaiming::<f64, _>(&backend);
}

#[test]
fn sequential_xavier_matches_leto_dispatch() {
    let backend = SequentialBackend;
    check_xavier::<f32, _>(&backend);
    check_xavier::<f64, _>(&backend);
}

#[test]
fn moirai_xavier_matches_leto_dispatch() {
    let backend = MoiraiBackend;
    check_xavier::<f32, _>(&backend);
    check_xavier::<f64, _>(&backend);
}
