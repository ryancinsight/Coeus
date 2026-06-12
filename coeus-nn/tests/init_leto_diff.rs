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
    let mut weight = Var::new(Tensor::<T, B>::zeros_on(shape, backend), true);

    coeus_nn::init::uniform_with_seed(&mut weight, -2.0, 5.0, 42);
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

    coeus_nn::init::normal_with_seed(&mut weight, 1.0, 2.0, 11);
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
