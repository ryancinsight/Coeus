//! Differential verification of the public CPU split path.
//!
//! `coeus_ops::split` delegates to the dynamic-rank `coeus-leto` structural
//! dispatch. The expected values below are derived from row-major slicing of
//! exact scalar values, so exact equality is the correct oracle for both scalar
//! widths.

use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, Scalar,
    SequentialBackend,
};
use coeus_tensor::{Tensor, Transpose};

fn tensor_from_slice<T, B>(shape: &[usize], data: &[T], backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend).expect("construct tensor")
}

fn assert_values<T: Scalar>(got: &[T], expected: &[T], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert_eq!(actual, reference, "{context}: mismatch at index {index}");
    }
}

fn values<T: Scalar, const N: usize>(input: [f64; N]) -> Vec<T> {
    input.into_iter().map(T::from_f64).collect()
}

fn check_backend<T, B>(backend: &B)
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let data = values::<T, 6>([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let tensor = tensor_from_slice::<T, B>(&[3, 2], &data, backend).transpose();

    let axis1 = coeus_ops::split(&tensor, 2, 1).expect("run operation");
    assert_eq!(axis1.len(), 2);
    assert_eq!(axis1[0].shape(), &[2, 2]);
    assert_values(
        axis1[0].as_slice(),
        &values([1.0, 2.0, 4.0, 5.0]),
        "axis-1 chunk 0",
    );
    assert_eq!(axis1[1].shape(), &[2, 1]);
    assert_values(axis1[1].as_slice(), &values([3.0, 6.0]), "axis-1 chunk 1");

    let axis0 = coeus_ops::split(&tensor, 1, 0).expect("run operation");
    assert_eq!(axis0.len(), 2);
    assert_eq!(axis0[0].shape(), &[1, 3]);
    assert_values(
        axis0[0].as_slice(),
        &values([1.0, 2.0, 3.0]),
        "axis-0 chunk 0",
    );
    assert_eq!(axis0[1].shape(), &[1, 3]);
    assert_values(
        axis0[1].as_slice(),
        &values([4.0, 5.0, 6.0]),
        "axis-0 chunk 1",
    );
}

#[test]
fn sequential_split_matches_reference() {
    let backend = SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_split_matches_reference() {
    let backend = MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}
