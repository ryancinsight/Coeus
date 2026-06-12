//! Differential verification of the public CPU concat path.
//!
//! `coeus_ops::cat` delegates to the dynamic-rank `coeus-leto` structural
//! dispatch. The expected values below are derived from row-major concatenation
//! of exact scalar values, so exact equality is the correct oracle for both
//! scalar widths.

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
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn assert_values<T: Scalar>(got: &[T], expected: &[T], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert_eq!(actual, reference, "{context}: mismatch at index {index}");
    }
}

fn check_backend<T, B>(backend: &B)
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let first_data: Vec<T> = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let second_data: Vec<T> = [7.0, 10.0, 8.0, 11.0, 9.0, 12.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let first = tensor_from_slice::<T, B>(&[3, 2], &first_data, backend).transpose();
    let second = tensor_from_slice::<T, B>(&[3, 2], &second_data, backend).transpose();
    let expected_axis0: Vec<T> = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]
    .into_iter()
    .map(T::from_f64)
    .collect();
    let expected_axis1: Vec<T> = [
        1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0,
    ]
    .into_iter()
    .map(T::from_f64)
    .collect();

    let axis0 = coeus_ops::cat(&[&first, &second], 0);
    assert_eq!(axis0.shape(), &[4, 3]);
    assert_values(axis0.as_slice(), &expected_axis0, "axis-0 concat");

    let axis1 = coeus_ops::cat(&[&first, &second], 1);
    assert_eq!(axis1.shape(), &[2, 6]);
    assert_values(axis1.as_slice(), &expected_axis1, "axis-1 concat");
}

#[test]
fn sequential_concat_matches_reference() {
    let backend = SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_concat_matches_reference() {
    let backend = MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}
