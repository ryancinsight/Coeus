//! Differential verification of the public CPU pad path.
//!
//! `coeus_ops::pad` delegates to the dynamic-rank `coeus-leto` structural
//! dispatch. The expected values below are derived from row-major coordinate
//! mapping over exactly represented scalar values, so exact equality is the
//! correct oracle for both scalar widths.

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

fn check_backend<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let data: Vec<T> = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        .into_iter()
        .map(T::from_f64)
        .collect();
    let tensor = tensor_from_slice::<T, B>(&[3, 2], &data, backend);
    let transposed = tensor.transpose();
    let fill = T::from_f64(-1.0);
    let expected: Vec<T> = [
        -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, -1.0,
    ]
    .into_iter()
    .map(T::from_f64)
    .collect();

    let padded = coeus_ops::pad(&transposed, &[(1, 0), (0, 1)], fill).expect("run operation");

    assert_eq!(padded.shape(), &[3, 4]);
    assert_values(padded.as_slice(), &expected, "transposed pad");
}

#[test]
fn sequential_pad_matches_reference() {
    let backend = SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_pad_matches_reference() {
    let backend = MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}
