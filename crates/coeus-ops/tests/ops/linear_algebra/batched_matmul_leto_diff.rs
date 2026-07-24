//! Differential verification of public batched CPU matmul.
//!
//! This covers the `coeus_ops::matmul` batching layer above
//! `BackendOps::matmul`: it builds per-batch 2-D layouts and dispatches each
//! slice to the CPU backend, which then routes through `coeus-leto`.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

fn batched_reference<T: Scalar>(
    a: &[T],
    batches: usize,
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
    b_batches: usize,
) -> Vec<T> {
    let mut out = vec![T::zero(); batches * m * n];
    for batch in 0..batches {
        let a_base = batch * m * k;
        let b_base = (batch % b_batches) * k * n;
        let c_base = batch * m * n;
        for row in 0..m {
            for col in 0..n {
                let mut acc = T::zero();
                for inner in 0..k {
                    acc += a[a_base + row * k + inner] * b[b_base + inner * n + col];
                }
                out[c_base + row * n + col] = acc;
            }
        }
    }
    out
}

fn tensor_from_slice<T, B>(shape: &[usize], data: &[T], backend: &B) -> Tensor<T, B>
where
    T: Scalar,
    B: ComputeBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn assert_same_bits<T: Scalar>(got: &[T], expected: &[T]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            Scalar::to_f64(actual).to_bits(),
            Scalar::to_f64(reference).to_bits(),
            "batched matmul mismatch at index {index}",
        );
    }
}

fn check_equal_batch_matmul<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: ComputeBackend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let a: Vec<T> = (1..=12).map(|value| T::from_f64(value as f64)).collect();
    let b: Vec<T> = (1..=12)
        .map(|value| T::from_f64((value + 20) as f64))
        .collect();

    let a_tensor = tensor_from_slice::<T, B>(&[2, 2, 3], &a, backend);
    let b_tensor = tensor_from_slice::<T, B>(&[2, 3, 2], &b, backend);
    let got =
        coeus_ops::matmul(&a_tensor, &b_tensor, backend);
    let expected = batched_reference(&a, 2, 2, 3, &b, 2, 2);

    assert_eq!(got.shape(), &[2, 2, 2]);
    assert_same_bits(got.as_slice(), &expected);
}

fn check_rhs_broadcast_matmul<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: ComputeBackend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let a: Vec<T> = (1..=12).map(|value| T::from_f64(value as f64)).collect();
    let b: Vec<T> = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
        .into_iter()
        .map(T::from_f64)
        .collect();

    let a_tensor = tensor_from_slice::<T, B>(&[2, 2, 3], &a, backend);
    let b_tensor = tensor_from_slice::<T, B>(&[3, 2], &b, backend);
    let got =
        coeus_ops::matmul(&a_tensor, &b_tensor, backend);
    let expected = batched_reference(&a, 2, 2, 3, &b, 2, 1);

    assert_eq!(got.shape(), &[2, 2, 2]);
    assert_same_bits(got.as_slice(), &expected);
}

fn check_backend<T, B>(backend: &B)
where
    T: Scalar + leto_ops::Scalar,
    B: ComputeBackend + coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    check_equal_batch_matmul::<T, B>(backend);
    check_rhs_broadcast_matmul::<T, B>(backend);
}

#[test]
fn sequential_batched_matmul_matches_reference() {
    let backend = coeus_core::SequentialBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}

#[test]
fn moirai_batched_matmul_matches_reference() {
    let backend = coeus_core::MoiraiBackend;
    check_backend::<f32, _>(&backend);
    check_backend::<f64, _>(&backend);
}
