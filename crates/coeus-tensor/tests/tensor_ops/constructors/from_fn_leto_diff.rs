//! Differential verification for tensor coordinate constructors.
//!
//! `Tensor::from_fn_on` routes coordinate generation through
//! `coeus-leto::from_shape_fn_values`.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::Tensor;

fn check_backend<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();
    let shape = [2usize, 3, 2];
    let tensor = Tensor::<i32, B>::from_fn_on(shape, &backend, |index| {
        i32::try_from(index[0] * 100 + index[1] * 10 + index[2]).unwrap()
    });
    let expected = coeus_leto::from_shape_fn_values(&shape, |index| {
        i32::try_from(index[0] * 100 + index[1] * 10 + index[2]).unwrap()
    })
    .unwrap();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape(), &shape);
    assert_eq!(tensor.as_slice(), expected.as_slice());
    assert_eq!(
        tensor.as_slice(),
        &[0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121]
    );
}

#[test]
fn sequential_from_fn_matches_leto_dispatch() {
    check_backend::<SequentialBackend>();
}

#[test]
fn moirai_from_fn_matches_leto_dispatch() {
    check_backend::<MoiraiBackend>();
}
