//! Differential verification for public tensor identity constructors.
//!
//! `Tensor::eye_on` routes coordinate generation through
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

    let tensor = Tensor::<i32, B>::eye_on(4, &backend);
    let expected =
        coeus_leto::from_shape_fn_values(&[4usize, 4], |index| i32::from(index[0] == index[1]))
            .unwrap();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape(), &[4, 4]);
    assert_eq!(tensor.as_slice(), expected.as_slice());
    assert_eq!(
        tensor.as_slice(),
        &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    );

    let empty = Tensor::<i32, B>::eye_on(0, &backend);
    let expected_empty =
        coeus_leto::from_shape_fn_values(&[0usize, 0], |index| i32::from(index[0] == index[1]))
            .unwrap();

    assert!(empty.is_contiguous());
    assert_eq!(empty.shape(), &[0, 0]);
    assert_eq!(empty.as_slice(), expected_empty.as_slice());
    assert_eq!(empty.as_slice(), &[] as &[i32]);
}

#[test]
fn sequential_identity_matches_leto_dispatch() {
    check_backend::<SequentialBackend>();
}

#[test]
fn moirai_identity_matches_leto_dispatch() {
    check_backend::<MoiraiBackend>();
}
