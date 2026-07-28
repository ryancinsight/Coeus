//! Differential verification for public tensor contiguous materialization.
//!
//! `Tensor::to_contiguous_on` routes CPU-addressable views through
//! `coeus-leto::contiguous_values`; this test asserts exact value parity for
//! offset, strided, and transposed layouts.

use coeus_core::{CpuAddressableStorage, MoiraiBackend, SequentialBackend};
use coeus_tensor::{Tensor, Transpose};

fn assert_contiguous_matches_leto<B>()
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<i32>:
        coeus_core::CpuAddressableStorage<i32> + coeus_core::CpuAddressableStorageMut<i32>,
{
    let backend = B::default();
    let data = (0..12).collect::<Vec<i32>>();
    let tensor = Tensor::<i32, B>::from_slice_on([3usize, 4], &data, &backend).expect("construct tensor");

    let view = tensor.slice(&[(0, 3), (1, 4)]).transpose();
    assert_eq!(view.shape(), &[3, 3]);
    assert!(!view.is_contiguous());

    let contiguous = view
        .to_contiguous_on(&backend)
        .expect("materialize contiguous tensor");
    let expected = coeus_leto::contiguous_values(view.layout(), view.storage().as_slice()).unwrap();

    assert!(contiguous.is_contiguous());
    assert_eq!(contiguous.shape(), view.shape());
    assert_eq!(contiguous.as_slice(), expected.as_slice());
    assert_eq!(contiguous.as_slice(), &[1, 5, 9, 2, 6, 10, 3, 7, 11]);
}

#[test]
fn sequential_to_contiguous_matches_leto_dispatch() {
    assert_contiguous_matches_leto::<SequentialBackend>();
}

#[test]
fn moirai_to_contiguous_matches_leto_dispatch() {
    assert_contiguous_matches_leto::<MoiraiBackend>();
}
