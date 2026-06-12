//! Differential verification for cross-backend tensor transfers.
//!
//! Non-contiguous cross-backend transfers route through
//! `coeus-leto::contiguous_values` before copying into the destination backend.

use coeus_core::{CpuAddressableStorage, MoiraiBackend, SequentialBackend};
use coeus_tensor::{Tensor, Transpose};

#[test]
fn sequential_to_moirai_transfer_matches_leto_materialization() {
    let seq = SequentialBackend;
    let moirai = MoiraiBackend;
    let data = (0..12).collect::<Vec<i32>>();
    let tensor = Tensor::<i32, SequentialBackend>::from_slice_on([3usize, 4], &data, &seq);
    let view = tensor.slice(&[(0, 3), (1, 4)]).transpose();

    let expected = coeus_leto::contiguous_values(view.layout(), view.storage().as_slice()).unwrap();
    let transferred = view.to_backend_on(&seq, &moirai);

    assert!(transferred.is_contiguous());
    assert_eq!(transferred.shape(), view.shape());
    assert_eq!(transferred.as_slice(), expected.as_slice());
    assert_eq!(transferred.as_slice(), &[1, 5, 9, 2, 6, 10, 3, 7, 11]);
}

#[test]
fn moirai_to_sequential_transfer_matches_leto_materialization() {
    let seq = SequentialBackend;
    let moirai = MoiraiBackend;
    let data = (0..24).collect::<Vec<i32>>();
    let tensor = Tensor::<i32, MoiraiBackend>::from_slice_on([2usize, 3, 4], &data, &moirai);
    let view = tensor.slice(&[(0, 2), (1, 3), (0, 4)]).t_nd();

    let expected = coeus_leto::contiguous_values(view.layout(), view.storage().as_slice()).unwrap();
    let transferred = view.to_backend_on(&moirai, &seq);

    assert!(transferred.is_contiguous());
    assert_eq!(transferred.shape(), view.shape());
    assert_eq!(transferred.as_slice(), expected.as_slice());
    assert_eq!(transferred.get(&[1, 3, 1]), view.get(&[1, 3, 1]));
}
