//! Value-parity fixtures for the leto-chained unfold/fold delegations.
//!
//! Each case pins exact outputs through the [`UnfoldFoldOps`](crate::UnfoldFoldOps)
//! entry points (validation plus delegated traversal), covering padding,
//! dilation, stride, overlap accumulation, the fold-after-unfold adjoint
//! identity, degenerate empty outputs, and the kernel-exceeds rejection the
//! contract gate requires (`Ok(0)` in leto would be a behavior change here).

use super::{fold1d, fold2d, unfold1d, unfold2d};
use crate::UnfoldFoldOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuStorage, Layout, SequentialBackend};

fn backend() -> SequentialBackend {
    SequentialBackend::new()
}

fn storage(data: &[f32]) -> CpuStorage<f32> {
    CpuStorage::from_slice(data).expect("invariant: test storage succeeds")
}

#[test]
fn unfold1d_dilated_padded_matches_fixture() {
    let backend = backend();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut output = storage(&[0.0; 6]);
    backend
        .unfold1d(
            &input,
            &Layout::new([1, 1, 6].into()),
            2,
            2,
            1,
            2,
            &mut output,
            &Layout::new([1, 2, 3].into()),
        )
        .expect("invariant: valid unfold1d fixture succeeds");
    assert_eq!(output.as_slice(), &[0.0, 2.0, 4.0, 2.0, 4.0, 6.0]);
    // The kernel entry point agrees with the dispatched path.
    let mut direct = [0.0_f32; 6];
    unfold1d(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Layout::new([1, 1, 6].into()),
        2,
        2,
        1,
        2,
        &mut direct,
        &Layout::new([1, 2, 3].into()),
    )
    .expect("invariant: valid unfold1d fixture succeeds");
    assert_eq!(direct, [0.0, 2.0, 4.0, 2.0, 4.0, 6.0]);
}

#[test]
fn unfold2d_strided_matches_fixture() {
    let backend = backend();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let mut output = storage(&[0.0; 16]);
    backend
        .unfold2d(
            &input,
            &Layout::new([1, 1, 3, 3].into()),
            2,
            2,
            1,
            1,
            0,
            0,
            1,
            1,
            &mut output,
            &Layout::new([1, 4, 4].into()),
        )
        .expect("invariant: valid unfold2d fixture succeeds");
    assert_eq!(
        output.as_slice(),
        &[1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0]
    );
    let mut direct = [0.0_f32; 16];
    unfold2d(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &Layout::new([1, 1, 3, 3].into()),
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        &mut direct,
        &Layout::new([1, 4, 4].into()),
    )
    .expect("invariant: valid unfold2d fixture succeeds");
    assert_eq!(
        direct,
        [1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0]
    );
}

#[test]
fn fold1d_accumulates_overlap() {
    let backend = backend();
    let input = storage(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    let mut output = storage(&[0.0; 4]);
    backend
        .fold1d(
            &input,
            &Layout::new([1, 2, 3].into()),
            4,
            2,
            1,
            0,
            1,
            &mut output,
            &Layout::new([1, 1, 4].into()),
        )
        .expect("invariant: valid fold1d fixture succeeds");
    assert_eq!(output.as_slice(), &[1.0, 12.0, 23.0, 30.0]);
    let mut direct = [0.0_f32; 4];
    fold1d(
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        &Layout::new([1, 2, 3].into()),
        4,
        2,
        1,
        0,
        1,
        &mut direct,
        &Layout::new([1, 1, 4].into()),
    )
    .expect("invariant: valid fold1d fixture succeeds");
    assert_eq!(direct, [1.0, 12.0, 23.0, 30.0]);
}

#[test]
fn fold2d_accumulates_coverage_grid() {
    let backend = backend();
    let input = storage(&[2.0; 16]);
    let mut output = storage(&[0.0; 9]);
    backend
        .fold2d(
            &input,
            &Layout::new([1, 4, 4].into()),
            3,
            3,
            2,
            2,
            1,
            1,
            0,
            0,
            1,
            1,
            &mut output,
            &Layout::new([1, 1, 3, 3].into()),
        )
        .expect("invariant: valid fold2d fixture succeeds");
    assert_eq!(
        output.as_slice(),
        &[2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0]
    );
    let mut direct = [0.0_f32; 9];
    fold2d(
        &[2.0; 16],
        &Layout::new([1, 4, 4].into()),
        3,
        3,
        2,
        2,
        1,
        1,
        0,
        0,
        1,
        1,
        &mut direct,
        &Layout::new([1, 1, 3, 3].into()),
    )
    .expect("invariant: valid fold2d fixture succeeds");
    assert_eq!(direct, [2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0]);
}

#[test]
fn fold_after_unfold_scales_by_coverage() {
    // Adjoint identity: folding the unfolded signal reproduces each input
    // element scaled by its window coverage count ([2, 3, 3, 3, 2] here).
    let backend = backend();
    let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut unfolded = storage(&[0.0; 15]);
    backend
        .unfold1d(
            &input,
            &Layout::new([1, 1, 5].into()),
            3,
            1,
            1,
            1,
            &mut unfolded,
            &Layout::new([1, 3, 5].into()),
        )
        .expect("invariant: valid unfold1d fixture succeeds");
    let mut refolded = storage(&[0.0; 5]);
    backend
        .fold1d(
            &unfolded,
            &Layout::new([1, 3, 5].into()),
            5,
            3,
            1,
            1,
            1,
            &mut refolded,
            &Layout::new([1, 1, 5].into()),
        )
        .expect("invariant: valid fold1d fixture succeeds");
    assert_eq!(refolded.as_slice(), &[2.0, 6.0, 9.0, 12.0, 10.0]);
}

#[test]
fn rejects_kernel_exceeding_padded_input() {
    // The contract gate pins this rejection: leto reports an empty success
    // for kernel-exceeds-input, but this surface (and PyTorch) errors.
    let backend = backend();
    let input = storage(&[1.0, 2.0]);
    let mut output = storage(&[0.0; 3]);
    let result = backend.unfold1d(
        &input,
        &Layout::new([1, 1, 2].into()),
        3,
        1,
        0,
        1,
        &mut output,
        &Layout::new([1, 3, 1].into()),
    );
    assert!(matches!(
        result,
        Err(BackendError::Storage { operation: "unfold1d", reason })
            if reason.contains("effective kernel exceeds the padded input")
    ));
}

#[test]
fn degenerate_channel_empty_round_trips() {
    // Zero channels collapse both sides to empty storage; delegation must
    // succeed without touching any element.
    let backend = backend();
    let input = storage(&[]);
    let mut output = storage(&[]);
    backend
        .unfold1d(
            &input,
            &Layout::new([1, 0, 5].into()),
            3,
            1,
            0,
            1,
            &mut output,
            &Layout::new([1, 0, 3].into()),
        )
        .expect("invariant: degenerate unfold1d succeeds");
    assert!(output.as_slice().is_empty());
}
