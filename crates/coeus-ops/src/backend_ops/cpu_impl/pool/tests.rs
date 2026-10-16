//! Rejection contract for the leto-chained pooling delegations.
//!
//! Ports of the historical `validation.rs` unit tests, rewritten against the
//! kernel entry points: geometry errors now surface from leto through
//! [`map_leto_error`](super::super::error::map_leto_error), and rank mismatches
//! from the delegation-local [`check_rank`](super::check_rank). The
//! uninitialized-input defense is intentionally not ported: input
//! initialization is the `unsafe allocate` caller's contract, and no other
//! leto-chained kernel (convolution, unfold/fold) re-checks it.

use super::{avg_pool1d, max_pool1d};
use coeus_core::{BackendError, CpuAddressableStorage, CpuStorage, Layout, SequentialBackend};

#[test]
fn rejects_input_layout_outside_storage() {
    let backend = SequentialBackend::new();
    let input = CpuStorage::from_slice(&[2.0_f32]).expect("input ok");
    let mut output = CpuStorage::from_slice(&[7.0_f32, 8.0]).expect("output ok");
    let input_layout = Layout::new([1, 1, 2].into());
    let output_layout = Layout::new([1, 1, 2].into());

    let result = max_pool1d(
        &backend,
        &input,
        &input_layout,
        1,
        1,
        0,
        1,
        &mut output,
        &output_layout,
    );

    assert!(matches!(
        result,
        Err(BackendError::Storage {
            operation: "max pool 1d",
            ..
        })
    ));
    assert_eq!(output.as_slice(), &[7.0, 8.0]);
}

#[test]
fn rejects_overlapping_parallel_output_layout() {
    let backend = SequentialBackend::new();
    let input = CpuStorage::from_slice(&[2.0_f32, 3.0]).expect("input ok");
    let mut output = CpuStorage::from_slice(&[7.0_f32, 8.0]).expect("output ok");
    let input_layout = Layout::new([1, 1, 2].into());
    let output_layout =
        Layout::from_shape_strides([1, 1, 2].into(), smallvec::smallvec![2, 2, 0], 0);

    let result = max_pool1d(
        &backend,
        &input,
        &input_layout,
        1,
        1,
        0,
        1,
        &mut output,
        &output_layout,
    );

    assert_eq!(
        result,
        Err(BackendError::Storage {
            operation: "max pool 1d",
            reason: "pooling output layout must be injective (no aliasing)".to_string(),
        })
    );
}

#[test]
fn rejects_zero_stride_before_kernel_arithmetic() {
    let backend = SequentialBackend::new();
    let input = CpuStorage::from_slice(&[2.0_f32, 3.0]).expect("input ok");
    let mut output = CpuStorage::from_slice(&[7.0_f32, 8.0]).expect("output ok");
    let input_layout = Layout::new([1, 1, 2].into());
    let output_layout = Layout::new([1, 1, 2].into());

    let result = avg_pool1d(
        &backend,
        &input,
        &input_layout,
        1,
        0,
        0,
        1,
        &mut output,
        &output_layout,
    );

    assert_eq!(
        result,
        Err(BackendError::Storage {
            operation: "avg pool 1d",
            reason: "Invalid input: window stride extents must be nonzero".to_string(),
        })
    );
}

#[test]
fn max_pool_rejects_short_output_before_writing() {
    let backend = SequentialBackend::new();
    let input = CpuStorage::from_slice(&[2.0_f32, 3.0]).expect("input ok");
    let mut output = CpuStorage::from_slice(&[7.0_f32]).expect("output ok");
    let input_layout = Layout::new([1, 1, 2].into());
    let output_layout = Layout::new([1, 1, 2].into());

    let result = max_pool1d(
        &backend,
        &input,
        &input_layout,
        1,
        1,
        0,
        1,
        &mut output,
        &output_layout,
    );

    assert!(matches!(
        result,
        Err(BackendError::Storage {
            operation: "max pool 1d",
            ..
        })
    ));
    assert_eq!(output.as_slice(), &[7.0]);
}

#[test]
fn one_dimensional_pool_rejects_four_dimensional_layouts_before_access() {
    let backend = SequentialBackend::new();
    let input = CpuStorage::from_slice(&[2.0_f32, 3.0, 4.0, 5.0]).expect("input ok");
    let mut output = CpuStorage::from_slice(&[7.0_f32, 8.0]).expect("output ok");
    let input_layout = Layout::new([1, 1, 2, 2].into());
    let output_layout = Layout::new([1, 1, 2].into());

    let result = max_pool1d(
        &backend,
        &input,
        &input_layout,
        1,
        1,
        0,
        1,
        &mut output,
        &output_layout,
    );

    assert_eq!(
        result,
        Err(BackendError::Storage {
            operation: "max pool 1d",
            reason: "max pool 1d expects rank 3 ([N, C, spatial...]), got rank 4".to_string(),
        })
    );
}
