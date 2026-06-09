//! Differential verification of the contiguous reduction CPU path.
//!
//! When the reduced axis has unit stride, each output's run is a flat slice and
//! is routed to the hermes SIMD SSOT (`sum`/`min`/`max`). min/max are exactly
//! associative (value-identical to a scalar fold for non-NaN inputs); sum
//! reassociates under SIMD, so it is checked within an epsilon bound.

use coeus_core::{
    ComputeBackend, CpuAddressableStorageMut, Layout, MoiraiBackend, Scalar, SequentialBackend,
    Shape,
};
use coeus_ops::{BackendOps, CpuBackend, ReductionOp};

/// Reduce the last (unit-stride) axis of a contiguous `[rows, cols]` tensor.
fn reduce_last_axis<T: Scalar, B: CpuBackend>(
    backend: &B,
    op: ReductionOp,
    rows: usize,
    cols: usize,
    data: &[T],
) -> Vec<T>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let a_layout = Layout::new(Shape::from(vec![rows, cols]));
    let c_layout = Layout::new(Shape::from(vec![rows, 1]));

    let mut a_buf = ComputeBackend::allocate::<T>(backend, rows * cols);
    let mut c_buf = ComputeBackend::allocate::<T>(backend, rows);
    backend.copy_to_device(data, &mut a_buf);

    backend.reduce(op, &a_buf, &a_layout, 1, &mut c_buf, &c_layout);

    let mut out = vec![T::zero(); rows];
    backend.copy_to_host(&c_buf, &mut out);
    out
}

fn check_f32<B: CpuBackend>(backend: &B)
where
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    // cols spans the SIMD lane/tail boundary; rows exercises multiple output runs.
    for &(rows, cols) in &[
        (1usize, 1usize),
        (3, 7),
        (4, 8),
        (5, 17),
        (8, 1024),
        (33, 257),
    ] {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32).sin() * 4.0 - 1.0)
            .collect();
        let rowof = |r: usize| &data[r * cols..(r + 1) * cols];

        let sum = reduce_last_axis(backend, ReductionOp::Sum, rows, cols, &data);
        for (r, &got) in sum.iter().enumerate() {
            let expected: f32 = rowof(r).iter().copied().sum();
            let tol = 1e-4 * (1.0 + expected.abs()); // reassociation epsilon
            assert!(
                (got - expected).abs() <= tol,
                "sum row {r} ({rows}x{cols}): got {got}, expected {expected}",
            );
        }

        let min = reduce_last_axis(backend, ReductionOp::Min, rows, cols, &data);
        let max = reduce_last_axis(backend, ReductionOp::Max, rows, cols, &data);
        for (r, (&gmin, &gmax)) in min.iter().zip(&max).enumerate() {
            let emin = rowof(r).iter().copied().fold(f32::INFINITY, f32::min);
            let emax = rowof(r).iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Exact: min/max do not reassociate.
            assert_eq!(
                gmin.to_bits(),
                emin.to_bits(),
                "min row {r} ({rows}x{cols})"
            );
            assert_eq!(
                gmax.to_bits(),
                emax.to_bits(),
                "max row {r} ({rows}x{cols})"
            );
        }
    }
}

#[test]
fn sequential_reduction_matches_reference() {
    check_f32(&SequentialBackend);
}

#[test]
fn moirai_reduction_matches_reference() {
    check_f32(&MoiraiBackend);
}
