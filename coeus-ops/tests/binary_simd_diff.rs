//! Differential verification of the contiguous elementwise-binary CPU path.
//!
//! `MulOp` routes f32/f64 contiguous products through the hermes-simd SSOT
//! (`hermes_simd::elementwise_mul`); the remaining ops use the scalar default.
//! Each binary op is a single per-lane IEEE operation, so the SIMD result is
//! bitwise-identical to a scalar reference — these tests assert exact equality
//! (no epsilon), across sizes that span the parallel chunk boundary (8192).

use coeus_core::{
    ComputeBackend, CpuAddressableStorageMut, Layout, MoiraiBackend, Scalar, SequentialBackend,
    Shape,
};
use coeus_ops::backend_ops::ElementwiseOps;
use coeus_ops::{BinaryOp, CpuBackend};

/// Sizes chosen to exercise: empty-ish, sub-SIMD-width, exact chunk, chunk+1,
/// multi-chunk, and a non-chunk-multiple multi-chunk tail.
const SIZES: &[usize] = &[1, 2, 7, 8, 31, 8192, 8193, 16_384, 20_001];

/// Scalar reference for a single op, in the element type's native precision.
fn reference<T: Scalar>(op: BinaryOp, x: T, y: T) -> T {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
    }
}

/// Drive the public CPU kernel for one (backend, op, length) and return the host result.
fn device_binary<T: Scalar + leto_ops::Scalar, B: CpuBackend>(
    backend: &B,
    op: BinaryOp,
    a: &[T],
    b: &[T],
) -> Vec<T>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = a.len();
    let layout = Layout::new(Shape::from(vec![n]));

    let mut a_buf = ComputeBackend::allocate::<T>(backend, n);
    let mut b_buf = ComputeBackend::allocate::<T>(backend, n);
    let mut c_buf = ComputeBackend::allocate::<T>(backend, n);
    backend.copy_to_device(a, &mut a_buf);
    backend.copy_to_device(b, &mut b_buf);

    backend.elementwise_binary(op, &a_buf, &layout, &b_buf, &layout, &mut c_buf, &layout);

    let mut out = vec![T::zero(); n];
    backend.copy_to_host(&c_buf, &mut out);
    out
}

fn check_op<T: Scalar + leto_ops::Scalar, B: CpuBackend>(backend: &B, op: BinaryOp)
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    for &n in SIZES {
        // Deterministic inputs; b is always >= 1 so Div has no zero divisor.
        let a: Vec<T> = (0..n).map(|i| T::from_f64(i as f64 * 0.5 - 3.0)).collect();
        let b: Vec<T> = (0..n).map(|i| T::from_f64((i % 7) as f64 + 1.0)).collect();

        let got = device_binary(backend, op, &a, &b);
        let expected: Vec<T> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| reference(op, x, y))
            .collect();

        for i in 0..n {
            // Bitwise-exact: single IEEE op per lane, scalar == SIMD.
            assert_eq!(
                Scalar::to_f64(got[i]).to_bits(),
                Scalar::to_f64(expected[i]).to_bits(),
                "{op:?} mismatch at i={i}, n={n}",
            );
        }
    }
}

const OPS: &[BinaryOp] = &[BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];

#[test]
fn sequential_f32_matches_scalar_reference() {
    let backend = SequentialBackend;
    for &op in OPS {
        check_op::<f32, _>(&backend, op);
    }
}

#[test]
fn sequential_f64_matches_scalar_reference() {
    let backend = SequentialBackend;
    for &op in OPS {
        check_op::<f64, _>(&backend, op);
    }
}

#[test]
fn moirai_f32_matches_scalar_reference() {
    let backend = MoiraiBackend;
    for &op in OPS {
        check_op::<f32, _>(&backend, op);
    }
}

#[test]
fn moirai_f64_matches_scalar_reference() {
    let backend = MoiraiBackend;
    for &op in OPS {
        check_op::<f64, _>(&backend, op);
    }
}
