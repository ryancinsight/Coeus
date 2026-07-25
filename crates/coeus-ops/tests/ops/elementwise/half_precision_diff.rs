//! Differential verification of coeus-ops on F16 and Bf16.
//!
//! All test inputs are small integers exactly representable in both
//! half-precision formats:
//!   F16:  11-bit mantissa → exact for integers up to 2^11 = 2048.
//!   Bf16: 7-bit mantissa  → exact for integers up to 2^7  = 128.
//!
//! Assertions compare `T::to_f32()` output against the f32 reference rounded
//! back to `T` — this avoids asserting equality across different float widths.
//!
//! SequentialBackend and MoiraiBackend are verified against the same reference;
//! divergence between them indicates a backend dispatch bug.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

// ── constructors ─────────────────────────────────────────────────────────────

fn t_f16<B>(shape: &[usize], vals: &[f32], backend: &B) -> Tensor<F16, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<F16>: CpuAddressableStorageMut<F16>,
{
    let data: Vec<F16> = vals.iter().map(|&v| F16::from_f32(v)).collect();
    Tensor::from_slice_on(shape.to_vec(), &data, backend)
}

fn t_bf16<B>(shape: &[usize], vals: &[f32], backend: &B) -> Tensor<Bf16, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<Bf16>: CpuAddressableStorageMut<Bf16>,
{
    let data: Vec<Bf16> = vals.iter().map(|&v| Bf16::from_f32(v)).collect();
    Tensor::from_slice_on(shape.to_vec(), &data, backend)
}

// ── assertion helpers ─────────────────────────────────────────────────────────

fn assert_f16_exact(got: &[F16], expected_f32: &[f32], context: &str) {
    assert_eq!(got.len(), expected_f32.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected_f32).enumerate() {
        let e_f16 = F16::from_f32(e);
        assert_eq!(
            g,
            e_f16,
            "{context}[{i}]: got {} expected {}",
            g.to_f32(),
            e
        );
    }
}

fn assert_bf16_exact(got: &[Bf16], expected_f32: &[f32], context: &str) {
    assert_eq!(got.len(), expected_f32.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected_f32).enumerate() {
        let e_bf16 = Bf16::from_f32(e);
        assert_eq!(
            g,
            e_bf16,
            "{context}[{i}]: got {} expected {}",
            g.to_f32(),
            e
        );
    }
}

// ── F16 checks ────────────────────────────────────────────────────────────────

fn check_f16<B>(backend: &B)
where
    B: coeus_ops::BackendOps<F16> + Default,
    B::DeviceBuffer<F16>: CpuAddressableStorage<F16> + CpuAddressableStorageMut<F16>,
{
    // add: [1,2,3,4] + [4,3,2,1] = [5,5,5,5]
    let a = t_f16(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    let b = t_f16(&[4], &[4.0, 3.0, 2.0, 1.0], backend);
    let sum = coeus_ops::add(&a, &b, backend);
    assert_f16_exact(sum.as_slice(), &[5.0, 5.0, 5.0, 5.0], "F16 add");

    // matmul: A=[[1,2],[3,4]] @ I₂ = A — identity matmul, exact.
    let mat = t_f16(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let eye = t_f16(&[2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let prod = coeus_ops::matmul(&mat, &eye, backend);
    assert_eq!(prod.shape(), &[2, 2], "F16 matmul shape");
    assert_f16_exact(prod.as_slice(), &[1.0, 2.0, 3.0, 4.0], "F16 matmul A@I=A");

    // sum over a vector: [1,2,3,4,5,6] → 21  (coeus_ops::sum returns T)
    let v = t_f16(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let s: F16 = coeus_ops::sum(&v, backend).expect("valid F16 sum");
    assert_eq!(
        s,
        F16::from_f32(21.0),
        "F16 sum: got {}, expected 21",
        s.to_f32()
    );

    // relu: [-3,-1,0,2,4] → [0,0,0,2,4]
    let x = t_f16(&[5], &[-3.0, -1.0, 0.0, 2.0, 4.0], backend);
    let r = coeus_ops::relu(&x, backend);
    assert_f16_exact(r.as_slice(), &[0.0, 0.0, 0.0, 2.0, 4.0], "F16 relu");
}

// ── Bf16 checks ───────────────────────────────────────────────────────────────

fn check_bf16<B>(backend: &B)
where
    B: coeus_ops::BackendOps<Bf16> + Default,
    B::DeviceBuffer<Bf16>: CpuAddressableStorage<Bf16> + CpuAddressableStorageMut<Bf16>,
{
    // add: [1,2,3,4] + [4,3,2,1] = [5,5,5,5]
    let a = t_bf16(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    let b = t_bf16(&[4], &[4.0, 3.0, 2.0, 1.0], backend);
    let sum = coeus_ops::add(&a, &b, backend);
    assert_bf16_exact(sum.as_slice(), &[5.0, 5.0, 5.0, 5.0], "Bf16 add");

    // matmul: A @ I = A
    let mat = t_bf16(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let eye = t_bf16(&[2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let prod = coeus_ops::matmul(&mat, &eye, backend);
    assert_eq!(prod.shape(), &[2, 2], "Bf16 matmul shape");
    assert_bf16_exact(prod.as_slice(), &[1.0, 2.0, 3.0, 4.0], "Bf16 matmul A@I=A");

    // sum: [1,2,3,4,5,6] → 21  (within Bf16's 7-bit mantissa; 21 < 128)
    let v = t_bf16(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let s: Bf16 = coeus_ops::sum(&v, backend).expect("valid Bf16 sum");
    assert_eq!(
        s,
        Bf16::from_f32(21.0),
        "Bf16 sum: got {}, expected 21",
        s.to_f32()
    );

    // relu: [-3,-1,0,2,4] → [0,0,0,2,4]
    let x = t_bf16(&[5], &[-3.0, -1.0, 0.0, 2.0, 4.0], backend);
    let r = coeus_ops::relu(&x, backend);
    assert_bf16_exact(r.as_slice(), &[0.0, 0.0, 0.0, 2.0, 4.0], "Bf16 relu");
}

// ── test wrappers ─────────────────────────────────────────────────────────────

#[test]
fn sequential_f16_ops_match_reference() {
    let backend = SequentialBackend;
    check_f16(&backend);
}

#[test]
fn moirai_f16_ops_match_reference() {
    let backend = MoiraiBackend;
    check_f16(&backend);
}

#[test]
fn sequential_bf16_ops_match_reference() {
    let backend = SequentialBackend;
    check_bf16(&backend);
}

#[test]
fn moirai_bf16_ops_match_reference() {
    let backend = MoiraiBackend;
    check_bf16(&backend);
}
