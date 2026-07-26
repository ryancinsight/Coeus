//! Differential verification of `einsum` and `einsum3`.
//!
//! All test inputs are small integer-valued matrices so the reference values
//! are exact (no floating-point rounding); tolerances are therefore 0.0 for
//! all integer-computable references and 1e-5 for cases involving a sqrt.
//!
//! Patterns exercised:
//!   "ij,jk->ik"    — standard matrix multiply
//!   "ij->ji"        — 2-D transpose
//!   "ii->"          — trace (sum of diagonal)
//!   "i,i->"         — 1-D dot product
//!   "i,j->ij"       — outer product
//!   "ij,j->i"       — matrix-vector multiply
//!   einsum3 "ij,jk,kl->il" — triple matmul chain
//!
//! SequentialBackend and MoiraiBackend are verified against the same
//! analytical references.  Divergence between them indicates a backend bug.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn from_slice_f32<B>(shape: &[usize], data: &[f32], backend: &B) -> Tensor<f32, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn assert_exact(got: &[f32], expected: &[f32], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(g, e, "{context}[{i}]: got {g}, expected {e}");
    }
}

// ── EINSUM ────────────────────────────────────────────────────────────────

fn check_einsum<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    // "ij,jk->ik" — standard [2×2] matmul.
    // A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    // C[0,0]=1·5+2·7=19, C[0,1]=1·6+2·8=22, C[1,0]=3·5+4·7=43, C[1,1]=3·6+4·8=50
    let a = from_slice_f32(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0], backend);
    let b = from_slice_f32(&[2, 2], &[5.0f32, 6.0, 7.0, 8.0], backend);
    let c = coeus_ops::einsum("ij,jk->ik", &[&a, &b], backend).expect("valid einsum test shapes");
    assert_eq!(c.shape(), &[2, 2], "matmul shape");
    assert_exact(c.as_slice(), &[19.0f32, 22.0, 43.0, 50.0], "einsum matmul");

    // "ij->ji" — [2×3] transpose.
    // A=[[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]] shape [3,2]
    let a2 = from_slice_f32(&[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let t = coeus_ops::einsum("ij->ji", &[&a2], backend).expect("valid einsum test shapes");
    assert_eq!(t.shape(), &[3, 2], "transpose shape");
    assert_exact(
        t.as_slice(),
        &[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0],
        "einsum transpose",
    );

    // "ii->" — trace of [[1,2,3],[4,5,6],[7,8,9]].
    // trace = 1+5+9 = 15
    let sq = from_slice_f32(
        &[3, 3],
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        backend,
    );
    let tr = coeus_ops::einsum("ii->", &[&sq], backend).expect("valid einsum test shapes");
    assert_eq!(tr.shape(), &[1], "trace shape");
    assert_exact(tr.as_slice(), &[15.0f32], "einsum trace");

    // "i,i->" — 1-D dot product.
    // a=[1,2,3], b=[4,5,6]: dot = 4+10+18 = 32
    let v1 = from_slice_f32(&[3], &[1.0f32, 2.0, 3.0], backend);
    let v2 = from_slice_f32(&[3], &[4.0f32, 5.0, 6.0], backend);
    let dot = coeus_ops::einsum("i,i->", &[&v1, &v2], backend).expect("valid einsum test shapes");
    assert_eq!(dot.shape(), &[1], "dot shape");
    assert_exact(dot.as_slice(), &[32.0f32], "einsum dot");

    // "i,j->ij" — outer product.
    // a=[1,2], b=[3,4,5]: [[3,4,5],[6,8,10]] shape [2,3]
    let u1 = from_slice_f32(&[2], &[1.0f32, 2.0], backend);
    let u2 = from_slice_f32(&[3], &[3.0f32, 4.0, 5.0], backend);
    let outer =
        coeus_ops::einsum("i,j->ij", &[&u1, &u2], backend).expect("valid einsum test shapes");
    assert_eq!(outer.shape(), &[2, 3], "outer shape");
    assert_exact(
        outer.as_slice(),
        &[3.0f32, 4.0, 5.0, 6.0, 8.0, 10.0],
        "einsum outer",
    );

    // "ij,j->i" — matrix-vector multiply.
    // A=[[1,2],[3,4]], x=[5,6]: y=[1·5+2·6, 3·5+4·6]=[17, 39]
    let mat = from_slice_f32(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0], backend);
    let vec_x = from_slice_f32(&[2], &[5.0f32, 6.0], backend);
    let mv =
        coeus_ops::einsum("ij,j->i", &[&mat, &vec_x], backend).expect("valid einsum test shapes");
    assert_eq!(mv.shape(), &[2], "mat-vec shape");
    assert_exact(mv.as_slice(), &[17.0f32, 39.0], "einsum mat-vec");
}

#[test]
fn sequential_einsum_matches_reference() {
    let backend = SequentialBackend;
    check_einsum(&backend);
}

#[test]
fn moirai_einsum_matches_reference() {
    let backend = MoiraiBackend;
    check_einsum(&backend);
}

// ── EINSUM3 ───────────────────────────────────────────────────────────────

fn check_einsum3<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    // "ij,jk,kl->il" — triple [2×2] matmul chain.
    // A=[[1,2],[3,4]], B=I₂, C=2·I₂ → A@I@2I = 2A = [[2,4],[6,8]]
    let a = from_slice_f32(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0], backend);
    let b = from_slice_f32(&[2, 2], &[1.0f32, 0.0, 0.0, 1.0], backend);
    let c = from_slice_f32(&[2, 2], &[2.0f32, 0.0, 0.0, 2.0], backend);
    let result = coeus_ops::einsum3("ij,jk,kl->il", &a, &b, &c, backend)
        .expect("valid three-operand einsum test shapes");
    assert_eq!(result.shape(), &[2, 2], "einsum3 shape");
    assert_exact(
        result.as_slice(),
        &[2.0f32, 4.0, 6.0, 8.0],
        "einsum3 A@I@2I",
    );

    // Non-trivial chain: A=[[1,0],[0,2]], B=[[3,1],[0,1]], C=[[1,0],[0,1]]
    // A@B = [[3,1],[0,2]], (A@B)@C = [[3,1],[0,2]] (C=I, so unchanged)
    let a2 = from_slice_f32(&[2, 2], &[1.0f32, 0.0, 0.0, 2.0], backend);
    let b2 = from_slice_f32(&[2, 2], &[3.0f32, 1.0, 0.0, 1.0], backend);
    let c2 = from_slice_f32(&[2, 2], &[1.0f32, 0.0, 0.0, 1.0], backend);
    let res2 = coeus_ops::einsum3("ij,jk,kl->il", &a2, &b2, &c2, backend)
        .expect("valid three-operand einsum test shapes");
    assert_eq!(res2.shape(), &[2, 2], "einsum3 non-trivial shape");
    assert_exact(
        res2.as_slice(),
        &[3.0f32, 1.0, 0.0, 2.0],
        "einsum3 diag@upper@I",
    );
}

#[test]
fn sequential_einsum3_matches_reference() {
    let backend = SequentialBackend;
    check_einsum3(&backend);
}

#[test]
fn moirai_einsum3_matches_reference() {
    let backend = MoiraiBackend;
    check_einsum3(&backend);
}
