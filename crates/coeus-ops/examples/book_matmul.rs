//! Matrix multiplication via `coeus_ops::matmul`.
//!
//! `matmul(a, b, backend)` computes the generalized matrix product for 2-D
//! and batched N-D inputs.  The backend is a zero-sized unit struct; switching
//! from [`SequentialBackend`] to [`MoiraiBackend`] or a GPU backend changes
//! the execution policy without changing the call site.
#![expect(
    clippy::print_stdout,
    reason = "ratchet COEUS-LINT-1: demonstration/diagnostic output"
)]

use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_ops::matmul;
use coeus_tensor::Tensor;

fn main() {
    // ── 2×3 × 3×2 → 2×2 ──
    let backend = SequentialBackend::new();
    let a =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::<f32, SequentialBackend>::from_slice(
        [3, 2],
        &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    let c = matmul(&a, &b, &backend);
    assert_eq!(c.shape(), &[2, 2]);
    let expected = [58.0_f32, 64.0, 139.0, 154.0];
    for (got, want) in c.as_slice().iter().zip(&expected) {
        assert!((got - want).abs() < 1e-4, "matmul mismatch: {got} ≠ {want}");
    }
    println!("A × B (2×3 · 3×2) = {:?}", c.as_slice());

    // ── Identity matrix property: A × I = A ──
    let identity = Tensor::<f32, SequentialBackend>::from_slice(
        [3, 3],
        &[1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let a3: Tensor<f32, SequentialBackend> =
        Tensor::from_slice([2, 3], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = matmul(&a3, &identity, &backend);
    for (got, want) in result.as_slice().iter().zip(a3.as_slice()) {
        assert!((got - want).abs() < 1e-6, "A×I should equal A");
    }
    println!("A × I = {:?}", result.as_slice());

    // ── Same result on MoiraiBackend ──
    let m_backend = MoiraiBackend;
    let ma = Tensor::<f32, MoiraiBackend>::from_slice_on(
        vec![2, 3],
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &m_backend,
    );
    let mb = Tensor::<f32, MoiraiBackend>::from_slice_on(
        vec![3, 2],
        &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        &m_backend,
    );
    let mc = matmul(&ma, &mb, &m_backend);
    for (got, want) in mc.as_slice().iter().zip(&expected) {
        assert!((got - want).abs() < 1e-4);
    }
    println!("same result on MoiraiBackend: {:?}", mc.as_slice());

    println!("all matmul assertions passed");
}
