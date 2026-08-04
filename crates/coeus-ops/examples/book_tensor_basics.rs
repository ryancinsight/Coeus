//! Tensor construction and scalar reductions.
//!
//! [`MoiraiBackend`] is the moirai-integrated CPU execution backend.
//! [`Tensor::from_slice_on`] creates a dense tensor from a host slice;
//! [`coeus_ops::sum`] and [`coeus_ops::mean`] reduce over all elements.

use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

fn main() {
    let backend = MoiraiBackend;

    // ── 1-D tensor from a known slice ──
    let a: Tensor<f64, MoiraiBackend> =
        Tensor::from_slice_on(vec![5], &[1.0_f64, 2.0, 3.0, 4.0, 5.0], &backend);
    assert_eq!(a.shape(), &[5]);
    println!("a = {:?}", a.as_slice());

    // ── Scalar sum and mean ──
    let s = coeus_ops::sum(&a, &backend).expect("sum");
    let m = coeus_ops::mean(&a, &backend).expect("mean");
    println!("sum  = {s}");  // 15.0
    println!("mean = {m}");  // 3.0
    assert!((s - 15.0).abs() < 1e-10);
    assert!((m - 3.0).abs() < 1e-10);

    // ── 2-D tensor (3 × 2) ──
    let mat: Tensor<f64, MoiraiBackend> =
        Tensor::from_slice_on(vec![3, 2], &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    assert_eq!(mat.shape(), &[3, 2]);
    let mat_sum = coeus_ops::sum(&mat, &backend).expect("mat sum");
    println!("3×2 matrix sum = {mat_sum}");  // 21.0
    assert!((mat_sum - 21.0).abs() < 1e-10);

    println!("all tensor-basics assertions passed");
}
