// ── Pool1d parity tests ──
//
// Verifies MaxPool1d and AvgPool1d forward outputs against hand-computed values.

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{AvgPool1d, MaxPool1d, Module};
use coeus_tensor::Tensor;

fn seq_var(shape: impl Into<coeus_core::Shape>, data: &[f32]) -> Var<f32, SequentialBackend> {
    Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(shape, data),
        false,
    )
}

#[test]
fn maxpool1d_no_pad() {
    // Input [1, 1, 6]: [1, 3, 2, 4, 1, 5], k=2, s=2, p=0
    // Windows: [1,3]=3, [2,4]=4, [1,5]=5  → output [3, 4, 5]
    let pool = MaxPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([1, 1, 6], &[1.0, 3.0, 2.0, 4.0, 1.0, 5.0]);
    let y = pool.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 3]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 3.0).abs() < 1e-6, "window 0");
    assert!((s[1] - 4.0).abs() < 1e-6, "window 1");
    assert!((s[2] - 5.0).abs() < 1e-6, "window 2");
}

#[test]
fn maxpool1d_stride1() {
    // Input [1, 1, 4]: [1, 4, 2, 3], k=2, s=1
    // Windows: [1,4]=4, [4,2]=4, [2,3]=3  → output [4, 4, 3]
    let pool = MaxPool1d::<f32, SequentialBackend>::with_params(2, 1, 0, 1);
    let x = seq_var([1, 1, 4], &[1.0, 4.0, 2.0, 3.0]);
    let y = pool.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 3]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 4.0).abs() < 1e-6);
    assert!((s[1] - 4.0).abs() < 1e-6);
    assert!((s[2] - 3.0).abs() < 1e-6);
}

#[test]
fn maxpool1d_multi_channel() {
    // Input [1, 2, 4], k=2, s=2
    // Ch0: [1,3]=3, [2,4]=4  → [3, 4]
    // Ch1: [5,7]=7, [6,8]=8  → [7, 8]
    let pool = MaxPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([1, 2, 4], &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
    let y = pool.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 2, 2]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 3.0).abs() < 1e-6, "ch0 w0");
    assert!((s[1] - 4.0).abs() < 1e-6, "ch0 w1");
    assert!((s[2] - 7.0).abs() < 1e-6, "ch1 w0");
    assert!((s[3] - 8.0).abs() < 1e-6, "ch1 w1");
}

#[test]
fn avgpool1d_no_pad() {
    // Input [1, 1, 4]: [1, 3, 2, 4], k=2, s=2
    // Windows: (1+3)/2=2.0, (2+4)/2=3.0
    let pool = AvgPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([1, 1, 4], &[1.0, 3.0, 2.0, 4.0]);
    let y = pool.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 2]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 2.0).abs() < 1e-6, "window 0");
    assert!((s[1] - 3.0).abs() < 1e-6, "window 1");
}

#[test]
fn avgpool1d_stride1() {
    // Input [1, 1, 4]: [2, 4, 6, 8], k=3, s=1
    // Windows: (2+4+6)/3=4.0, (4+6+8)/3=6.0
    let pool = AvgPool1d::<f32, SequentialBackend>::with_params(3, 1, 0, 1);
    let x = seq_var([1, 1, 4], &[2.0, 4.0, 6.0, 8.0]);
    let y = pool.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 2]);
    let s = y.tensor.as_slice();
    assert!((s[0] - 4.0).abs() < 1e-6, "window 0");
    assert!((s[1] - 6.0).abs() < 1e-6, "window 1");
}

#[test]
fn avgpool1d_batch() {
    // Input [2, 1, 4], k=2, s=2: batch of two independent rows
    let pool = AvgPool1d::<f32, SequentialBackend>::new(2);
    let x = seq_var([2, 1, 4], &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
    let y = pool.forward(&x);
    assert_eq!(y.tensor.shape(), &[2, 1, 2]);
    let s = y.tensor.as_slice();
    // batch0: 2.0, 6.0
    assert!((s[0] - 2.0).abs() < 1e-6);
    assert!((s[1] - 6.0).abs() < 1e-6);
    // batch1: 3.0, 7.0
    assert!((s[2] - 3.0).abs() < 1e-6);
    assert!((s[3] - 7.0).abs() < 1e-6);
}
