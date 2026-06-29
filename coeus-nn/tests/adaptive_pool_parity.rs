/// Adaptive pooling parity tests: AdaptiveAvgPool1d/2d, AdaptiveMaxPool1d/2d.
///
/// All results verified analytically against the region-based pooling formula.
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, Module,
};
use coeus_tensor::Tensor;

// ── AdaptiveAvgPool1d ─────────────────────────────────────────────────────────

#[test]
fn adaptive_avg_pool1d_output_size_equals_input() {
    // output_size = L → identity (each element is its own average)
    let m = AdaptiveAvgPool1d::<f64, SequentialBackend>::new(4);
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 4], &data),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 4]);
    assert_eq!(y.tensor.as_slice(), &data);
}

#[test]
fn adaptive_avg_pool1d_halves_length() {
    // [1, 1, 4] → [1, 1, 2]: region 0=[0,2), region 1=[2,4)
    let m = AdaptiveAvgPool1d::<f64, SequentialBackend>::new(2);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 4], &[2.0_f64, 4.0, 6.0, 8.0]),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 2]);
    assert_eq!(y.tensor.as_slice(), &[3.0, 7.0]); // (2+4)/2=3, (6+8)/2=7
}

#[test]
fn adaptive_avg_pool1d_global() {
    // output_size=1 → global average
    let m = AdaptiveAvgPool1d::<f64, SequentialBackend>::new(1);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(
            vec![1, 2, 4],
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 2, 1]);
    // channel 0: mean(1,2,3,4)=2.5, channel 1: mean(5,6,7,8)=6.5
    let s = y.tensor.as_slice();
    assert!((s[0] - 2.5).abs() < 1e-10, "ch0: got {}", s[0]);
    assert!((s[1] - 6.5).abs() < 1e-10, "ch1: got {}", s[1]);
}

// ── AdaptiveMaxPool1d ─────────────────────────────────────────────────────────

#[test]
fn adaptive_max_pool1d_halves_length() {
    // [1, 1, 4] → [1, 1, 2]: max of each pair
    let m = AdaptiveMaxPool1d::<f64, SequentialBackend>::new(2);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 4], &[1.0_f64, 3.0, 2.0, 4.0]),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 2]);
    assert_eq!(y.tensor.as_slice(), &[3.0, 4.0]);
}

#[test]
fn adaptive_max_pool1d_global() {
    // output_size=1 → global max
    let m = AdaptiveMaxPool1d::<f64, SequentialBackend>::new(1);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 5], &[2.0_f64, 7.0, 1.0, 5.0, 3.0]),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 1]);
    assert_eq!(y.tensor.as_slice(), &[7.0]);
}

// ── AdaptiveAvgPool2d ─────────────────────────────────────────────────────────

#[test]
fn adaptive_avg_pool2d_global_single_channel() {
    // [1, 1, 2, 2] → [1, 1, 1, 1]: global average
    let m = AdaptiveAvgPool2d::<f64, SequentialBackend>::square(1);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &[1.0_f64, 3.0, 5.0, 7.0]),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 1, 1]);
    assert!((y.tensor.as_slice()[0] - 4.0).abs() < 1e-10); // mean=4
}

#[test]
fn adaptive_avg_pool2d_halves_each_dim() {
    // [1, 1, 4, 4] → [1, 1, 2, 2]: 2x2 non-overlapping tiles
    let m = AdaptiveAvgPool2d::<f64, SequentialBackend>::new(2, 2);
    let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 2, 2]);
    let s = y.tensor.as_slice();
    // top-left 2x2: [1,2,5,6] → avg=3.5
    assert!((s[0] - 3.5).abs() < 1e-10, "tl: {}", s[0]);
    // top-right 2x2: [3,4,7,8] → avg=5.5
    assert!((s[1] - 5.5).abs() < 1e-10, "tr: {}", s[1]);
    // bottom-left 2x2: [9,10,13,14] → avg=11.5
    assert!((s[2] - 11.5).abs() < 1e-10, "bl: {}", s[2]);
    // bottom-right 2x2: [11,12,15,16] → avg=13.5
    assert!((s[3] - 13.5).abs() < 1e-10, "br: {}", s[3]);
}

// ── AdaptiveMaxPool2d ─────────────────────────────────────────────────────────

#[test]
fn adaptive_max_pool2d_halves_each_dim() {
    // [1, 1, 4, 4] → [1, 1, 2, 2]: max of each 2x2 tile
    let m = AdaptiveMaxPool2d::<f64, SequentialBackend>::new(2, 2);
    let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 2, 2]);
    let s = y.tensor.as_slice();
    // top-left 2x2: max([1,2,5,6])=6
    assert_eq!(s[0], 6.0);
    // top-right 2x2: max([3,4,7,8])=8
    assert_eq!(s[1], 8.0);
    // bottom-left 2x2: max([9,10,13,14])=14
    assert_eq!(s[2], 14.0);
    // bottom-right 2x2: max([11,12,15,16])=16
    assert_eq!(s[3], 16.0);
}

#[test]
fn adaptive_avg_pool2d_matches_global_avg_pool() {
    // output=(1,1) should produce the same result as GlobalAvgPool2d
    use coeus_nn::GlobalAvgPool2d;

    let m_adaptive = AdaptiveAvgPool2d::<f64, SequentialBackend>::square(1);
    let m_global = GlobalAvgPool2d::<f64, SequentialBackend>::new();
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 2, 2, 3], &data),
        false,
    );
    let ya = m_adaptive.forward(&x);
    let yg = m_global.forward(&x);
    assert_eq!(ya.tensor.shape(), &[1, 2, 1, 1]);
    assert_eq!(yg.tensor.shape(), &[1, 2, 1, 1]);
    for (a, b) in ya.tensor.as_slice().iter().zip(yg.tensor.as_slice().iter()) {
        assert!((a - b).abs() < 1e-10, "adaptive={a}, global={b}");
    }
}
