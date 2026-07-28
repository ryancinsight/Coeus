/// Unfold1d/Fold1d/Unfold2d/Fold2d parity tests.
///
/// Verified against PyTorch `nn.Unfold` / `nn.Fold` semantics.
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{Fold1d, Fold2d, Module, Unfold1d, Unfold2d};
use coeus_tensor::Tensor;

// ── Unfold1d ─────────────────────────────────────────────────────────────────

#[test]
fn unfold1d_output_shape() {
    // [1, 2, 6], kernel=3, stride=1, padding=0, dilation=1 → [1, 6, 4]
    let m = Unfold1d::<f32, SequentialBackend>::new(3, 1, 0, 1);
    let x = Var::new(Tensor::<f32, SequentialBackend>::ones(vec![1, 2, 6]), false);
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 6, 4]);
}

#[test]
fn unfold1d_identity_kernel1_stride1() {
    // kernel_size=1, stride=1, no padding: unfold is identity reshape.
    // Input [1, 2, 4] → output [1, 2, 4] with C*k=2, L_out=4.
    let m = Unfold1d::<f64, SequentialBackend>::new(1, 1, 0, 1);
    let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 2, 4], &data),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 2, 4]);
    // Each C*k slot is a single element — data should pass through unchanged.
    assert_eq!(y.tensor.as_slice(), data.as_slice());
}

#[test]
fn unfold1d_values_kernel3() {
    // Input [1, 1, 5], kernel=3, stride=1, padding=0, dilation=1
    // Positions: [0,1,2], [1,2,3], [2,3,4]
    // Output shape: [1, 3, 3]
    let data = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let m = Unfold1d::<f64, SequentialBackend>::new(3, 1, 0, 1);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 5], &data),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 3, 3]);
    let s = y.tensor.as_slice();
    // channel 0, kernel 0: [10,20,30]
    assert_eq!(s[0], 10.0);
    assert_eq!(s[1], 20.0);
    assert_eq!(s[2], 30.0);
    // channel 0, kernel 1: [20,30,40]
    assert_eq!(s[3], 20.0);
    assert_eq!(s[4], 30.0);
    assert_eq!(s[5], 40.0);
    // channel 0, kernel 2: [30,40,50]
    assert_eq!(s[6], 30.0);
    assert_eq!(s[7], 40.0);
    assert_eq!(s[8], 50.0);
}

// ── Fold1d ────────────────────────────────────────────────────────────────────

#[test]
fn fold1d_output_shape() {
    // Reverse of unfold1d_output_shape: [1, 6, 4] → [1, 2, 6]
    let m = Fold1d::<f32, SequentialBackend>::new(6, 3, 1, 0, 1);
    let x = Var::new(Tensor::<f32, SequentialBackend>::ones(vec![1, 6, 4]), false);
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 2, 6]);
}

#[test]
fn fold1d_unfold1d_roundtrip_no_overlap() {
    // Stride = kernel_size means no overlap, so fold(unfold(x)) == count * x
    // (count = 1 for no overlap).
    let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let x_orig = Tensor::<f64, SequentialBackend>::from_slice(vec![1, 2, 4], &data);
    let x = Var::new(x_orig.clone(), false);

    let unfold = Unfold1d::<f64, SequentialBackend>::new(2, 2, 0, 1);
    let fold = Fold1d::<f64, SequentialBackend>::new(4, 2, 2, 0, 1);

    let unfolded = unfold.forward(&x);
    assert_eq!(unfolded.tensor.shape(), &[1, 4, 2]); // C*k=4, L_out=2
    let refolded = fold.forward(&unfolded);
    assert_eq!(refolded.tensor.shape(), &[1, 2, 4]);

    for (a, &e) in refolded.tensor.as_slice().iter().zip(data.iter()) {
        assert!((*a - e).abs() < 1e-10, "roundtrip: got {a}, expected {e}");
    }
}

// ── Unfold2d ──────────────────────────────────────────────────────────────────

#[test]
fn unfold2d_output_shape() {
    // [1, 2, 4, 4], kernel=2, stride=2, no padding, no dilation → [1, 8, 4]
    let m = Unfold2d::<f32, SequentialBackend>::new(2, 2, 0, 1);
    let x = Var::new(
        Tensor::<f32, SequentialBackend>::ones(vec![1, 2, 4, 4]),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 8, 4]); // C*kH*kW=2*4=8, H_out*W_out=2*2=4
}

#[test]
fn unfold2d_values_single_window() {
    // [1, 1, 2, 2], kernel=2 → one window: the whole 2x2 patch.
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let m = Unfold2d::<f64, SequentialBackend>::new(2, 1, 0, 1);
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &data),
        false,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 4, 1]); // C*kH*kW=4, L_out=1
    let s = y.tensor.as_slice();
    assert_eq!(s, &[1.0, 2.0, 3.0, 4.0]);
}

// ── Fold2d ────────────────────────────────────────────────────────────────────

#[test]
fn fold2d_output_shape() {
    // Reverse: [1, 8, 4] → [1, 2, 4, 4]
    let m = Fold2d::<f32, SequentialBackend>::new(4, 4, 2, 2, 0, 1);
    let x = Var::new(Tensor::<f32, SequentialBackend>::ones(vec![1, 8, 4]), false);
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 2, 4, 4]);
}

#[test]
fn fold2d_unfold2d_roundtrip_no_overlap() {
    // Stride = kernel_size, no overlap → fold(unfold(x)) == x.
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &data),
        false,
    );

    let unfold = Unfold2d::<f64, SequentialBackend>::new(2, 2, 0, 1);
    let fold = Fold2d::<f64, SequentialBackend>::new(4, 4, 2, 2, 0, 1);

    let unfolded = unfold.forward(&x);
    let refolded = fold.forward(&unfolded);
    assert_eq!(refolded.tensor.shape(), &[1, 1, 4, 4]);

    for (a, &e) in refolded.tensor.as_slice().iter().zip(data.iter()) {
        assert!((*a - e).abs() < 1e-10, "roundtrip: got {a}, expected {e}");
    }
}

// ── Unfold1d differentiability (G-045) ───────────────────────────────────────

#[test]
fn unfold1d_backward_accumulates_window_overlap() {
    // Unfold (im2col) is linear, so d sum(unfold(x)) / d x_i equals the number of
    // windows containing position i. kernel=3, stride=1 on length 5 yields the
    // overlap counts [1, 2, 3, 2, 1]; the col2im (fold1d) backward must produce
    // exactly that (previously this layer was forward-only with dx=0).
    let m = Unfold1d::<f64, SequentialBackend>::new(3, 1, 0, 1);
    let data = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 5], &data),
        true,
    );
    m.forward(&x)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().expect("unfold1d input gradient");
    assert_eq!(grad.as_slice(), &[1.0, 2.0, 3.0, 2.0, 1.0]);
    assert!(
        grad.as_slice().iter().any(|&g| g > 0.0),
        "unfold1d gradient is all-zero — backward not propagating"
    );
}

#[test]
fn unfold2d_backward_accumulates_window_overlap() {
    // 2x2 kernel, stride 1 on [1,1,3,3]: the 2D window-overlap counts are the
    // outer product of the per-axis [1,2,1] counts → [[1,2,1],[2,4,2],[1,2,1]].
    let m = Unfold2d::<f64, SequentialBackend>::new(2, 1, 0, 1);
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 1, 3, 3], &data),
        true,
    );
    m.forward(&x)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().expect("unfold2d input gradient");
    assert_eq!(
        grad.as_slice(),
        &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
    );
}

#[test]
fn fold1d_backward_is_im2col_of_ones() {
    // Fold (col2im) is linear; with non-overlapping tiles (stride=kernel, no pad)
    // each input column maps to exactly one output position, so
    // d sum(fold(x))/dx = 1 everywhere (= unfold1d of the all-ones output grad).
    let m = Fold1d::<f64, SequentialBackend>::new(6, 2, 2, 0, 1);
    let data: Vec<f64> = (1..=6).map(|i| i as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 2, 3], &data),
        true,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 6]);
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        x.grad().expect("fold1d input gradient").as_slice(),
        &[1.0; 6]
    );
}

#[test]
fn fold2d_backward_is_im2col_of_ones() {
    let m = Fold2d::<f64, SequentialBackend>::new(4, 4, 2, 2, 0, 1);
    let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(vec![1, 4, 4], &data),
        true,
    );
    let y = m.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 1, 4, 4]);
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        x.grad().expect("fold2d input gradient").as_slice(),
        &[1.0; 16]
    );
}
