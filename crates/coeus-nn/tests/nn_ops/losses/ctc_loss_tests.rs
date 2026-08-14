#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
/// CTC (Connectionist Temporal Classification) Loss tests.
///
/// Analytical oracles derived from the log-space forward-backward DP.
///
/// # Test case 1: single frame, single label
/// T=1, N=1, C=3 (blank=0, labels: 1, 2)
/// log_probs = [[log(0.1), log(0.6), log(0.3)]]
/// target = [1], input_len=[1], target_len=[1]
/// Extended: [blank=0, 1, blank=0] → ls=3
/// α(0,0)=log(0.1), α(0,1)=log(0.6), α(0,2)=-inf (need 2 frames for ls=3)
/// Wait — with T=1 and ls=3, actually: init only α[0]=log p(blank), α[1]=log p(label)
/// log P = log_sum_exp(α[0, ls-1], α[0, ls-2]) = log_sum_exp(-inf, log(0.6)) = log(0.6)
/// CTC loss = -log(0.6)
///
/// # Test case 2: gradient propagates
use coeus_autograd::{ctc_loss, log_softmax, Var};
use coeus_core::SequentialBackend;
use coeus_nn::ctc_loss as nn_ctc_loss;
use coeus_tensor::Tensor;

type B = SequentialBackend;

fn t3(data: &[f64], shape: [usize; 3]) -> Var<f64, B> {
    Var::new(Tensor::<f64, B>::from_slice(shape.to_vec(), data), true)
}

fn lp(data: &[f64], shape: [usize; 3]) -> Var<f64, B> {
    Var::new(Tensor::<f64, B>::from_slice(shape.to_vec(), data), false)
}

fn get_loss_val(v: &Var<f64, B>) -> f64 {
    v.tensor.as_slice()[0]
}

#[test]
fn ctc_loss_single_frame_single_label() {
    // T=1, N=1, C=3, blank=0, target=[1]
    // log_probs: ln(0.1), ln(0.6), ln(0.3)
    // log P = ln(0.6) → CTC loss = -ln(0.6)
    let x = lp(&[0.1_f64.ln(), 0.6_f64.ln(), 0.3_f64.ln()], [1, 1, 3]);
    let loss = ctc_loss(&x, &[1usize], &[1], &[1], 0);
    let expected = -0.6_f64.ln();
    assert!(
        (get_loss_val(&loss) - expected).abs() < 1e-10,
        "CTC single frame: got {:.10}, expected {:.10}",
        get_loss_val(&loss),
        expected
    );
}

#[test]
fn ctc_loss_two_frames_single_label() {
    // T=2, N=1, C=3, blank=0, target=[1]
    // Extended: [0, 1, 0], ls=3
    // α(0,0)=ln0.5, α(0,1)=ln0.3, α(0,2)=-inf
    // α(1,0) = ln0.5 + ln0.4 = ln(0.2)
    // α(1,1) = log_sum_exp(ln0.3, ln0.5) + ln0.4 = ln(0.8) + ln0.4 = ln(0.32)
    // α(1,2) = log_sum_exp(-inf, ln0.3) + ln0.4 = ln(0.12)
    //   (ext[2]=0==ext[0]=0 so no skip)
    // log P = log_sum_exp(ln0.12, ln0.32) = ln(0.44)
    let data = [
        0.5_f64.ln(),
        0.3_f64.ln(),
        0.2_f64.ln(),
        0.4_f64.ln(),
        0.4_f64.ln(),
        0.2_f64.ln(),
    ];
    let x = lp(&data, [2, 1, 3]);
    let loss = ctc_loss(&x, &[1usize], &[2], &[1], 0);
    let expected = -(0.44_f64.ln());
    assert!(
        (get_loss_val(&loss) - expected).abs() < 1e-8,
        "CTC two frames: got {:.10}, expected {:.10}",
        get_loss_val(&loss),
        expected
    );
}

#[test]
fn ctc_loss_batch_two_samples() {
    // Two samples, mean loss = -ln(0.6)
    // Sample 0: target=[1], p(1)=0.6 → loss=-ln(0.6)
    // Sample 1: target=[2], p(2)=0.6 → loss=-ln(0.6)
    let data = [
        0.1_f64.ln(),
        0.6_f64.ln(),
        0.3_f64.ln(), // frame0, sample0
        0.1_f64.ln(),
        0.3_f64.ln(),
        0.6_f64.ln(), // frame0, sample1
    ];
    let x = lp(&data, [1, 2, 3]);
    let loss = ctc_loss(&x, &[1usize, 2], &[1, 1], &[1, 1], 0);
    let expected = -0.6_f64.ln();
    assert!(
        (get_loss_val(&loss) - expected).abs() < 1e-10,
        "CTC batch: got {:.10}, expected {:.10}",
        get_loss_val(&loss),
        expected
    );
}

#[test]
fn ctc_loss_backward_runs() {
    // Verify gradients propagate through log_softmax -> ctc_loss.
    let logits = t3(&[1.0, 2.0, 0.5, 0.8, 1.5, 0.3], [2, 1, 3]);
    let log_probs = log_softmax(&logits, 2);
    let loss = ctc_loss(&log_probs, &[1usize], &[2], &[1], 0);
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(
        logits.grad().is_some(),
        "gradient must propagate through ctc_loss"
    );
    let grad = logits.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 1, 3]);
    let nonzero = grad.as_slice().iter().any(|&v: &f64| v.abs() > 1e-15);
    assert!(nonzero, "CTC gradient must have at least one nonzero entry");
}

#[test]
fn nn_ctc_loss_matches_autograd() {
    let data = [0.1_f64.ln(), 0.6_f64.ln(), 0.3_f64.ln()];
    let x1 = lp(&data, [1, 1, 3]);
    let x2 = lp(&data, [1, 1, 3]);
    let l1 = ctc_loss(&x1, &[1usize], &[1], &[1], 0);
    let l2 = nn_ctc_loss(&x2, &[1usize], &[1], &[1], 0);
    let v1 = get_loss_val(&l1);
    let v2 = get_loss_val(&l2);
    assert_eq!(v1, v2, "nn and autograd ctc_loss must match: {v1} != {v2}");
}
