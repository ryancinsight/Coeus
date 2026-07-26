//! Verification for the differentiable selective scan (Mamba/S6 recurrence).
//!
//! Two independent oracles:
//!   * Value: a small `L = 3`, `state_dim = 2` recurrence unrolled by hand. The
//!     expected `h_t` follow directly from `h_t = a_bar_t ⊙ h_{t-1} + u_t`
//!     (`h_0 = u_0`), independent of the implementation's loop structure.
//!   * Gradient: central finite differences w.r.t. EACH input (`a_bar`, `u`).
//!     `loss = sum(h)` is a *linear* function of each individual input scalar
//!     (holding the others fixed) — `h_t` is linear in each earlier `a_bar` and
//!     every `u`, and the sum preserves linearity — so the central-difference
//!     truncation error is exactly zero and the only residual is `f32` round-off
//!     `~ eps * |loss| / h`. With `|loss| < ~10`, `eps_f32 = 1.19e-7`, and
//!     `h = 1/64`, that floor is `~ 1.19e-7 * 10 / (1/64) ~= 7.6e-5`; the `1e-3`
//!     tolerance is that floor with a >10x margin for accumulation across the
//!     length-3 temporal reduction. `length = 3` exercises the full temporal
//!     gradient chain (`a_bar` at `t` feeds every later `h`).

use coeus_autograd::{selective_scan, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn value_matches_hand_unrolled_recurrence() {
    let backend = MoiraiBackend;
    // Shape [batch=1, length=3, state=2]; two independent channels along axis 2.
    // channel 0: a = 0.5, 0.8, 0.3 ; u = 1.0, 2.0, -0.5
    //   h0 = 1.0 ; h1 = 0.8*1.0 + 2.0 = 2.8 ; h2 = 0.3*2.8 - 0.5 = 0.34
    // channel 1: a = 0.1, 0.2, 0.9 ; u = -1.0, 0.5, 1.5
    //   h0 = -1.0 ; h1 = 0.2*(-1.0) + 0.5 = 0.3 ; h2 = 0.9*0.3 + 1.5 = 1.77
    let a_data = [0.5, 0.1, 0.8, 0.2, 0.3, 0.9];
    let u_data = [1.0, -1.0, 2.0, 0.5, -0.5, 1.5];
    let expected = [1.0f32, -1.0, 2.8, 0.3, 0.34, 1.77];

    let a_bar = Var::new(Tensor::from_slice_on([1, 3, 2], &a_data, &backend), false);
    let u = Var::new(Tensor::from_slice_on([1, 3, 2], &u_data, &backend), false);
    let h = selective_scan(&a_bar, &u);

    let got = h.tensor.as_slice();
    for (i, (&value, &want)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (value - want).abs() <= 1e-6,
            "h[{i}]: got {value}, expected {want}"
        );
    }
}

// Shape [batch=2, length=3, state=2] — batch > 1 exercises the outer flattening.
const SHAPE: [usize; 3] = [2, 3, 2];
const A_DATA: [f32; 12] = [
    0.5, -0.4, 0.8, 0.2, 0.3, -0.9, 0.6, 0.1, -0.7, 0.4, 0.9, -0.2,
];
const U_DATA: [f32; 12] = [
    1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.3, -0.8, 1.2, -0.6, 0.7, -1.1,
];

/// Scalar loss `sum(selective_scan(a_bar, u))` with grad tracking off.
fn loss(a_data: &[f32], u_data: &[f32]) -> f64 {
    let backend = MoiraiBackend;
    let a_bar = Var::new(Tensor::from_slice_on(SHAPE, a_data, &backend), false);
    let u = Var::new(Tensor::from_slice_on(SHAPE, u_data, &backend), false);
    let h = selective_scan(&a_bar, &u);
    h.tensor.as_slice().iter().map(|&v| f64::from(v)).sum()
}

#[test]
fn a_bar_gradient_matches_central_difference() {
    let backend = MoiraiBackend;
    let a_bar = Var::new(Tensor::from_slice_on(SHAPE, &A_DATA, &backend), true);
    let u = Var::new(Tensor::from_slice_on(SHAPE, &U_DATA, &backend), true);
    let h = selective_scan(&a_bar, &u);
    sum(&h).backward();
    let analytic = a_bar.grad().expect("tracked a_bar gradient");
    let analytic = analytic.as_slice();

    let step = 1.0f64 / 64.0;
    for i in 0..A_DATA.len() {
        let mut plus = A_DATA;
        let mut minus = A_DATA;
        plus[i] += step as f32;
        minus[i] -= step as f32;
        let fd = (loss(&plus, &U_DATA) - loss(&minus, &U_DATA)) / (2.0 * step);
        assert!(
            (f64::from(analytic[i]) - fd).abs() <= 1e-3,
            "a_bar[{i}]: analytic {}, finite-diff {fd}",
            analytic[i]
        );
    }
}

#[test]
fn u_gradient_matches_central_difference() {
    let backend = MoiraiBackend;
    let a_bar = Var::new(Tensor::from_slice_on(SHAPE, &A_DATA, &backend), true);
    let u = Var::new(Tensor::from_slice_on(SHAPE, &U_DATA, &backend), true);
    let h = selective_scan(&a_bar, &u);
    sum(&h).backward();
    let analytic = u.grad().expect("tracked u gradient");
    let analytic = analytic.as_slice();

    let step = 1.0f64 / 64.0;
    for i in 0..U_DATA.len() {
        let mut plus = U_DATA;
        let mut minus = U_DATA;
        plus[i] += step as f32;
        minus[i] -= step as f32;
        let fd = (loss(&A_DATA, &plus) - loss(&A_DATA, &minus)) / (2.0 * step);
        assert!(
            (f64::from(analytic[i]) - fd).abs() <= 1e-3,
            "u[{i}]: analytic {}, finite-diff {fd}",
            analytic[i]
        );
    }
}
