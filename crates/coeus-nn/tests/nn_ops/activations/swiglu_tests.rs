// ── SwiGLU gated feed-forward unit: value-semantic tests ──
//
// `SwiGlu::new` builds two ones-initialised Linear layers, so each projection
// maps a row to its element sum: inner(row) = outer(row) = S = sum(row). With
// no bias the forward is therefore the closed form
//     SwiGLU(row)[j] = silu(S) * S   for every output column j,
// which we assert exactly (analytic oracle), plus the parameter inventory and
// gradient flow through the composed silu/mul/matmul graph.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{Module, SwiGlu};
use coeus_tensor::Tensor;

/// Reference SiLU: x · sigmoid(x).
fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn assert_close_slice(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (&g, &e) in got.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() <= tol,
            "{label}: expected {e} got {g} (diff {:.3e})",
            (g - e).abs()
        );
    }
}

#[test]
fn swiglu_forward_matches_analytic() {
    // ones-weight projection ⇒ each output column equals the input row sum S.
    // Two rows with sums 6 and 1; d_output = 2 (both columns share S).
    let (d_input, d_output) = (3usize, 2usize);
    let data = vec![1.0_f64, 2.0, 3.0, 0.0, 1.0, 0.0]; // [2 × 3], row sums 6, 1
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, d_input], &data),
        true,
    );

    let swiglu = SwiGlu::<f64, MoiraiBackend>::new(d_input, d_output, false);
    let output = swiglu.forward(&input).expect("valid SwiGLU input");

    // Tolerance: the only transcendental is silu's sigmoid (~1 ulp); the result
    // magnitude is ~36, so the analytic error is O(1e-14). 1e-10 is a safe
    // margin well below any real divergence.
    let (s0, s1) = (6.0_f64, 1.0_f64);
    let (v0, v1) = (silu_f64(s0) * s0, silu_f64(s1) * s1);
    let expected = vec![v0, v0, v1, v1]; // [2 × 2], row-major
    assert_close_slice("swiglu_forward", output.tensor.as_slice(), &expected, 1e-10);
}

#[test]
fn swiglu_parameter_inventory() {
    // No bias: two weight matrices. With bias: two weights + two bias vectors.
    let no_bias = SwiGlu::<f64, MoiraiBackend>::new(4, 8, false);
    assert_eq!(no_bias.parameters().len(), 2);
    let with_bias = SwiGlu::<f64, MoiraiBackend>::new(4, 8, true);
    assert_eq!(with_bias.parameters().len(), 4);
}

#[test]
fn swiglu_backward_populates_parameter_grads() {
    let swiglu = SwiGlu::<f64, MoiraiBackend>::new(3, 2, true);
    let data = vec![0.5_f64, -1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &data),
        true,
    );

    let output = swiglu.forward(&input).expect("valid SwiGLU input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    for (i, p) in swiglu.parameters().iter().enumerate() {
        let grad = p
            .grad()
            .unwrap_or_else(|| panic!("param {i} missing gradient"));
        assert_eq!(
            grad.as_slice().len(),
            p.tensor.as_slice().len(),
            "param {i}: gradient shape mismatch"
        );
        assert!(
            grad.as_slice().iter().all(|x: &f64| x.is_finite()),
            "param {i}: gradient has non-finite entries"
        );
    }
}
