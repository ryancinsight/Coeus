// ── F16 half-precision smoke tests ──
//
// Verifies that coeus-ops and coeus-tensor support eunomia F16 as a Scalar
// type, enabling training in half-precision for memory-constrained use cases.

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use eunomia::F16;

/// Half-precision tensors can be constructed and undergo basic arithmetic.
#[test]
fn f16_tensor_add_smoke() {
    let a = Tensor::<F16, SequentialBackend>::from_slice(
        vec![4],
        &[
            F16::from_f32(1.0),
            F16::from_f32(2.0),
            F16::from_f32(3.0),
            F16::from_f32(4.0),
        ],
    );
    let b = Tensor::<F16, SequentialBackend>::from_slice(vec![4], &[F16::from_f32(0.5); 4]);
    let backend = SequentialBackend::new();
    let c = coeus_ops::add(&a, &b, &backend);
    let expected = [1.5f32, 2.5, 3.5, 4.5];
    for (got, want) in c.as_slice().iter().zip(expected.iter()) {
        let diff = (got.to_f32() - want).abs();
        assert!(
            diff < 0.01,
            "F16 add: got {:.4} want {:.4}",
            got.to_f32(),
            want
        );
    }
}

/// Half-precision matmul is computed via F16 scalar ops.
#[test]
fn f16_matmul_smoke() {
    let a_data: Vec<F16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| F16::from_f32(v))
        .collect();
    let b_data: Vec<F16> = [1.0f32, 0.0, 0.0, 1.0]
        .iter()
        .map(|&v| F16::from_f32(v))
        .collect();
    let a = Tensor::<F16, SequentialBackend>::from_slice(vec![2, 2], &a_data);
    let b = Tensor::<F16, SequentialBackend>::from_slice(vec![2, 2], &b_data);
    let backend = SequentialBackend::new();
    let c = coeus_ops::matmul(&a, &b, &backend);
    assert_eq!(c.shape(), &[2, 2]);
    // a @ I = a
    let expected_f32 = [1.0f32, 2.0, 3.0, 4.0];
    for (got, want) in c.as_slice().iter().zip(expected_f32.iter()) {
        let diff = (got.to_f32() - want).abs();
        assert!(
            diff < 0.01,
            "F16 matmul: got {:.4} want {:.4}",
            got.to_f32(),
            want
        );
    }
}

/// Autograd through F16 tracked variables.
#[test]
fn f16_autograd_smoke() {
    use coeus_autograd::Var;
    let x = Var::new(
        Tensor::<F16, SequentialBackend>::from_slice(
            vec![3],
            &[F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)],
        ),
        true,
    );
    let y = coeus_autograd::sum(&coeus_autograd::mul(&x, &x));
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().expect("F16 sum(x^2) must produce gradient");
    // d/dx(sum(x^2)) = 2x
    let expected_grad = [2.0f32, 4.0, 6.0];
    for (got, want) in gx.as_slice().iter().zip(expected_grad.iter()) {
        let diff = (got.to_f32() - want).abs();
        assert!(
            diff < 0.1,
            "F16 autograd grad: got {:.4} want {:.4}",
            got.to_f32(),
            want
        );
    }
}

/// The mean gradient over `n = 65520` elements is `1/n`, though `n` itself
/// is not an F16: it rounds past the largest finite F16, 65504, to `+inf`, so
/// dividing one by the converted count gives exactly zero. The correctly
/// rounded `1/65520 ≈ 1.52625e-5` is the subnormal `256 · 2^-24`; correct
/// rounding bounds it within half the subnormal spacing, `2^-25`, of `1/n`.
#[test]
fn f16_mean_gradient_at_count_past_f16_range() {
    use coeus_autograd::{mean, mean_axis, Var};

    const N: usize = 65_520;
    let expected = F16::from_f32(256.0 / 16_777_216.0);
    let exact = 1.0_f64 / 65_520.0;
    assert!(
        (f64::from(expected.to_f32()) - exact).abs() <= 2.0_f64.powi(-25),
        "the oracle is the F16 nearest 1/{N}"
    );

    let flat = Var::new(
        Tensor::<F16, SequentialBackend>::from_slice(vec![N], &vec![F16::from_f32(0.0); N]),
        true,
    );
    mean(&flat)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let flat_grad = flat.grad().expect("F16 mean must produce a gradient");
    assert!(
        flat_grad.as_slice().iter().all(|&g| g == expected),
        "mean gradient must be 1/{N} correctly rounded"
    );

    let rows = Var::new(
        Tensor::<F16, SequentialBackend>::from_slice(vec![1, N], &vec![F16::from_f32(0.0); N]),
        true,
    );
    mean_axis(&rows, 1)
        .backward_with_seed(Tensor::from_slice(vec![1, 1], &[F16::from_f32(1.0)]))
        .expect("invariant: valid autograd fixture completes backward");
    let rows_grad = rows.grad().expect("F16 mean_axis must produce a gradient");
    assert!(
        rows_grad.as_slice().iter().all(|&g| g == expected),
        "mean_axis gradient must be 1/{N} correctly rounded"
    );
}

/// The forward mean remains nonzero when the count overflows F16 but its
/// reciprocal is exactly representable: one followed by zeros has mean 2^-16.
#[test]
fn f16_mean_forward_at_count_past_f16_range() {
    use coeus_autograd::mean;

    const N: usize = 65_536;
    let mut values = vec![F16::from_f32(0.0); N];
    values[0] = F16::from_f32(1.0);
    let input = coeus_autograd::Var::new(
        Tensor::<F16, SequentialBackend>::from_slice(vec![N], &values),
        false,
    );
    let output = mean(&input);
    let actual = output.tensor.as_slice()[0].to_f32();
    let expected = 2.0_f32.powi(-16);
    assert_eq!(
        actual, expected,
        "F16 mean must preserve the exact reciprocal"
    );
}
