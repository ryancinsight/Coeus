use coeus_autograd::Var;
use coeus_core::{Complex, MoiraiBackend};
use coeus_fft::{fft_1d, fft_1d_var, fft_energy, ifft_1d};
use coeus_tensor::Tensor;

fn assert_close(actual: f64, expected: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= 1e-10,
        "{label}: actual={actual}, expected={expected}, diff={diff}"
    );
}

#[test]
fn fft_1d_prime_length_matches_dft() {
    // Non-power-of-2 (prime N=3) exercises Apollo's mixed-radix/Bluestein path.
    // Closed-form DFT of [1,2,3]: X0=6, X1=-1.5 + (sqrt3/2)i, X2=conj(X1).
    let s3 = 3.0_f64.sqrt() / 2.0;
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([3], &[1.0, 2.0, 3.0]);
    let spectrum = fft_1d(&signal);
    let expected = [
        Complex::new(6.0, 0.0),
        Complex::new(-1.5, s3),
        Complex::new(-1.5, -s3),
    ];
    for (i, (&got, &want)) in spectrum.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(got.re, want.re, &format!("N3 X{i}.re"));
        assert_close(got.im, want.im, &format!("N3 X{i}.im"));
    }
}

#[test]
fn fft_1d_arbitrary_length_roundtrips() {
    // ifft(fft(x)) == x for a range of non-power-of-2 lengths (prime, composite),
    // confirming Apollo handles arbitrary N and that the 1/N inverse normalization
    // holds regardless of factorization.
    for n in [3usize, 5, 6, 7, 9, 15] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
        let signal = Tensor::<f64, MoiraiBackend>::from_slice([n], &data);
        let recon = ifft_1d(&fft_1d(&signal));
        for (i, (&r, &x)) in recon.as_slice().iter().zip(data.iter()).enumerate() {
            assert_close(r, x, &format!("N{n} roundtrip[{i}]"));
        }
    }
}

#[test]
fn fft_1d_matches_hand_dft_and_roundtrips() {
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let spectrum = fft_1d(&signal);
    let expected = [
        Complex::new(10.0, 0.0),
        Complex::new(-2.0, 2.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-2.0, -2.0),
    ];
    for (index, (&actual, &want)) in spectrum.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual.re, want.re, &format!("fft[{index}].re"));
        assert_close(actual.im, want.im, &format!("fft[{index}].im"));
    }

    let reconstructed = ifft_1d(&spectrum);
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert_close(actual, want, &format!("ifft[{index}]"));
    }
}

#[test]
fn fft_1d_var_accumulates_input_gradient_from_complex_seed() {
    let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]), true);
    let y = fft_1d_var(&x);
    let seed = Tensor::from_slice(
        [4],
        &[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.5, -0.5),
            Complex::new(0.0, -1.0),
        ],
    );
    y.backward_with_seed(seed);
    let grad = x.grad().unwrap();
    let expected = [1.5, -1.5, 1.5, 2.5];
    for (index, (&actual, &want)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual, want, &format!("dx[{index}]"));
    }
}

#[test]
fn fft_1d_f32_matches_hand_dft_and_roundtrips() {
    // Exercises the `impl FftScalar for f32` path. Same closed-form DFT as the
    // f64 case; tolerance scaled to f32: eps_f32 ~= 1.19e-7, radix error ~O(logN*eps),
    // 1e-4 is a safe margin that still rejects real defects.
    let tol = 1e-4_f32;
    let signal = Tensor::<f32, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let spectrum = fft_1d(&signal);
    let expected = [
        Complex::new(10.0_f32, 0.0),
        Complex::new(-2.0, 2.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-2.0, -2.0),
    ];
    for (index, (&actual, &want)) in spectrum.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.re - want.re).abs() < tol && (actual.im - want.im).abs() < tol,
            "f32 fft[{index}]: got {actual:?}, want {want:?}"
        );
    }
    let reconstructed = ifft_1d(&spectrum);
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert!(
            (actual - want).abs() < tol,
            "f32 ifft[{index}]: got {actual}, want {want}"
        );
    }
}

#[test]
fn fft_1d_f32_agrees_with_f64_reference() {
    // Differential check: the f32 backend must agree with the f64 reference
    // within f32 precision (per numerical_discipline reduction-order bounds).
    let data = [0.25, -1.0, 0.5, 2.0, -0.75, 1.25, -0.5, 0.125];
    let data32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let ref64 = fft_1d(&Tensor::<f64, MoiraiBackend>::from_slice([8], &data));
    let got32 = fft_1d(&Tensor::<f32, MoiraiBackend>::from_slice([8], &data32));
    let tol = 1e-4_f32;
    for (index, (&r, &g)) in ref64.as_slice().iter().zip(got32.as_slice()).enumerate() {
        assert!(
            (r.re as f32 - g.re).abs() < tol && (r.im as f32 - g.im).abs() < tol,
            "f32 vs f64 bin {index}: f32={g:?}, f64={r:?}"
        );
    }
}

#[test]
fn fft_energy_gradient_matches_parseval_oracle() {
    let data = [0.25, -1.0, 0.5, 2.0, -0.75, 1.25, -0.5, 0.125];
    let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([8], &data), true);
    let loss = fft_energy(&x);
    let expected_energy = data.iter().map(|v| v * v).sum::<f64>() * data.len() as f64;
    assert_close(loss.tensor.as_slice()[0], expected_energy, "fft_energy");

    loss.backward();
    let grad = x.grad().unwrap();
    for (index, (&actual, &value)) in grad.as_slice().iter().zip(data.iter()).enumerate() {
        let expected = 2.0 * data.len() as f64 * value;
        assert_close(actual, expected, &format!("d_energy_dx[{index}]"));
    }
}
