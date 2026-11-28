#![expect(
    clippy::unwrap_used,
    reason = "test/bench/example code asserts and reports by design"
)]

use coeus_autograd::Var;
use coeus_core::{Complex, MoiraiBackend};
use coeus_fft::{
    fft_1d, fft_1d_var, fft_2d, fft_2d_var, fft_3d, fft_3d_var, fft_energy, ifft_1d, ifft_1d_var,
    ifft_2d, ifft_2d_var, ifft_3d, ifft_3d_var, irfft_1d, irfft_1d_var, irfft_2d, irfft_2d_var,
    irfft_3d, irfft_3d_var, rfft_1d, rfft_1d_var, rfft_2d, rfft_2d_var, rfft_3d, rfft_3d_var,
};
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
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([3], &[1.0, 2.0, 3.0])
        .expect("invariant: test backend operation succeeds");
    let spectrum = fft_1d(&signal).expect("invariant: test FFT backend operation succeeds");
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
        let signal = Tensor::<f64, MoiraiBackend>::from_slice([n], &data)
            .expect("invariant: test backend operation succeeds");
        let recon =
            ifft_1d(&fft_1d(&signal).expect("invariant: test FFT backend operation succeeds"))
                .expect("invariant: test FFT backend operation succeeds");
        for (i, (&r, &x)) in recon.as_slice().iter().zip(data.iter()).enumerate() {
            assert_close(r, x, &format!("N{n} roundtrip[{i}]"));
        }
    }
}

#[test]
fn fft_1d_matches_hand_dft_and_roundtrips() {
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let spectrum = fft_1d(&signal).expect("invariant: test FFT backend operation succeeds");
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

    let reconstructed = ifft_1d(&spectrum).expect("invariant: test FFT backend operation succeeds");
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
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = fft_1d_var(&x).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice(
        [4],
        &[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.5, -0.5),
            Complex::new(0.0, -1.0),
        ],
    )
    .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    let expected = [1.5, -1.5, 1.5, 2.5];
    for (index, (&actual, &want)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual, want, &format!("dx[{index}]"));
    }
}

#[test]
fn fft_2d_matches_separable_hand_dft_and_roundtrips() {
    // X[k1,k2] = sum over the 2x2 input of x[n1,n2] W2^(k1*n1+k2*n2) with
    // W2 = -1: X[0,0]=10, X[0,1]=-2, X[1,0]=-4, X[1,1]=0.
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let spectrum = fft_2d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(spectrum.shape(), &[2, 2]);
    let expected = [
        Complex::new(10.0, 0.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-4.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    for (index, (&actual, &want)) in spectrum.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual.re, want.re, &format!("fft2d[{index}].re"));
        assert_close(actual.im, want.im, &format!("fft2d[{index}].im"));
    }

    let reconstructed = ifft_2d(&spectrum).expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert_close(actual, want, &format!("ifft2d[{index}]"));
    }
}

#[test]
fn fft_2d_rectangular_roundtrips() {
    // Non-square [2, 3] exercises separate row/column plan factorization.
    let data: Vec<f64> = (0..6).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &data)
        .expect("invariant: test backend operation succeeds");
    let reconstructed =
        ifft_2d(&fft_2d(&signal).expect("invariant: test FFT backend operation succeeds"))
            .expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed.as_slice().iter().zip(data.iter()).enumerate() {
        assert_close(actual, want, &format!("ifft2d[{index}]"));
    }
}

#[test]
fn fft_3d_dc_bin_is_sum_and_roundtrips() {
    // X[0,0,0] = sum of all 8 inputs = 36; every other bin is checked
    // implicitly by the roundtrip below.
    let signal = Tensor::<f64, MoiraiBackend>::from_slice(
        [2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .expect("invariant: test backend operation succeeds");
    let spectrum = fft_3d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(spectrum.shape(), &[2, 2, 2]);
    assert_close(spectrum.as_slice()[0].re, 36.0, "fft3d[0,0,0].re");
    assert_close(spectrum.as_slice()[0].im, 0.0, "fft3d[0,0,0].im");

    let reconstructed = ifft_3d(&spectrum).expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert_close(actual, want, &format!("ifft3d[{index}]"));
    }
}

#[test]
fn fft_3d_rectangular_roundtrips() {
    // Non-cubic [2, 3, 2] exercises independent per-axis plan factorization.
    let data: Vec<f64> = (0..12).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 3, 2], &data)
        .expect("invariant: test backend operation succeeds");
    let reconstructed =
        ifft_3d(&fft_3d(&signal).expect("invariant: test FFT backend operation succeeds"))
            .expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed.as_slice().iter().zip(data.iter()).enumerate() {
        assert_close(actual, want, &format!("ifft3d[{index}]"));
    }
}

#[test]
fn fft_3d_var_accumulates_input_gradient() {
    // Adjoint check: with a unit seed on X[0,0,0] only, dx = N*ifft(seed)
    // is constant 1s (ifft of a delta is uniform 1/N, scaled by N).
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = fft_3d_var(&x).expect("invariant: test FFT backend operation succeeds");
    let mut seed_data = vec![Complex::new(0.0, 0.0); 8];
    seed_data[0] = Complex::new(1.0, 0.0);
    let seed = Tensor::from_slice([2, 2, 2], &seed_data)
        .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    for (index, &actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual, 1.0, &format!("dx3d[{index}]"));
    }
}

#[test]
fn fft_2d_var_accumulates_input_gradient() {
    // Adjoint check: with a unit seed on X[0,0] only, dx = Nx*Ny*ifft(seed)
    // is constant 1s (ifft of a delta is uniform 1/N, scaled by N).
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = fft_2d_var(&x).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice(
        [2, 2],
        &[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    )
    .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    for (index, &actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual, 1.0, &format!("dx2d[{index}]"));
    }
}

#[test]
fn fft_1d_f32_matches_hand_dft_and_roundtrips() {
    // Exercises the `impl FftScalar for f32` path. Same closed-form DFT as the
    // f64 case; tolerance scaled to f32: eps_f32 ~= 1.19e-7, radix error ~O(logN*eps),
    // 1e-4 is a safe margin that still rejects real defects.
    let tol = 1e-4_f32;
    let signal = Tensor::<f32, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let spectrum = fft_1d(&signal).expect("invariant: test FFT backend operation succeeds");
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
    let reconstructed = ifft_1d(&spectrum).expect("invariant: test FFT backend operation succeeds");
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
    let ref64 = fft_1d(
        &Tensor::<f64, MoiraiBackend>::from_slice([8], &data)
            .expect("invariant: test backend operation succeeds"),
    )
    .expect("invariant: test FFT backend operation succeeds");
    let got32 = fft_1d(
        &Tensor::<f32, MoiraiBackend>::from_slice([8], &data32)
            .expect("invariant: test backend operation succeeds"),
    )
    .expect("invariant: test FFT backend operation succeeds");
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
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([8], &data).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let loss = fft_energy(&x).expect("invariant: test FFT backend operation succeeds");
    let expected_energy = data.iter().map(|v| v * v).sum::<f64>() * data.len() as f64;
    assert_close(loss.tensor.as_slice()[0], expected_energy, "fft_energy");

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    for (index, (&actual, &value)) in grad.as_slice().iter().zip(data.iter()).enumerate() {
        let expected = 2.0 * data.len() as f64 * value;
        assert_close(actual, expected, &format!("d_energy_dx[{index}]"));
    }
}

#[test]
fn rfft_1d_matches_full_spectrum_prefix_and_roundtrips() {
    // Half spectrum of [1,2,3,4] is the first three bins of the full DFT:
    // X = [10, -2+2i, -2, -2-2i]; half keeps [10, -2+2i, -2].
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let half = rfft_1d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[3]);
    let expected = [
        Complex::new(10.0, 0.0),
        Complex::new(-2.0, 2.0),
        Complex::new(-2.0, 0.0),
    ];
    for (index, (&actual, &want)) in half.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual.re, want.re, &format!("rfft[{index}].re"));
        assert_close(actual.im, want.im, &format!("rfft[{index}].im"));
    }

    let reconstructed = irfft_1d(&half, 4).expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert_close(actual, want, &format!("irfft[{index}]"));
    }
}

#[test]
fn rfft_1d_odd_length_roundtrips() {
    // Odd n = 5: half holds 3 bins; the Nyquist bin has no mirror image.
    let data: Vec<f64> = (0..5).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([5], &data)
        .expect("invariant: test backend operation succeeds");
    let half = rfft_1d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[3]);
    let reconstructed = irfft_1d(&half, 5).expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed.as_slice().iter().zip(data.iter()).enumerate() {
        assert_close(actual, want, &format!("irfft5[{index}]"));
    }
}

#[test]
fn rfft_1d_var_accumulates_mirrored_adjoint_gradient() {
    // Seed only the DC half-bin: dx = N*ifft(mirror([1,0,0])) is constant 1s.
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = rfft_1d_var(&x).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice(
        [3],
        &[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    )
    .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    for (index, &actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual, 1.0, &format!("drfft_dx[{index}]"));
    }
}

#[test]
fn irfft_1d_var_folds_mirror_lanes_into_half_gradient() {
    // Seed a delta at output position 0 with n = 4: the full adjoint is
    // uniform 1/4 per bin, and the fold doubles the interior bin (k=1 gets
    // its own lane plus its mirror's conjugate).
    let half = Var::<Complex<f64>, MoiraiBackend>::new(
        Tensor::from_slice(
            [3],
            &[
                Complex::new(0.3, 0.0),
                Complex::new(0.1, 0.2),
                Complex::new(-0.4, 0.0),
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = irfft_1d_var(&half, 4).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice([4], &[1.0, 0.0, 0.0, 0.0])
        .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = half.grad().unwrap();
    // fft([1,0,0,0]) = [1,1,1,1]; fold gives [1/4, (1+1)/4, 1/4] = [0.25, 0.5, 0.25].
    let expected = [0.25, 0.5, 0.25];
    for (index, (&actual, &want)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual.re, want, &format!("dirfft[{index}].re"));
        assert_close(actual.im, 0.0, &format!("dirfft[{index}].im"));
    }
}

#[test]
fn rfft_2d_matches_full_prefix_and_roundtrips() {
    // Even-width [2,2] field: half width is 2/2 + 1 = 2, so no truncation;
    // this pins the row-major bin layout and the DC bin exactly.
    // Full 2-D DFT of [[1,2],[3,4]]: X = [[10,-2],[-4,0]].
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let half = rfft_2d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[2, 2]);
    let expected = [
        Complex::new(10.0, 0.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-4.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    // NOTE: this expectation pins the row-major full-spectrum layout of the
    // half bins (the last axis is the truncated one). If Apollo ever changes
    // which axis it halves, this test names the new layout rather than passing
    // silently.
    for (index, (&actual, &want)) in half.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_close(actual.re, want.re, &format!("rfft2d[{index}].re"));
        assert_close(actual.im, want.im, &format!("rfft2d[{index}].im"));
    }

    let reconstructed = irfft_2d(&half, 2, 2)
        .expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert_close(actual, want, &format!("irfft2d[{index}]"));
    }
}

#[test]
fn rfft_2d_truncates_last_axis_like_full_prefix() {
    // Odd-width [2,3] field: half width is 3/2 + 1 = 2. The half spectrum
    // must equal the full spectrum's [.., 0..2] prefix bin for bin; this is
    // the defining contract of the truncation.
    let data: Vec<f64> = (0..6).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &data)
        .expect("invariant: test backend operation succeeds");
    let half = rfft_2d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[2, 2]);
    let full = fft_2d(&signal).expect("invariant: test FFT backend operation succeeds");
    let full_slice = full.as_slice();
    for row in 0..2 {
        for (lane, &got) in half.as_slice()[row * 2..row * 2 + 2].iter().enumerate() {
            let want = full_slice[row * 3 + lane];
            assert_close(got.re, want.re, &format!("prefix[{row},{lane}].re"));
            assert_close(got.im, want.im, &format!("prefix[{row},{lane}].im"));
        }
    }
}

#[test]
fn rfft_2d_rectangular_roundtrips() {
    // Non-square [2, 3]: half width is 3/2 + 1 = 2.
    let data: Vec<f64> = (0..6).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &data)
        .expect("invariant: test backend operation succeeds");
    let half = rfft_2d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[2, 2]);
    let reconstructed = irfft_2d(&half, 2, 3)
        .expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(data.iter())
        .enumerate()
    {
        assert_close(actual, want, &format!("irfft2d[{index}]"));
    }
}

#[test]
fn rfft_3d_dc_bin_is_sum_and_roundtrips() {
    // X[0,0,0] = sum of all 8 inputs = 36.
    let signal = Tensor::<f64, MoiraiBackend>::from_slice(
        [2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .expect("invariant: test backend operation succeeds");
    let half = rfft_3d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[2, 2, 2]);
    assert_close(half.as_slice()[0].re, 36.0, "rfft3d[0,0,0].re");
    assert_close(half.as_slice()[0].im, 0.0, "rfft3d[0,0,0].im");

    let reconstructed =
        irfft_3d(&half, 2, 2, 2).expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(signal.as_slice())
        .enumerate()
    {
        assert_close(actual, want, &format!("irfft3d[{index}]"));
    }
}

#[test]
fn rfft_3d_rectangular_roundtrips() {
    // Non-cubic [2, 3, 2]: half depth is 2/2 + 1 = 2.
    let data: Vec<f64> = (0..12).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 3, 2], &data)
        .expect("invariant: test backend operation succeeds");
    let half = rfft_3d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[2, 3, 2]);
    let reconstructed = irfft_3d(&half, 2, 3, 2)
        .expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(data.iter())
        .enumerate()
    {
        assert_close(actual, want, &format!("irfft3d[{index}]"));
    }
}

#[test]
fn rfft_2d_odd_width_roundtrips() {
    // Odd ny = 3 exercises the asymmetric mirror (no Nyquist column).
    let data: Vec<f64> = (0..6).map(|i| (i as f64 * 0.37).sin() - 0.2).collect();
    let signal = Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &data)
        .expect("invariant: test backend operation succeeds");
    let half = rfft_2d(&signal).expect("invariant: test FFT backend operation succeeds");
    assert_eq!(half.shape(), &[2, 2]);
    let reconstructed = irfft_2d(&half, 2, 3)
        .expect("invariant: test FFT backend operation succeeds");
    for (index, (&actual, &want)) in reconstructed
        .as_slice()
        .iter()
        .zip(data.iter())
        .enumerate()
    {
        assert_close(actual, want, &format!("irfft2d-odd[{index}]"));
    }
}

#[test]
fn rfft_2d_var_accumulates_mirrored_adjoint_gradient() {
    // Seed only the DC half-bin: dx = Nx*Ny*ifft(mirror(seed)) is constant 1s.
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = rfft_2d_var(&x).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice(
        [2, 2],
        &[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    )
    .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    for (index, &actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual, 1.0, &format!("drfft2d[{index}]"));
    }
}

#[test]
fn irfft_2d_var_folds_mirror_lanes() {
    // Seed a delta at output (0,0) with nx=ny=2: the full adjoint is uniform
    // 1/4, and the fold doubles the (0,1)/(1,0)/(1,1)... precisely: corners
    // (0,0),(0,1),(1,0),(1,1) are all self-mirrors for even dims, so every
    // lane keeps 1/4.
    let half = Var::<Complex<f64>, MoiraiBackend>::new(
        Tensor::from_slice(
            [2, 2],
            &[
                Complex::new(0.3, 0.0),
                Complex::new(0.1, 0.2),
                Complex::new(-0.4, 0.1),
                Complex::new(0.2, -0.3),
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = irfft_2d_var(&half, 2, 2).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice([2, 2], &[1.0, 0.0, 0.0, 0.0])
        .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    // fft([1,0,0,0]) = [1,1,1,1]; every 2x2 lane is a self-mirror corner,
    // so the fold keeps 1/4 everywhere.
    let grad = half.grad().unwrap();
    for (index, actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual.re, 0.25, &format!("dirfft2d[{index}].re"));
        assert_close(actual.im, 0.0, &format!("dirfft2d[{index}].im"));
    }
}

#[test]
fn irfft_2d_var_doubles_interior_lanes() {
    // Odd width ny = 3: lane (0,1)/(1,1) are interior (mirror lives outside
    // the half set), so their gradient doubles; (0,0)/(1,0) are self-mirrors.
    let half = Var::<Complex<f64>, MoiraiBackend>::new(
        Tensor::from_slice(
            [2, 2],
            &[
                Complex::new(0.3, 0.0),
                Complex::new(0.1, 0.2),
                Complex::new(-0.4, 0.1),
                Complex::new(0.2, -0.3),
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = irfft_2d_var(&half, 2, 3).expect("invariant: test FFT backend operation succeeds");
    let seed = Tensor::from_slice([2, 3], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    // fft([1,0,...]) is uniform 1/6; fold: k2=0 lanes single (1/6),
    // k2=1 lanes doubled (2/6).
    let grad = half.grad().unwrap();
    let expected_re = [1.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0];
    for (index, (&actual, &want)) in grad.as_slice().iter().zip(expected_re.iter()).enumerate() {
        assert_close(actual.re, want, &format!("dirfft2d-odd[{index}].re"));
        assert_close(actual.im, 0.0, &format!("dirfft2d-odd[{index}].im"));
    }
}

#[test]
fn rfft_3d_var_accumulates_mirrored_adjoint_gradient() {
    // Seed only the DC half-bin: dx is constant 1s.
    let x = Var::<f64, MoiraiBackend>::new(
        Tensor::from_slice([2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = rfft_3d_var(&x).expect("invariant: test FFT backend operation succeeds");
    let mut seed_data = vec![Complex::new(0.0, 0.0); 8];
    seed_data[0] = Complex::new(1.0, 0.0);
    let seed = Tensor::from_slice([2, 2, 2], &seed_data)
        .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = x.grad().unwrap();
    for (index, &actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual, 1.0, &format!("drfft3d[{index}]"));
    }
}

#[test]
fn irfft_3d_var_folds_mirror_lanes() {
    // All-even [2,2,2]: all 8 half lanes are self-mirror corners, so a delta
    // seed yields a uniform 1/8 gradient.
    let half_data = vec![
        Complex::new(0.3, 0.0),
        Complex::new(0.1, 0.2),
        Complex::new(-0.4, 0.1),
        Complex::new(0.2, -0.3),
        Complex::new(0.5, 0.0),
        Complex::new(-0.1, 0.4),
        Complex::new(0.7, -0.2),
        Complex::new(-0.3, -0.1),
    ];
    let half = Var::<Complex<f64>, MoiraiBackend>::new(
        Tensor::from_slice([2, 2, 2], &half_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y =
        irfft_3d_var(&half, 2, 2, 2).expect("invariant: test FFT backend operation succeeds");
    let mut seed_data = vec![0.0; 8];
    seed_data[0] = 1.0;
    let seed = Tensor::from_slice([2, 2, 2], &seed_data)
        .expect("invariant: test backend operation succeeds");
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let grad = half.grad().unwrap();
    for (index, actual) in grad.as_slice().iter().enumerate() {
        assert_close(actual.re, 0.125, &format!("dirfft3d[{index}].re"));
        assert_close(actual.im, 0.0, &format!("dirfft3d[{index}].im"));
    }
}
