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
