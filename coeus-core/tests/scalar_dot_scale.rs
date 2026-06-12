use coeus_core::Scalar;

fn sequential_dot(data_a: &[f32], data_b: &[f32]) -> f32 {
    data_a.iter().zip(data_b).map(|(&x, &y)| x * y).sum::<f32>()
}

fn dot_tolerance(data_a: &[f32], data_b: &[f32]) -> f32 {
    let n = data_a.len() as f32;
    let eps = f32::EPSILON;
    let gamma = (n * eps) / (1.0 - n * eps);
    let magnitude = data_a
        .iter()
        .zip(data_b)
        .map(|(&x, &y)| (x * y).abs())
        .sum::<f32>();
    gamma * magnitude
}

#[test]
fn native_float_dot_slice_matches_scalar_reference_with_derived_bound() {
    for &n in &[0usize, 1, 7, 8, 31, 257, 1024] {
        let data_a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.25).sin()).collect();
        let data_b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).cos()).collect();

        let got = f32::dot_slice(&data_a, &data_b);
        let expected = sequential_dot(&data_a, &data_b);
        let tol = dot_tolerance(&data_a, &data_b);

        assert!(
            (got - expected).abs() <= tol,
            "dot n={n}: got {got}, expected {expected}, tol {tol}",
        );
    }
}

#[test]
fn native_float_scale_slice_matches_scalar_reference_exactly() {
    for &n in &[0usize, 1, 7, 8, 31, 257, 1024] {
        let mut got: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 3.0).collect();
        let mut expected = got.clone();
        let scale = -0.5_f32;

        f32::scale_slice(&mut got, scale);
        for value in &mut expected {
            *value *= scale;
        }

        for (i, (&actual, &reference)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(actual.to_bits(), reference.to_bits(), "scale n={n} i={i}",);
        }
    }
}
