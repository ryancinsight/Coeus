use tensor::{Tensor, Result, CpuBackend, Float32, DenseStorage};
use tensor::ops;

type CpuTensorF32 = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

#[test]
fn test_comparison_ops() -> Result<()> {
    let backend = CpuBackend::default();
    let data = vec![
        Float32::new(1.0),
        Float32::new(f32::NAN),
        Float32::new(f32::INFINITY),
        Float32::new(-f32::INFINITY),
        Float32::new(0.0),
    ];
    let x = CpuTensorF32::from_vec_with_backend(data, &[5], backend.clone())?;

    let isnan = ops::isnan(&x)?;
    let isnan_data = isnan.as_slice();
    assert_eq!(isnan_data[0].get(), 0.0);
    assert_eq!(isnan_data[1].get(), 1.0);
    assert_eq!(isnan_data[2].get(), 0.0);
    assert_eq!(isnan_data[3].get(), 0.0);
    assert_eq!(isnan_data[4].get(), 0.0);

    let isinf = ops::isinf(&x)?;
    let isinf_data = isinf.as_slice();
    assert_eq!(isinf_data[0].get(), 0.0);
    assert_eq!(isinf_data[1].get(), 0.0);
    assert_eq!(isinf_data[2].get(), 1.0);
    assert_eq!(isinf_data[3].get(), 1.0);
    assert_eq!(isinf_data[4].get(), 0.0);

    let isfinite = ops::isfinite(&x)?;
    let isfinite_data = isfinite.as_slice();
    assert_eq!(isfinite_data[0].get(), 1.0);
    assert_eq!(isfinite_data[1].get(), 0.0);
    assert_eq!(isfinite_data[2].get(), 0.0);
    assert_eq!(isfinite_data[3].get(), 0.0);
    assert_eq!(isfinite_data[4].get(), 1.0);

    Ok(())
}

#[test]
fn test_logical_ops() -> Result<()> {
    let backend = CpuBackend::default();
    let a = CpuTensorF32::from_vec_with_backend(
        vec![Float32::new(1.0), Float32::new(1.0), Float32::new(0.0), Float32::new(0.0)],
        &[4],
        backend.clone(),
    )?;
    let b = CpuTensorF32::from_vec_with_backend(
        vec![Float32::new(1.0), Float32::new(0.0), Float32::new(1.0), Float32::new(0.0)],
        &[4],
        backend.clone(),
    )?;

    let and = ops::logical_and(&a, &b)?;
    let and_data = and.as_slice();
    assert_eq!(and_data[0].get(), 1.0);
    assert_eq!(and_data[1].get(), 0.0);
    assert_eq!(and_data[2].get(), 0.0);
    assert_eq!(and_data[3].get(), 0.0);

    let or = ops::logical_or(&a, &b)?;
    let or_data = or.as_slice();
    assert_eq!(or_data[0].get(), 1.0);
    assert_eq!(or_data[1].get(), 1.0);
    assert_eq!(or_data[2].get(), 1.0);
    assert_eq!(or_data[3].get(), 0.0);

    Ok(())
}

#[test]
fn test_math_parity_ops() -> Result<()> {
    let backend = CpuBackend::default();
    let x = CpuTensorF32::from_vec_with_backend(
        vec![Float32::new(0.0), Float32::new(1.0), Float32::new(2.0)],
        &[3],
        backend.clone(),
    )?;

    // log1p: ln(1+x)
    let log1p = ops::log1p(&x)?;
    let log1p_data = log1p.as_slice();
    assert!((log1p_data[0].get() - 0.0f32.ln_1p()).abs() < 1e-6);
    assert!((log1p_data[1].get() - 1.0f32.ln_1p()).abs() < 1e-6);

    // expm1: exp(x)-1
    let expm1 = ops::expm1(&x)?;
    let expm1_data = expm1.as_slice();
    assert!((expm1_data[1].get() - 1.0f32.exp_m1()).abs() < 1e-6);

    Ok(())
}
