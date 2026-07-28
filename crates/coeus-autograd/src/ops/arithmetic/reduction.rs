use super::traits::{ReductionAutogradOp, reduction_op};
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;

/// ZST tag for sum reduction autograd.
pub struct SumOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for SumOp {
    const OP_NAME: &'static str = "sum";

    #[inline(always)]
    fn forward(
        a: &Tensor<T, B>,
        _param: Option<usize>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let total = coeus_ops::sum(a, backend)?;
        Tensor::from_slice_on([1], &[total], backend)
    }

    #[inline(always)]
    fn scaler(
        _a: &Tensor<T, B>,
        _param: Option<usize>,
        _backend: &B,
    ) -> Result<Option<Tensor<T, B>>, B::Error> {
        Ok(None)
    }
}

/// ZST tag for sum-along-axis reduction autograd.
pub struct SumAxisOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for SumAxisOp {
    const OP_NAME: &'static str = "sum_axis";

    #[inline(always)]
    fn forward(
        a: &Tensor<T, B>,
        param: Option<usize>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::sum_axis(
            a,
            param.ok_or_else(|| {
                B::Error::from(coeus_core::BackendError::Storage {
                    operation: "sum_axis",
                    reason: "missing reduction axis".to_owned(),
                })
            })?,
            backend,
        )
    }

    #[inline(always)]
    fn scaler(
        _a: &Tensor<T, B>,
        _param: Option<usize>,
        _backend: &B,
    ) -> Result<Option<Tensor<T, B>>, B::Error> {
        Ok(None)
    }
}

/// ZST tag for mean reduction autograd.
pub struct MeanOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for MeanOp {
    const OP_NAME: &'static str = "mean";

    #[inline(always)]
    fn forward(
        a: &Tensor<T, B>,
        _param: Option<usize>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let total = coeus_ops::sum(a, backend)?;
        let n = a.numel() as f64;
        Tensor::from_slice_on([1], &[total / T::from_f64(n)], backend)
    }

    #[inline(always)]
    fn scaler(
        a: &Tensor<T, B>,
        _param: Option<usize>,
        backend: &B,
    ) -> Result<Option<Tensor<T, B>>, B::Error> {
        let n = a.numel() as f64;
        Ok(Some(Tensor::full_on([1], T::from_f64(1.0 / n), backend)?))
    }
}

/// ZST tag for mean-along-axis reduction autograd.
pub struct MeanAxisOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for MeanAxisOp {
    const OP_NAME: &'static str = "mean_axis";

    #[inline(always)]
    fn forward(
        a: &Tensor<T, B>,
        param: Option<usize>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::mean_axis(
            a,
            param.ok_or_else(|| {
                B::Error::from(coeus_core::BackendError::Storage {
                    operation: "mean_axis",
                    reason: "missing reduction axis".to_owned(),
                })
            })?,
            backend,
        )
    }

    #[inline(always)]
    fn scaler(
        a: &Tensor<T, B>,
        param: Option<usize>,
        backend: &B,
    ) -> Result<Option<Tensor<T, B>>, B::Error> {
        let axis = param.ok_or_else(|| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "mean_axis",
                reason: "missing reduction axis".to_owned(),
            })
        })?;
        let axis_len = a.shape()[axis] as f64;
        Ok(Some(Tensor::full_on(
            [1],
            T::from_f64(1.0 / axis_len),
            backend,
        )?))
    }
}

/// Tracked sum reduction of all elements.
#[must_use]
#[inline]
pub fn sum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    reduction_op::<T, B, SumOp>(a, None)
}

/// Tracked mean reduction of all elements.
#[must_use]
#[inline]
pub fn mean<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    reduction_op::<T, B, MeanOp>(a, None)
}

/// Tracked sum reduction along an axis.
#[must_use]
#[inline]
pub fn sum_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Result<Var<T, B>, B::Error> {
    reduction_op::<T, B, SumAxisOp>(a, Some(axis))
}

/// Tracked mean reduction along an axis.
#[must_use]
#[inline]
pub fn mean_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Result<Var<T, B>, B::Error> {
    reduction_op::<T, B, MeanAxisOp>(a, Some(axis))
}

/// Tracked sum that treats NaN as zero (`torch.nansum`).
///
/// Replaces every NaN in `a` with 0 via tracked `masked_fill`, then reduces
/// with `sum`. Differentiable: gradient at NaN positions is zero because the
/// mask removes those entries from the cleaned tensor.
#[must_use]
#[inline]
pub fn nansum<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let mask_data: Vec<T> = a
        .tensor
        .as_slice()
        .iter()
        .map(|&v| {
            if <T as Float>::is_nan(v) {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect();
    let mask = crate::Var::new(
        coeus_tensor::Tensor::from_slice_on(a.tensor.shape_cloned(), &mask_data, &backend)?,
        false,
    )?;
    let cleaned = crate::ops::shape::masked_fill(a, &mask, T::zero())?;
    sum(&cleaned)
}

/// Tracked mean that treats NaN as missing (`torch.nanmean`).
///
/// Replaces NaN with 0, counts non-NaN elements, then divides sum by count.
/// Differentiable: gradient at NaN positions is zero.
#[must_use]
#[inline]
pub fn nanmean<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let slice = a.tensor.as_slice();
    let count = slice.iter().filter(|&&v| !<T as Float>::is_nan(v)).count();
    let mask_data: Vec<T> = slice
        .iter()
        .map(|&v| {
            if <T as Float>::is_nan(v) {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect();
    let mask = crate::Var::new(
        coeus_tensor::Tensor::from_slice_on(a.tensor.shape_cloned(), &mask_data, &backend)?,
        false,
    )?;
    let cleaned = crate::ops::shape::masked_fill(a, &mask, T::zero())?;
    let s = sum(&cleaned)?;
    crate::scalar_div(&s, T::from_f64(count as f64))
}

#[cfg(test)]
mod nan_reduction_tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    type B = SequentialBackend;

    fn nan_var(data: &[f64]) -> Var<f64, B> {
        Var::new(Tensor::<f64, B>::from_slice(vec![data.len()], data).expect("valid tensor construction"), true).expect("valid variable construction")
    }

    #[test]
    fn nansum_treats_nan_as_zero() {
        let x = nan_var(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let s = nansum(&x).expect("valid autograd operation");
        let v = s.tensor.as_slice()[0];
        assert!((v - 9.0).abs() < 1e-10, "nansum: expected 9, got {v}");
        s.backward().expect("valid backward propagation");
        assert_eq!(
            x.grad().unwrap().as_slice(),
            &[1.0, 0.0, 1.0, 0.0, 1.0],
            "nansum gradient zeros NaN positions"
        );
    }

    #[test]
    fn nansum_all_finite_matches_sum() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let x = nan_var(&data);
        let ns = nansum(&x).expect("valid autograd operation");
        let s = sum(&x).expect("valid autograd operation");
        assert!((ns.tensor.as_slice()[0] - s.tensor.as_slice()[0]).abs() < 1e-10);
    }

    #[test]
    fn nanmean_excludes_nan() {
        let x = nan_var(&[2.0, f64::NAN, 4.0, f64::NAN, 6.0]);
        let m = nanmean(&x).expect("valid autograd operation");
        let v = m.tensor.as_slice()[0];
        assert!((v - 4.0).abs() < 1e-10, "nanmean: expected 4.0, got {v}");
        m.backward().expect("valid backward propagation");
        let grad = x.grad().unwrap();
        let expected = [1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 3.0];
        for (i, (&actual, &expected)) in grad.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "nanmean grad[{i}]: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn nansum_value_matches_finite_sum() {
        // The value of nansum must equal the sum of finite elements.
        let x = nan_var(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let s = nansum(&x).expect("valid autograd operation");
        assert!(
            (s.tensor.as_slice()[0] - 9.0).abs() < 1e-10,
            "nansum value should be 9.0"
        );
    }
}
