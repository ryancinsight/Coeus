use super::traits::{reduction_op, ReductionAutogradOp};
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;

pub struct SumOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for SumOp {
    const OP_NAME: &'static str = "sum";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, _param: Option<usize>, backend: &B) -> Tensor<T, B> {
        let total = coeus_ops::sum(a, backend);
        Tensor::from_slice_on([1], &[total], backend)
    }

    #[inline(always)]
    fn scaler(_a: &Tensor<T, B>, _param: Option<usize>, _backend: &B) -> Option<Tensor<T, B>> {
        None
    }
}

pub struct SumAxisOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for SumAxisOp {
    const OP_NAME: &'static str = "sum_axis";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sum_axis(a, param.unwrap(), backend)
    }

    #[inline(always)]
    fn scaler(_a: &Tensor<T, B>, _param: Option<usize>, _backend: &B) -> Option<Tensor<T, B>> {
        None
    }
}

pub struct MeanOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for MeanOp {
    const OP_NAME: &'static str = "mean";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, _param: Option<usize>, backend: &B) -> Tensor<T, B> {
        let total = coeus_ops::sum(a, backend);
        let n = a.numel() as f64;
        Tensor::from_slice_on([1], &[total / T::from_f64(n)], backend)
    }

    #[inline(always)]
    fn scaler(a: &Tensor<T, B>, _param: Option<usize>, backend: &B) -> Option<Tensor<T, B>> {
        let n = a.numel() as f64;
        Some(Tensor::full_on([1], T::from_f64(1.0 / n), backend))
    }
}

pub struct MeanAxisOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> ReductionAutogradOp<T, B> for MeanAxisOp {
    const OP_NAME: &'static str = "mean_axis";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Tensor<T, B> {
        coeus_ops::mean_axis(a, param.unwrap(), backend)
    }

    #[inline(always)]
    fn scaler(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Option<Tensor<T, B>> {
        let axis_len = a.shape()[param.unwrap()] as f64;
        Some(Tensor::full_on([1], T::from_f64(1.0 / axis_len), backend))
    }
}

/// Tracked sum reduction of all elements.
#[must_use]
#[inline]
pub fn sum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    reduction_op::<T, B, SumOp>(a, None)
}

/// Tracked mean reduction of all elements.
#[must_use]
#[inline]
pub fn mean<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    reduction_op::<T, B, MeanOp>(a, None)
}

/// Tracked sum reduction along an axis.
#[must_use]
#[inline]
pub fn sum_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    reduction_op::<T, B, SumAxisOp>(a, Some(axis))
}

/// Tracked mean reduction along an axis.
#[must_use]
#[inline]
pub fn mean_axis<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
) -> Var<T, B> {
    reduction_op::<T, B, MeanAxisOp>(a, Some(axis))
}
