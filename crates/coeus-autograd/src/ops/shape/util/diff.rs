use crate::var::Var;
use coeus_core::Scalar;

/// N-th order discrete difference along `dim` (`torch.diff`).
///
/// Each pass computes `out[i] = x[i + 1] - x[i]` along `dim`, shrinking that
/// dimension's extent by one; applied `n` times. `n == 0` returns `x` unchanged.
/// The inverse of [`cumsum`](super::cumsum). Differentiable via composition of the
/// tracked `slice` and `sub`.
///
/// # Panics
/// If `dim` is out of range, or the extent along `dim` is exhausted before `n`
/// differences complete.
#[must_use]
pub fn diff<T, B>(x: &Var<T, B>, n: usize, dim: usize) -> Var<T, B>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
{
    let ndim = x.tensor.ndim();
    assert!(dim < ndim, "diff: dim {dim} out of range for rank {ndim}");
    let mut result = x.clone();
    for _ in 0..n {
        let dims: Vec<usize> = result.tensor.shape().to_vec();
        let len = dims[dim];
        assert!(
            len >= 1,
            "diff: dimension {dim} exhausted before {n} differences"
        );
        let ranges = |shift: bool| -> Vec<(usize, usize)> {
            dims.iter()
                .enumerate()
                .map(|(d, &e)| {
                    if d != dim {
                        (0, e)
                    } else if shift {
                        (1, e)
                    } else {
                        (0, e - 1)
                    }
                })
                .collect()
        };
        let front = crate::ops::slice(&result, &ranges(true));
        let back = crate::ops::slice(&result, &ranges(false));
        result = crate::ops::sub(&front, &back);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn diff_first_and_second_order_and_gradient() {
        let x =
            Var::<f64, MoiraiBackend>::new(Tensor::from_slice([4], &[1.0, 3.0, 6.0, 10.0]), true);

        // n=1: [3-1, 6-3, 10-6] = [2, 3, 4].
        let d1 = diff(&x, 1, 0);
        assert_eq!(d1.tensor.shape(), &[3]);
        assert_eq!(d1.tensor.as_slice(), &[2.0, 3.0, 4.0]);

        // n=2: diff([2,3,4]) = [1, 1].
        let d2 = diff(&x, 2, 0);
        assert_eq!(d2.tensor.as_slice(), &[1.0, 1.0]);

        // n=0 is the identity.
        assert_eq!(diff(&x, 0, 0).tensor.as_slice(), x.tensor.as_slice());

        // Gradient of sum(diff): dx = [-1, 0, 0, 1] (telescoping cancellation).
        d1.backward();
        assert_eq!(x.grad().unwrap().as_slice(), &[-1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn diff_along_inner_dim() {
        // [2,3] rows [1,2,4] and [0,10,30]; diff dim=1 -> [[1,2],[10,20]].
        let x = Var::<f64, MoiraiBackend>::new(
            Tensor::from_slice([2, 3], &[1.0, 2.0, 4.0, 0.0, 10.0, 30.0]),
            false,
        );
        let d = diff(&x, 1, 1);
        assert_eq!(d.tensor.shape(), &[2, 2]);
        assert_eq!(d.tensor.to_contiguous().as_slice(), &[1.0, 2.0, 10.0, 20.0]);
    }
}
