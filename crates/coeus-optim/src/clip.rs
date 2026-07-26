// ── Gradient clipping ──
//
// Global L2 gradient norm clipping across all parameter variables.
// Uses zero-copy CpuAddressable storage reads (Cow-safe: no unnecessary allocation).

use coeus_autograd::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float};

/// Clip the global L2 gradient norm across `params` to `max_norm`.
///
/// 1. Accumulate `Σ v² ` for every element of every parameter gradient.
/// 2. Compute `total_norm = sqrt(total_sq)` in native `T` precision.
/// 3. If `total_norm > max_norm`, scale all gradient elements by `max_norm / total_norm`.
///
/// Returns the pre-clip total norm (in `T` precision).
///
/// # Examples
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_optim::clip_grad_norm;
/// use coeus_tensor::Tensor;
///
/// let x: Var<f32> = Var::new(Tensor::from_slice(vec![2], &[1.0f32, 1.0]), true);
/// // Gradient [3.0, 4.0] has L2 norm 5.0; clipping to 2.5 scales by 0.5.
/// x.set_grad(Tensor::from_slice(vec![2], &[3.0f32, 4.0]));
///
/// let pre_norm = clip_grad_norm(&[x.clone()], 2.5f32);
/// assert!((pre_norm - 5.0).abs() < 1e-5);
///
/// let g = x.grad().unwrap();
/// assert!((g.as_slice()[0] - 1.5).abs() < 1e-5);
/// assert!((g.as_slice()[1] - 2.0).abs() < 1e-5);
/// ```
///
/// # Bounds
/// Requires `B::DeviceBuffer<T>` to implement `CpuAddressableStorage<T>` and
/// `CpuAddressableStorageMut<T>` so that gradient slices are directly readable and
/// writable without a device round-trip.
///
/// # Precision
/// All arithmetic executes in `T` — no implicit widening to `f64`.
pub fn clip_grad_norm<T, B>(params: &[Var<T, B>], max_norm: T) -> T
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    clip_grad_norm_iter(params.iter(), max_norm)
}

pub(crate) fn clip_grad_norm_iter<'a, T, B, I>(params: I, max_norm: T) -> T
where
    T: Float + 'a,
    B: coeus_ops::BackendOps<T> + Default + 'a,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    I: Iterator<Item = &'a Var<T, B>> + Clone,
{
    // Pass 1: sum of squared gradient elements (native T precision).
    let mut total_sq = T::zero();
    for param in params.clone() {
        let Some(ref grad_arc) = param.grad else {
            continue;
        };
        let grad = grad_arc.read();
        // grad tensors are always contiguous (constructed via zeros_on).
        let slice: &[T] = grad.as_slice();
        for &v in slice {
            total_sq += v * v;
        }
    }
    // sqrt in native T precision (Scalar::sqrt_val).
    let total_norm = total_sq.sqrt_val();

    // Pass 2: scale if over the limit.
    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        for param in params {
            let Some(ref grad_arc) = param.grad else {
                continue;
            };
            let grad = grad_arc.write();
            let slice: &mut [T] = grad.as_mut_slice();
            for v in slice {
                *v *= clip_coef;
            }
        }
    }

    total_norm
}
