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
    // Pass 1: sum of squared gradient elements (native T precision).
    let mut total_sq = T::zero();
    for param in params {
        let Some(ref grad_arc) = param.grad else {
            continue;
        };
        let grad = grad_arc.read();
        // grad tensors are always contiguous (constructed via zeros_on).
        let slice: &[T] = grad.as_slice();
        for &v in slice {
            total_sq = total_sq + v * v;
        }
    }
    // sqrt in native T precision (Scalar::sqrt_val).
    let total_norm = total_sq.sqrt_val();

    // Pass 2: scale if over the limit.
    let one = T::one();
    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        let backend = B::default();
        for param in params {
            let Some(ref grad_arc) = param.grad else {
                continue;
            };
            let grad = grad_arc.write();
            let slice: &mut [T] = grad.as_mut_slice();
            for v in slice {
                *v = *v * clip_coef;
            }
        }
        let _ = (one, backend); // suppress unused warnings when no params have grad
    }

    total_norm
}
