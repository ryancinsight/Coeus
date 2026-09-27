// ── Cross-product host fold ──
//
// Pure host-memory math shared by every backend without an on-device
// cross-product seam: `CpuBackend` reads/writes its addressable storage
// directly; `CudaBackend` and `HephaestusBackend<RocmProvider/MetalProvider>`
// copy through host memory first via `ComputeBackend::copy_to_host` /
// `copy_to_device`, since hephaestus has no `CrossProductOps` for those
// vendors yet (ADR 0077). `WgpuBackend` uses this same function only for the
// non-contiguous / non-last-axis layouts its on-device seam cannot express.

use coeus_core::{Layout, Scalar};

/// Fold the per-channel 3-vector cross product of `a` and `b` along `dim`
/// into `out`, all three already resident in host memory.
///
/// Right-handed convention: `(a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y -
/// a_y b_x)` per slice, matching `torch.cross` / `numpy.cross`.
///
/// Assumes `a`, `b`, and `out` each hold `layout.numel()` elements, that
/// `layout.shape()[dim] == 3`, and that `a`, `b`, and `layout` are
/// contiguous row-major with `layout`'s shape (the caller — `CrossOps`
/// implementors — materializes to this form before calling in).
pub fn cross_fold<T: Scalar>(a: &[T], b: &[T], layout: &Layout, dim: usize, out: &mut [T]) {
    let shape = layout.shape();
    let pre: usize = shape[..dim].iter().product();
    let post: usize = shape[dim + 1..].iter().product();
    let stride_pre = 3 * post;
    let stride_k = post;

    for pre_idx in 0..pre {
        for post_idx in 0..post {
            let base = pre_idx * stride_pre + post_idx;
            let ax = base;
            let ay = base + stride_k;
            let az = base + 2 * stride_k;
            out[ax] = a[ay] * b[az] - a[az] * b[ay];
            out[ay] = a[az] * b[ax] - a[ax] * b[az];
            out[az] = a[ax] * b[ay] - a[ay] * b[ax];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_vectors_along_last_axis() {
        let layout = Layout::new(vec![3].into());
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        let mut out = [0.0_f32; 3];
        cross_fold(&a, &b, &layout, 0, &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0]);
    }
}
