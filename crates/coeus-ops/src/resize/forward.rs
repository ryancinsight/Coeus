//! Forward spatial resize: nearest, bilinear, bicubic and area.

use super::taps::{ResizeMode, TapTable};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_tensor::Tensor;

/// Resize a 1-D spatial tensor `[N, C, L]` to `[N, C, new_L]`.
///
/// # Panics
/// If `input` is not rank-3 or `new_l` is zero.
pub fn resize_1d<T, B>(
    input: &Tensor<T, B>,
    new_l: usize,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 3, "resize_1d: expected [N,C,L] tensor");
    let scale = f64::from_count(shape[2]) / f64::from_count(new_l);
    resize_1d_with_scale(input, new_l, scale, mode)
}

/// [`resize_1d`] with an explicit sampling step of `scale` source units per
/// output position — pass `1 / scale_factor` to reproduce
/// `torch.nn.functional.interpolate(..., scale_factor=..)` for a fractional
/// factor, where torch keeps the requested factor instead of deriving it from
/// the rounded output size.
///
/// # Panics
/// If `input` is not rank-3, `new_l` is zero, or `scale` is not positive and
/// finite.
pub fn resize_1d_with_scale<T, B>(
    input: &Tensor<T, B>,
    new_l: usize,
    scale: f64,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 3, "resize_1d: expected [N,C,L] tensor");
    assert!(new_l > 0, "resize_1d: new length must be non-zero");
    assert!(
        scale > 0.0 && scale.is_finite(),
        "resize_1d: scale must be positive and finite, got {scale}"
    );
    let (n, c, l) = (shape[0], shape[1], shape[2]);

    let input = input.to_contiguous()?;
    let source = input.as_slice();
    let mut output = vec![T::ZERO; n * c * new_l];
    let mut taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * l;
        for (out_index, slot) in output[plane * new_l..(plane + 1) * new_l]
            .iter_mut()
            .enumerate()
        {
            taps.fill(out_index, l, new_l, mode, scale);
            let mut value = T::ZERO;
            for (index, weight) in taps.iter() {
                value += source[base + index] * T::from_f64(weight);
            }
            *slot = value;
        }
    }

    Tensor::from_slice_on(vec![n, c, new_l], &output, &B::default())
}

/// Resize a 2-D spatial tensor `[N, C, H, W]` to `[N, C, new_H, new_W]`.
///
/// # Panics
/// If `input` is not rank-4, or `new_h`/`new_w` is zero.
pub fn resize_2d<T, B>(
    input: &Tensor<T, B>,
    new_h: usize,
    new_w: usize,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "resize_2d: expected [N,C,H,W] tensor");
    assert!(
        new_h > 0 && new_w > 0,
        "resize_2d: new extents must be non-zero"
    );
    let (h, w) = (shape[2], shape[3]);
    resize_2d_with_scale(
        input,
        new_h,
        new_w,
        f64::from_count(h) / f64::from_count(new_h),
        f64::from_count(w) / f64::from_count(new_w),
        mode,
    )
}

/// [`resize_2d`] with explicit per-axis sampling steps — see
/// [`resize_1d_with_scale`] for why a `scale_factor` caller must not derive
/// them from the output size.
///
/// # Panics
/// If `input` is not rank-4, `new_h`/`new_w` is zero, or a scale is not
/// positive and finite.
pub fn resize_2d_with_scale<T, B>(
    input: &Tensor<T, B>,
    new_h: usize,
    new_w: usize,
    scale_h: f64,
    scale_w: f64,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "resize_2d: expected [N,C,H,W] tensor");
    assert!(
        new_h > 0 && new_w > 0,
        "resize_2d: new extents must be non-zero"
    );
    assert!(
        scale_h > 0.0 && scale_h.is_finite() && scale_w > 0.0 && scale_w.is_finite(),
        "resize_2d: scales must be positive and finite, got {scale_h} and {scale_w}"
    );
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

    let input = input.to_contiguous()?;
    let source = input.as_slice();
    let plane_in = h * w;
    let plane_out = new_h * new_w;
    let mut output = vec![T::ZERO; n * c * plane_out];
    let mut y_taps = TapTable::default();
    let mut x_taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * plane_in;
        for out_y in 0..new_h {
            y_taps.fill(out_y, h, new_h, mode, scale_h);
            for out_x in 0..new_w {
                x_taps.fill(out_x, w, new_w, mode, scale_w);
                let mut value = T::ZERO;
                for (y, y_weight) in y_taps.iter() {
                    for (x, x_weight) in x_taps.iter() {
                        value += source[base + y * w + x] * T::from_f64(y_weight * x_weight);
                    }
                }
                output[plane * plane_out + out_y * new_w + out_x] = value;
            }
        }
    }

    Tensor::from_slice_on(vec![n, c, new_h, new_w], &output, &B::default())
}
