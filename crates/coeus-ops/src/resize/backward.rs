//! Reverse-mode spatial resize: the gradient of every forward mode.

use super::taps::{ResizeMode, TapTable};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_tensor::Tensor;

/// Gradient of [`resize_1d`] with respect to its input.
///
/// `input_shape` is the shape of the resized input (`[N, C, L]`) and
/// `grad_output` the `[N, C, new_L]` upstream gradient.
///
/// # Panics
/// If `input_shape` is not rank-3 or `grad_output` does not match the shape the
/// forward pass would have produced.
pub fn resize_1d_backward<T, B>(
    input_shape: &[usize],
    new_l: usize,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        3,
        "resize_1d_backward: expected [N,C,L] input shape"
    );
    let scale = f64::from_count(input_shape[2]) / f64::from_count(new_l);
    resize_1d_backward_with_scale(input_shape, new_l, scale, mode, grad_output)
}

/// [`resize_1d_backward`] with the forward pass's explicit sampling step, so
/// the scatter uses the same taps the forward pass applied.
pub fn resize_1d_backward_with_scale<T, B>(
    input_shape: &[usize],
    new_l: usize,
    scale: f64,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        3,
        "resize_1d_backward: expected [N,C,L] input shape"
    );
    let (n, c, l) = (input_shape[0], input_shape[1], input_shape[2]);
    let expected = [n, c, new_l];
    assert_eq!(
        grad_output.shape(),
        expected,
        "resize_1d_backward: gradient shape must match the resized output"
    );

    let grad_output = grad_output.to_contiguous()?;
    let upstream = grad_output.as_slice();
    let mut gradient = vec![T::ZERO; n * c * l];
    let mut taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * l;
        for out_index in 0..new_l {
            taps.fill(out_index, l, new_l, mode, scale);
            let g = upstream[plane * new_l + out_index];
            for (index, weight) in taps.iter() {
                gradient[base + index] += g * T::from_f64(weight);
            }
        }
    }

    Tensor::from_slice_on(input_shape.to_vec(), &gradient, &B::default())
}

/// Gradient of [`resize_2d`] with respect to its input.
///
/// `input_shape` is the shape of the resized input (`[N, C, H, W]`) and
/// `grad_output` the `[N, C, new_H, new_W]` upstream gradient.
///
/// # Panics
/// If `input_shape` is not rank-4 or `grad_output` does not match the shape the
/// forward pass would have produced.
pub fn resize_2d_backward<T, B>(
    input_shape: &[usize],
    new_h: usize,
    new_w: usize,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        4,
        "resize_2d_backward: expected [N,C,H,W] input shape"
    );
    let (h, w) = (input_shape[2], input_shape[3]);
    resize_2d_backward_with_scale(
        input_shape,
        new_h,
        new_w,
        f64::from_count(h) / f64::from_count(new_h),
        f64::from_count(w) / f64::from_count(new_w),
        mode,
        grad_output,
    )
}

/// [`resize_2d_backward`] with the forward pass's explicit sampling steps.
pub fn resize_2d_backward_with_scale<T, B>(
    input_shape: &[usize],
    new_h: usize,
    new_w: usize,
    scale_h: f64,
    scale_w: f64,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        4,
        "resize_2d_backward: expected [N,C,H,W] input shape"
    );
    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let expected = [n, c, new_h, new_w];
    assert_eq!(
        grad_output.shape(),
        expected,
        "resize_2d_backward: gradient shape must match the resized output"
    );

    let grad_output = grad_output.to_contiguous()?;
    let upstream = grad_output.as_slice();
    let plane_in = h * w;
    let plane_out = new_h * new_w;
    let mut gradient = vec![T::ZERO; n * c * plane_in];
    let mut y_taps = TapTable::default();
    let mut x_taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * plane_in;
        for out_y in 0..new_h {
            y_taps.fill(out_y, h, new_h, mode, scale_h);
            for out_x in 0..new_w {
                x_taps.fill(out_x, w, new_w, mode, scale_w);
                let g = upstream[plane * plane_out + out_y * new_w + out_x];
                for (y, y_weight) in y_taps.iter() {
                    for (x, x_weight) in x_taps.iter() {
                        gradient[base + y * w + x] += g * T::from_f64(y_weight * x_weight);
                    }
                }
            }
        }
    }

    Tensor::from_slice_on(input_shape.to_vec(), &gradient, &B::default())
}
