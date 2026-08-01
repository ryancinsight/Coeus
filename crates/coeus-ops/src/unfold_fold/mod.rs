//! Sliding-window extraction and adjoint accumulation.

use crate::BackendOps;
use coeus_core::{BackendError, Scalar};
use coeus_tensor::Tensor;

pub(crate) mod validation;

use validation::{product, shape, window_count};

/// Extract one-dimensional sliding windows from `[N, C, L]`.
///
/// # Errors
///
/// Returns the backend-associated error when geometry, shape arithmetic, or
/// backend dispatch validation fails.
pub fn unfold1d<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    const OPERATION: &str = "unfold1d";
    let [n, c, length] = shape::<3>(OPERATION, input.shape())?;
    let windows = window_count(OPERATION, length, kernel_size, stride, padding, dilation)?;
    let channels = product(OPERATION, &[c, kernel_size])?;
    let _ = product(OPERATION, &[n, channels, windows])?;
    let mut output = Tensor::alloc_on([n, channels, windows], backend);
    let (output_storage, output_layout) = output.storage_mut_and_layout();
    backend.unfold1d(
        input.storage(),
        input.layout(),
        kernel_size,
        stride,
        padding,
        dilation,
        output_storage,
        output_layout,
    )?;
    Ok(output)
}

/// Accumulate one-dimensional windows into `[N, C, output_size]`.
///
/// # Errors
///
/// Returns the backend-associated error when geometry, shape arithmetic, or
/// backend dispatch validation fails.
pub fn fold1d<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    const OPERATION: &str = "fold1d";
    let [n, combined_channels, windows] = shape::<3>(OPERATION, input.shape())?;
    if kernel_size == 0 || combined_channels % kernel_size != 0 {
        return Err(BackendError::Storage {
            operation: OPERATION,
            reason: "combined channels must be divisible by the nonzero kernel size".to_owned(),
        }
        .into());
    }
    let expected_windows = window_count(
        OPERATION,
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
    )?;
    if windows != expected_windows {
        return Err(BackendError::ShapeMismatch {
            operation: OPERATION,
            lhs: vec![windows],
            rhs: vec![expected_windows],
        }
        .into());
    }
    let channels = combined_channels / kernel_size;
    let _ = product(OPERATION, &[n, channels, output_size])?;
    let mut output = Tensor::zeros_on([n, channels, output_size], backend);
    let (output_storage, output_layout) = output.storage_mut_and_layout();
    backend.fold1d(
        input.storage(),
        input.layout(),
        output_size,
        kernel_size,
        stride,
        padding,
        dilation,
        output_storage,
        output_layout,
    )?;
    Ok(output)
}

/// Extract two-dimensional sliding windows from `[N, C, H, W]`.
///
/// # Errors
///
/// Returns the backend-associated error when geometry, shape arithmetic, or
/// backend dispatch validation fails.
#[allow(clippy::too_many_arguments)]
pub fn unfold2d<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    const OPERATION: &str = "unfold2d";
    let [n, c, height, width] = shape::<4>(OPERATION, input.shape())?;
    let output_h = window_count(OPERATION, height, kernel_h, stride_h, padding_h, dilation_h)?;
    let output_w = window_count(OPERATION, width, kernel_w, stride_w, padding_w, dilation_w)?;
    let channels = product(OPERATION, &[c, kernel_h, kernel_w])?;
    let locations = product(OPERATION, &[output_h, output_w])?;
    let _ = product(OPERATION, &[n, channels, locations])?;
    let mut output = Tensor::alloc_on([n, channels, locations], backend);
    let (output_storage, output_layout) = output.storage_mut_and_layout();
    backend.unfold2d(
        input.storage(),
        input.layout(),
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_storage,
        output_layout,
    )?;
    Ok(output)
}

/// Accumulate two-dimensional windows into `[N, C, output_h, output_w]`.
///
/// # Errors
///
/// Returns the backend-associated error when geometry, shape arithmetic, or
/// backend dispatch validation fails.
#[allow(clippy::too_many_arguments)]
pub fn fold2d<T: Scalar, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    const OPERATION: &str = "fold2d";
    let [n, combined_channels, locations] = shape::<3>(OPERATION, input.shape())?;
    let kernel_area = product(OPERATION, &[kernel_h, kernel_w])?;
    if kernel_area == 0 || combined_channels % kernel_area != 0 {
        return Err(BackendError::Storage {
            operation: OPERATION,
            reason: "combined channels must be divisible by the nonzero kernel area".to_owned(),
        }
        .into());
    }
    let input_h = window_count(
        OPERATION, output_h, kernel_h, stride_h, padding_h, dilation_h,
    )?;
    let input_w = window_count(
        OPERATION, output_w, kernel_w, stride_w, padding_w, dilation_w,
    )?;
    let expected_locations = product(OPERATION, &[input_h, input_w])?;
    if locations != expected_locations {
        return Err(BackendError::ShapeMismatch {
            operation: OPERATION,
            lhs: vec![locations],
            rhs: vec![expected_locations],
        }
        .into());
    }
    let channels = combined_channels / kernel_area;
    let _ = product(OPERATION, &[n, channels, output_h, output_w])?;
    let mut output = Tensor::zeros_on([n, channels, output_h, output_w], backend);
    let (output_storage, output_layout) = output.storage_mut_and_layout();
    backend.fold2d(
        input.storage(),
        input.layout(),
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_storage,
        output_layout,
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{unfold1d, window_count};
    use coeus_core::{BackendError, SequentialBackend};
    use coeus_tensor::Tensor;

    #[test]
    fn rejects_zero_stride_before_allocation() {
        let backend = SequentialBackend::new();
        let input = Tensor::from_slice_on([1, 1, 3], &[1.0_f32, 2.0, 3.0], &backend);
        assert!(matches!(
            unfold1d(&input, 2, 0, 0, 1, &backend),
            Err(BackendError::Storage {
                operation: "unfold1d",
                ..
            })
        ));
    }

    #[test]
    fn rejects_geometry_overflow() {
        assert!(matches!(
            window_count("unfold1d", 1, usize::MAX, 1, 0, 2),
            Err(BackendError::Overflow {
                operation: "unfold1d",
                reason: "effective-kernel arithmetic overflow",
            })
        ));
    }
}
