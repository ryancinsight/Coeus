use crate::unfold_fold::validation::{product, require_signed_coordinates, window_count};
use coeus_core::{BackendError, Layout};

fn require_rank(
    operation: &'static str,
    layout: &Layout,
    expected: usize,
) -> Result<(), BackendError> {
    if layout.ndim() != expected || layout.strides().len() != expected {
        return Err(BackendError::Storage {
            operation,
            reason: format!(
                "expected rank {expected} with {expected} strides, received rank {} with {} strides",
                layout.ndim(),
                layout.strides().len()
            ),
        });
    }
    Ok(())
}

fn require_shape(
    operation: &'static str,
    layout: &Layout,
    expected: &[usize],
) -> Result<(), BackendError> {
    if layout.shape() == expected {
        Ok(())
    } else {
        Err(BackendError::ShapeMismatch {
            operation,
            lhs: layout.shape().to_vec(),
            rhs: expected.to_vec(),
        })
    }
}

fn require_storage_span(
    operation: &'static str,
    layout: &Layout,
    storage_len: usize,
) -> Result<(), BackendError> {
    if layout.shape().contains(&0) {
        return Ok(());
    }
    let last = layout.shape().iter().zip(layout.strides()).try_fold(
        layout.offset(),
        |offset, (&dimension, &stride)| {
            dimension
                .checked_sub(1)
                .and_then(|index| index.checked_mul(stride))
                .and_then(|span| offset.checked_add(span))
                .ok_or(BackendError::Overflow {
                    operation,
                    reason: "layout storage-span arithmetic overflow",
                })
        },
    )?;
    if last < storage_len {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: format!(
                "layout reaches physical element {last}, but storage contains {storage_len} elements"
            ),
        })
    }
}

fn require_writable_layout(operation: &'static str, layout: &Layout) -> Result<(), BackendError> {
    if layout.is_contiguous() {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: "output layout must be contiguous to prevent overlapping writes".to_owned(),
        })
    }
}

#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub(crate) fn unfold1d(
    input: &Layout,
    input_len: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &Layout,
    output_len: usize,
) -> Result<(), BackendError> {
    const OPERATION: &str = "unfold1d";
    require_rank(OPERATION, input, 3)?;
    require_rank(OPERATION, output, 3)?;
    let windows = window_count(
        OPERATION,
        input.shape()[2],
        kernel,
        stride,
        padding,
        dilation,
    )?;
    let channels = product(OPERATION, &[input.shape()[1], kernel])?;
    require_shape(OPERATION, output, &[input.shape()[0], channels, windows])?;
    let _ = product(OPERATION, output.shape())?;
    require_signed_coordinates(OPERATION, windows, kernel, stride, padding, dilation)?;
    require_storage_span(OPERATION, input, input_len)?;
    require_writable_layout(OPERATION, output)?;
    require_storage_span(OPERATION, output, output_len)
}

#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub(crate) fn fold1d(
    input: &Layout,
    input_len: usize,
    output_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &Layout,
    output_len: usize,
) -> Result<(), BackendError> {
    const OPERATION: &str = "fold1d";
    require_rank(OPERATION, input, 3)?;
    require_rank(OPERATION, output, 3)?;
    let windows = window_count(OPERATION, output_size, kernel, stride, padding, dilation)?;
    let channels = product(OPERATION, &[output.shape()[1], kernel])?;
    require_shape(OPERATION, input, &[output.shape()[0], channels, windows])?;
    require_shape(
        OPERATION,
        output,
        &[output.shape()[0], output.shape()[1], output_size],
    )?;
    let _ = product(OPERATION, input.shape())?;
    let _ = product(OPERATION, output.shape())?;
    require_signed_coordinates(OPERATION, windows, kernel, stride, padding, dilation)?;
    require_storage_span(OPERATION, input, input_len)?;
    require_writable_layout(OPERATION, output)?;
    require_storage_span(OPERATION, output, output_len)
}

#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub(crate) fn unfold2d(
    input: &Layout,
    input_len: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &Layout,
    output_len: usize,
) -> Result<(), BackendError> {
    const OPERATION: &str = "unfold2d";
    require_rank(OPERATION, input, 4)?;
    require_rank(OPERATION, output, 3)?;
    let output_h = window_count(
        OPERATION,
        input.shape()[2],
        kernel_h,
        stride_h,
        padding_h,
        dilation_h,
    )?;
    let output_w = window_count(
        OPERATION,
        input.shape()[3],
        kernel_w,
        stride_w,
        padding_w,
        dilation_w,
    )?;
    let channels = product(OPERATION, &[input.shape()[1], kernel_h, kernel_w])?;
    let locations = product(OPERATION, &[output_h, output_w])?;
    require_shape(OPERATION, output, &[input.shape()[0], channels, locations])?;
    let _ = product(OPERATION, output.shape())?;
    require_signed_coordinates(
        OPERATION, output_h, kernel_h, stride_h, padding_h, dilation_h,
    )?;
    require_signed_coordinates(
        OPERATION, output_w, kernel_w, stride_w, padding_w, dilation_w,
    )?;
    require_storage_span(OPERATION, input, input_len)?;
    require_writable_layout(OPERATION, output)?;
    require_storage_span(OPERATION, output, output_len)
}

#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
pub(crate) fn fold2d(
    input: &Layout,
    input_len: usize,
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
    output: &Layout,
    output_len: usize,
) -> Result<(), BackendError> {
    const OPERATION: &str = "fold2d";
    require_rank(OPERATION, input, 3)?;
    require_rank(OPERATION, output, 4)?;
    let input_h = window_count(
        OPERATION, output_h, kernel_h, stride_h, padding_h, dilation_h,
    )?;
    let input_w = window_count(
        OPERATION, output_w, kernel_w, stride_w, padding_w, dilation_w,
    )?;
    let channels = product(OPERATION, &[output.shape()[1], kernel_h, kernel_w])?;
    let locations = product(OPERATION, &[input_h, input_w])?;
    require_shape(OPERATION, input, &[output.shape()[0], channels, locations])?;
    require_shape(
        OPERATION,
        output,
        &[output.shape()[0], output.shape()[1], output_h, output_w],
    )?;
    let _ = product(OPERATION, input.shape())?;
    let _ = product(OPERATION, output.shape())?;
    require_signed_coordinates(
        OPERATION, input_h, kernel_h, stride_h, padding_h, dilation_h,
    )?;
    require_signed_coordinates(
        OPERATION, input_w, kernel_w, stride_w, padding_w, dilation_w,
    )?;
    require_storage_span(OPERATION, input, input_len)?;
    require_writable_layout(OPERATION, output)?;
    require_storage_span(OPERATION, output, output_len)
}

#[cfg(test)]
mod tests {
    use super::{require_storage_span, unfold1d};
    use coeus_core::{BackendError, Layout};
    use smallvec::smallvec;

    #[test]
    fn rejects_layout_reaching_past_storage() {
        let layout = Layout::from_shape_strides([2, 2].into(), smallvec![3, 1], 0);
        assert!(matches!(
            require_storage_span("unfold1d", &layout, 4),
            Err(BackendError::Storage { .. })
        ));
    }

    #[test]
    fn rejects_direct_dispatch_shape_mismatch() {
        let input = Layout::new([1, 1, 4].into());
        let output = Layout::new([1, 2, 2].into());
        assert!(matches!(
            unfold1d(&input, 4, 2, 1, 0, 1, &output, 4),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }
}
