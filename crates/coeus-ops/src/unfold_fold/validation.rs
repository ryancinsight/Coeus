use coeus_core::BackendError;

pub(crate) fn shape<const RANK: usize>(
    operation: &'static str,
    dimensions: &[usize],
) -> Result<[usize; RANK], BackendError> {
    dimensions.try_into().map_err(|_| BackendError::Storage {
        operation,
        reason: format!("expected rank {RANK}, received rank {}", dimensions.len()),
    })
}

pub(crate) fn product(operation: &'static str, factors: &[usize]) -> Result<usize, BackendError> {
    factors.iter().try_fold(1usize, |value, &factor| {
        value.checked_mul(factor).ok_or(BackendError::Overflow {
            operation,
            reason: "output-shape arithmetic overflow",
        })
    })
}

pub(crate) fn window_count(
    operation: &'static str,
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, BackendError> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(BackendError::Storage {
            operation,
            reason: "kernel, stride, and dilation must be nonzero".to_owned(),
        });
    }
    let effective_kernel = dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or(BackendError::Overflow {
            operation,
            reason: "effective-kernel arithmetic overflow",
        })?;
    let doubled_padding = padding.checked_mul(2).ok_or(BackendError::Overflow {
        operation,
        reason: "padding arithmetic overflow",
    })?;
    let padded = input
        .checked_add(doubled_padding)
        .ok_or(BackendError::Overflow {
            operation,
            reason: "padded-input arithmetic overflow",
        })?;
    let covered = padded
        .checked_sub(effective_kernel)
        .ok_or_else(|| BackendError::Storage {
            operation,
            reason: "effective kernel exceeds the padded input".to_owned(),
        })?;
    covered
        .checked_div(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or(BackendError::Overflow {
            operation,
            reason: "window-count arithmetic overflow",
        })
}

pub(crate) fn require_signed_coordinates(
    operation: &'static str,
    output_width: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(), BackendError> {
    let maximum = output_width
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|value| {
            kernel
                .saturating_sub(1)
                .checked_mul(dilation)
                .and_then(|kernel_span| value.checked_add(kernel_span))
        })
        .and_then(|value| value.checked_add(padding))
        .ok_or(BackendError::Overflow {
            operation,
            reason: "signed-coordinate arithmetic overflow",
        })?;
    if maximum <= isize::MAX as usize {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: "window coordinates exceed the signed host index range".to_owned(),
        })
    }
}
