use crate::backend::WgpuBackendError;
use crate::backend::{checked_numel, checked_u32_parameter, checked_workgroup_count};
use coeus_core::{BackendError, Layout};

pub(super) fn require_storage_span<T>(
    operation: &'static str,
    buffer: &wgpu::Buffer,
    layout: &Layout,
) -> Result<(), WgpuBackendError> {
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
    let element_size = std::mem::size_of::<T>();
    let required_bytes = last
        .checked_add(1)
        .and_then(|elements| elements.checked_mul(element_size))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(BackendError::Overflow {
            operation,
            reason: "layout byte-span arithmetic overflow",
        })?;
    if last > u32::MAX as usize {
        return Err(WgpuBackendError::AbiValueOutOfRange {
            operation,
            parameter: "layout physical index",
            value: last,
        });
    }
    if required_bytes <= buffer.size() {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: format!(
                "layout requires {required_bytes} bytes, but buffer contains {} bytes",
                buffer.size()
            ),
        }
        .into())
    }
}

pub(super) fn require_writable_layout(
    operation: &'static str,
    layout: &Layout,
) -> Result<(), WgpuBackendError> {
    if layout.is_contiguous() {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: "output layout must be contiguous to prevent overlapping writes".to_owned(),
        }
        .into())
    }
}

pub(super) fn require_signed_coordinates(
    operation: &'static str,
    output_width: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(), WgpuBackendError> {
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
    if i32::try_from(maximum).is_ok() {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: "window coordinates exceed the WGSL signed integer range".to_owned(),
        }
        .into())
    }
}

pub(super) fn require_rank(
    operation: &'static str,
    layout: &Layout,
    expected: usize,
) -> Result<(), WgpuBackendError> {
    let rank = layout.shape().len();
    if rank == expected {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: format!("expected rank {expected}, received rank {rank}"),
        }
        .into())
    }
}

pub(super) fn require_dimension(
    operation: &'static str,
    layout: &Layout,
    axis: usize,
    expected: usize,
) -> Result<(), WgpuBackendError> {
    let actual = layout
        .shape()
        .get(axis)
        .copied()
        .ok_or_else(|| BackendError::Storage {
            operation,
            reason: format!("layout has no axis {axis}"),
        })?;
    if actual == expected {
        Ok(())
    } else {
        Err(BackendError::Storage {
            operation,
            reason: format!("axis {axis} expected dimension {expected}, received {actual}"),
        }
        .into())
    }
}

pub(super) fn parameter(
    operation: &'static str,
    value: usize,
    name: &'static str,
    nonzero: bool,
) -> Result<u32, WgpuBackendError> {
    if nonzero && value == 0 {
        return Err(BackendError::Storage {
            operation,
            reason: format!("parameter {name} must be nonzero"),
        }
        .into());
    }
    checked_u32_parameter(operation, name, value)
}

pub(super) fn output_width(
    operation: &'static str,
    width: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, WgpuBackendError> {
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or(BackendError::Overflow {
            operation,
            reason: "effective-kernel arithmetic overflow",
        })?;
    let doubled_padding = padding.checked_mul(2).ok_or(BackendError::Overflow {
        operation,
        reason: "padding arithmetic overflow",
    })?;
    let padded = width
        .checked_add(doubled_padding)
        .ok_or(BackendError::Overflow {
            operation,
            reason: "padded-width arithmetic overflow",
        })?;
    let covered = padded
        .checked_sub(effective_kernel)
        .ok_or_else(|| BackendError::Storage {
            operation,
            reason: "effective kernel exceeds the padded input width".to_owned(),
        })?;
    covered
        .checked_div(stride)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            BackendError::Storage {
                operation,
                reason: "stride must be nonzero and output width must fit usize".to_owned(),
            }
            .into()
        })
}

pub(super) fn dispatch_count(
    operation: &'static str,
    output_layout: &Layout,
) -> Result<(usize, u32), WgpuBackendError> {
    let total = checked_numel(operation, output_layout.shape())?;
    let workgroups = checked_workgroup_count(operation, total)?;
    Ok((total, workgroups))
}

pub(super) fn checked_product(
    operation: &'static str,
    reason: &'static str,
    factors: &[usize],
) -> Result<usize, WgpuBackendError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| BackendError::Overflow { operation, reason }.into())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        dispatch_count, output_width, parameter, require_rank, require_signed_coordinates,
    };
    use crate::backend::WgpuBackendError;
    use coeus_core::{BackendError, Layout};

    #[test]
    fn rejects_zero_stride_without_panicking() {
        assert!(matches!(
            output_width("unfold1d", 8, 3, 0, 0, 1),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "unfold1d",
                ..
            }))
        ));
    }

    #[test]
    fn rejects_kernel_larger_than_padded_input() {
        assert!(matches!(
            output_width("unfold1d", 2, 3, 1, 0, 1),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "unfold1d",
                ..
            }))
        ));
    }

    #[test]
    fn rejects_effective_kernel_overflow() {
        assert!(matches!(
            output_width("unfold1d", 8, usize::MAX, 1, 0, 2),
            Err(WgpuBackendError::Validation(BackendError::Overflow {
                operation: "unfold1d",
                reason: "effective-kernel arithmetic overflow",
            }))
        ));
    }

    #[test]
    fn rejects_padding_overflow() {
        assert!(matches!(
            output_width("unfold1d", 8, 3, 1, usize::MAX, 1),
            Err(WgpuBackendError::Validation(BackendError::Overflow {
                operation: "unfold1d",
                reason: "padding arithmetic overflow",
            }))
        ));
    }

    #[test]
    fn rejects_output_element_count_overflow() {
        let layout = Layout::new([usize::MAX, 2].into());
        assert!(matches!(
            dispatch_count("unfold1d", &layout),
            Err(WgpuBackendError::Validation(BackendError::Overflow {
                operation: "unfold1d",
                reason: "output element-count arithmetic overflow",
            }))
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_parameter_outside_wgsl_abi() {
        let value = usize::try_from(u64::from(u32::MAX) + 1).expect("test value fits usize");
        assert!(matches!(
            parameter("unfold1d", value, "kernel_size", true),
            Err(WgpuBackendError::AbiValueOutOfRange {
                operation: "unfold1d",
                parameter: "kernel_size",
                value: rejected,
            }) if rejected == value
        ));
    }

    #[test]
    fn rejects_zero_nonzero_parameter() {
        assert!(matches!(
            parameter("unfold1d", 0, "dilation", true),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "unfold1d",
                ..
            }))
        ));
    }

    #[test]
    fn rejects_wrong_rank() {
        let layout = Layout::new([2, 3].into());
        assert!(matches!(
            require_rank("unfold1d", &layout, 3),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "unfold1d",
                ..
            }))
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_coordinates_outside_wgsl_signed_range() {
        let width = usize::try_from(i64::from(i32::MAX) + 2).expect("test value fits usize");
        assert!(matches!(
            require_signed_coordinates("unfold1d", width, 1, 1, 0, 1),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "unfold1d",
                ..
            }))
        ));
    }
}
