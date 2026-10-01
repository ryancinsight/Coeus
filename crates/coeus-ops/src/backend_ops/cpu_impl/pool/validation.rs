use coeus_core::{BackendError, Layout, Storage};

pub(crate) fn readable_storage<'a, T, S: Storage<T>>(
    operation: &'static str,
    storage: &'a S,
) -> Result<&'a [T], BackendError> {
    storage.try_as_slice().ok_or_else(|| BackendError::Storage {
        operation,
        reason: "input storage contains uninitialized elements".to_owned(),
    })
}

#[derive(Clone, Copy)]
pub(crate) struct PoolParameters {
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
    pub(crate) dilation: usize,
}

pub(crate) struct PoolCounts {
    pub(crate) input: usize,
    pub(crate) output: usize,
}

pub(crate) fn validate_forward<const SPATIAL_DIMENSIONS: usize>(
    operation: &'static str,
    input: &Layout,
    input_len: usize,
    output: &Layout,
    output_len: usize,
    parameters: PoolParameters,
) -> Result<usize, BackendError> {
    validate_geometry::<SPATIAL_DIMENSIONS>(operation, input, output, parameters)?;
    validate_storage(operation, input, input_len, false)?;
    validate_storage(operation, output, output_len, true)?;
    checked_numel(operation, output.shape())
}

pub(crate) fn validate_backward<const SPATIAL_DIMENSIONS: usize>(
    operation: &'static str,
    input: &Layout,
    input_len: usize,
    grad_output: &Layout,
    grad_output_len: usize,
    grad_input: &Layout,
    grad_input_len: usize,
    parameters: PoolParameters,
) -> Result<PoolCounts, BackendError> {
    validate_geometry::<SPATIAL_DIMENSIONS>(operation, input, grad_output, parameters)?;
    if input.shape() != grad_input.shape() {
        return Err(BackendError::ShapeMismatch {
            operation,
            lhs: input.shape().to_vec(),
            rhs: grad_input.shape().to_vec(),
        });
    }
    validate_storage(operation, input, input_len, false)?;
    validate_storage(operation, grad_output, grad_output_len, false)?;
    validate_storage(operation, grad_input, grad_input_len, true)?;
    Ok(PoolCounts {
        input: checked_numel(operation, input.shape())?,
        output: checked_numel(operation, grad_output.shape())?,
    })
}

fn validate_geometry<const SPATIAL_DIMENSIONS: usize>(
    operation: &'static str,
    input: &Layout,
    output: &Layout,
    parameters: PoolParameters,
) -> Result<(), BackendError> {
    let expected_rank = SPATIAL_DIMENSIONS + 2;
    for rank in [input.ndim(), output.ndim()] {
        if rank != expected_rank {
            return Err(BackendError::LayoutRankMismatch {
                operation,
                lhs: rank,
                rhs: expected_rank,
            });
        }
    }
    let rank = input.ndim();
    if input.strides().len() != rank || output.strides().len() != rank {
        return Err(BackendError::Storage {
            operation,
            reason: "layout shape and stride ranks differ".to_owned(),
        });
    }
    if input.shape()[..2] != output.shape()[..2] {
        return Err(BackendError::ShapeMismatch {
            operation,
            lhs: input.shape()[..2].to_vec(),
            rhs: output.shape()[..2].to_vec(),
        });
    }

    if parameters.kernel_size == 0 || parameters.stride == 0 || parameters.dilation == 0 {
        return Err(BackendError::Storage {
            operation,
            reason: "kernel size, stride, and dilation must be nonzero".to_owned(),
        });
    }

    let kernel_extent = parameters
        .kernel_size
        .checked_sub(1)
        .and_then(|extent| extent.checked_mul(parameters.dilation))
        .and_then(|extent| extent.checked_add(1))
        .ok_or(BackendError::Overflow {
            operation,
            reason: "pooling kernel extent overflow",
        })?;
    let twice_padding = parameters
        .padding
        .checked_mul(2)
        .ok_or(BackendError::Overflow {
            operation,
            reason: "pooling padding overflow",
        })?;
    let signed_limit = isize::MAX as usize;
    if parameters.padding > signed_limit
        || parameters.stride > signed_limit
        || parameters.dilation > signed_limit
        || parameters.kernel_size > signed_limit
    {
        return Err(BackendError::Overflow {
            operation,
            reason: "pooling parameters exceed signed coordinate range",
        });
    }

    for axis in 2..rank {
        let input_extent = input.shape()[axis];
        let padded_extent =
            input_extent
                .checked_add(twice_padding)
                .ok_or(BackendError::Overflow {
                    operation,
                    reason: "padded input extent overflow",
                })?;
        let expected_output = if padded_extent < kernel_extent {
            0
        } else {
            (padded_extent - kernel_extent)
                .checked_div(parameters.stride)
                .and_then(|extent| extent.checked_add(1))
                .ok_or(BackendError::Overflow {
                    operation,
                    reason: "pooling output extent overflow",
                })?
        };
        if output.shape()[axis] != expected_output {
            return Err(BackendError::ShapeMismatch {
                operation,
                lhs: output.shape().to_vec(),
                rhs: input.shape().to_vec(),
            });
        }

        if input_extent > signed_limit || expected_output > signed_limit {
            return Err(BackendError::Overflow {
                operation,
                reason: "pooling coordinate exceeds signed range",
            });
        }
        let forward_coordinate = expected_output
            .saturating_sub(1)
            .checked_mul(parameters.stride)
            .and_then(|coordinate| {
                parameters
                    .kernel_size
                    .checked_sub(1)
                    .and_then(|kernel| kernel.checked_mul(parameters.dilation))
                    .and_then(|kernel| coordinate.checked_add(kernel))
            })
            .ok_or(BackendError::Overflow {
                operation,
                reason: "pooling forward coordinate overflow",
            })?;
        let backward_coordinate = input_extent
            .saturating_sub(1)
            .checked_add(parameters.padding)
            .ok_or(BackendError::Overflow {
                operation,
                reason: "pooling backward coordinate overflow",
            })?;
        if forward_coordinate > signed_limit || backward_coordinate > signed_limit {
            return Err(BackendError::Overflow {
                operation,
                reason: "pooling coordinate exceeds signed range",
            });
        }
    }

    checked_numel(operation, output.shape())?;
    checked_numel(operation, input.shape())?;
    Ok(())
}

fn validate_storage(
    operation: &'static str,
    layout: &Layout,
    storage_len: usize,
    writable: bool,
) -> Result<(), BackendError> {
    let rank = layout.ndim();
    if layout.strides().len() != rank {
        return Err(BackendError::Storage {
            operation,
            reason: "layout shape and stride ranks differ".to_owned(),
        });
    }

    if layout.shape().contains(&0) {
        if layout.offset() > storage_len {
            return Err(BackendError::Storage {
                operation,
                reason: "empty layout offset exceeds storage length".to_owned(),
            });
        }
        return Ok(());
    }

    let last = layout
        .shape()
        .iter()
        .zip(layout.strides())
        .try_fold(layout.offset(), |offset, (&extent, &stride)| {
            (extent - 1)
                .checked_mul(stride)
                .and_then(|span| offset.checked_add(span))
        })
        .ok_or(BackendError::Overflow {
            operation,
            reason: "layout storage-span arithmetic overflow",
        })?;
    if last >= storage_len {
        return Err(BackendError::Storage {
            operation,
            reason: format!(
                "layout reaches physical element {last}, but storage contains {storage_len} elements"
            ),
        });
    }

    if writable {
        require_injective(operation, layout)?;
    }
    Ok(())
}

fn require_injective(operation: &'static str, layout: &Layout) -> Result<(), BackendError> {
    let mut axes = smallvec::SmallVec::<[(usize, usize); 5]>::new();
    for (&extent, &stride) in layout.shape().iter().zip(layout.strides()) {
        if extent > 1 {
            axes.push((stride, extent));
        }
    }
    axes.sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered = 1usize;
    for (stride, extent) in axes {
        if stride < covered {
            return Err(BackendError::AliasedLayout { operation });
        }
        covered = covered
            .checked_add(
                (extent - 1)
                    .checked_mul(stride)
                    .ok_or(BackendError::Overflow {
                        operation,
                        reason: "layout injectivity arithmetic overflow",
                    })?,
            )
            .ok_or(BackendError::Overflow {
                operation,
                reason: "layout injectivity arithmetic overflow",
            })?;
    }
    Ok(())
}

fn checked_numel(operation: &'static str, shape: &[usize]) -> Result<usize, BackendError> {
    shape.iter().try_fold(1usize, |numel, &extent| {
        numel.checked_mul(extent).ok_or(BackendError::Overflow {
            operation,
            reason: "pooling element count overflow",
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{validate_forward, PoolParameters};
    use coeus_core::{
        BackendError, ComputeBackend, CpuAddressableStorage, CpuStorage, Layout, SequentialBackend,
    };

    const PARAMETERS: PoolParameters = PoolParameters {
        kernel_size: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
    };

    #[test]
    fn rejects_output_layout_outside_storage() {
        let input = Layout::new([1, 1, 2].into());
        let output = Layout::new([1, 1, 2].into());

        assert!(matches!(
            validate_forward::<1>("max_pool1d", &input, 2, &output, 1, PARAMETERS),
            Err(BackendError::Storage {
                operation: "max_pool1d",
                ..
            })
        ));
    }

    #[test]
    fn rejects_overlapping_parallel_output_layout() {
        let input = Layout::new([1, 1, 2].into());
        let output = Layout::from_shape_strides([1, 1, 2].into(), smallvec::smallvec![2, 2, 0], 0);

        assert_eq!(
            validate_forward::<1>("max_pool1d", &input, 2, &output, 2, PARAMETERS),
            Err(BackendError::AliasedLayout {
                operation: "max_pool1d"
            })
        );
    }

    #[test]
    fn rejects_zero_stride_before_kernel_arithmetic() {
        let input = Layout::new([1, 1, 2].into());
        let output = Layout::new([1, 1, 2].into());
        let parameters = PoolParameters {
            stride: 0,
            ..PARAMETERS
        };

        assert!(matches!(
            validate_forward::<1>("avg_pool1d", &input, 2, &output, 2, parameters),
            Err(BackendError::Storage {
                operation: "avg_pool1d",
                ..
            })
        ));
    }

    #[test]
    fn max_pool_rejects_short_output_before_writing() {
        let backend = SequentialBackend::new();
        let input = CpuStorage::from_slice(&[2.0_f32, 3.0]).expect("input allocation succeeds");
        let mut output = CpuStorage::from_slice(&[7.0_f32]).expect("output allocation succeeds");
        let input_layout = Layout::new([1, 1, 2].into());
        let output_layout = Layout::new([1, 1, 2].into());

        let result = super::super::max_pool1d(
            &backend,
            &input,
            &input_layout,
            1,
            1,
            0,
            1,
            &mut output,
            &output_layout,
        );

        assert!(matches!(
            result,
            Err(BackendError::Storage {
                operation: "max_pool1d",
                ..
            })
        ));
        assert_eq!(output.as_slice(), &[7.0]);
    }

    #[test]
    fn max_pool_rejects_uninitialized_input_before_writing() {
        let backend = SequentialBackend::new();
        // SAFETY: the test verifies pooling rejects this storage before it
        // reads any uninitialized elements.
        let input = unsafe { backend.allocate::<f32>(2) }.expect("input allocation succeeds");
        let mut output =
            CpuStorage::from_slice(&[7.0_f32, 8.0]).expect("output allocation succeeds");
        let input_layout = Layout::new([1, 1, 2].into());
        let output_layout = Layout::new([1, 1, 2].into());

        let result = super::super::max_pool1d(
            &backend,
            &input,
            &input_layout,
            1,
            1,
            0,
            1,
            &mut output,
            &output_layout,
        );

        assert!(matches!(
            result,
            Err(BackendError::Storage {
                operation: "max_pool1d",
                ..
            })
        ));
        assert_eq!(output.as_slice(), &[7.0, 8.0]);
    }

    #[test]
    fn one_dimensional_pool_rejects_four_dimensional_layouts_before_access() {
        let backend = SequentialBackend::new();
        let layout = Layout::new([1, 1, 2, 2].into());
        let input =
            CpuStorage::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).expect("input allocation succeeds");
        let mut output =
            CpuStorage::from_slice(&[7.0_f32, 8.0, 9.0, 10.0]).expect("output allocation succeeds");

        let result =
            super::super::max_pool1d(&backend, &input, &layout, 1, 1, 0, 1, &mut output, &layout);

        assert_eq!(
            result,
            Err(BackendError::LayoutRankMismatch {
                operation: "max_pool1d",
                lhs: 4,
                rhs: 3,
            })
        );
        assert_eq!(output.as_slice(), &[7.0, 8.0, 9.0, 10.0]);

        let mut grad_input = CpuStorage::from_slice(&[11.0_f32, 12.0, 13.0, 14.0])
            .expect("gradient allocation succeeds");
        let grad_output = CpuStorage::from_slice(&[1.0_f32, 1.0, 1.0, 1.0])
            .expect("gradient allocation succeeds");
        let result = super::super::max_pool1d_backward(
            &backend,
            &grad_output,
            &layout,
            &input,
            &layout,
            1,
            1,
            0,
            1,
            &mut grad_input,
            &layout,
        );

        assert_eq!(
            result,
            Err(BackendError::LayoutRankMismatch {
                operation: "max_pool1d_backward",
                lhs: 4,
                rhs: 3,
            })
        );
        assert_eq!(grad_input.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
    }
}
