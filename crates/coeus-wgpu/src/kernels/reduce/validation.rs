use crate::backend::{
    checked_numel, checked_u32_parameter, checked_workgroup_count, LayoutError, WgpuBackendError,
    METADATA_BUFFER_SIZE,
};
use crate::kernels::layout::{GpuLayoutInfo, MAX_WGSL_RANK};
use coeus_core::{BackendError, Layout};

pub(super) const OPERATION: &str = "fused reduction";

pub(crate) struct ReductionDispatch {
    pub(super) expression_rank: u32,
    pub(super) axis: u32,
    pub(super) axis_length: u32,
    pub(super) workgroups: u32,
}

pub(super) struct BindingPlan {
    pub(super) output: u32,
    pub(super) layouts: u32,
    pub(super) axis: u32,
}

pub(crate) fn validate_reduction(
    expression_shape: &[usize],
    op: coeus_ops::ReductionOp,
    axis: usize,
    output_layout: &Layout,
    limits: &wgpu::Limits,
) -> Result<ReductionDispatch, WgpuBackendError> {
    if expression_shape.len() > MAX_WGSL_RANK {
        return Err(WgpuBackendError::Layout(LayoutError::UnsupportedRank {
            rank: expression_shape.len(),
            max: MAX_WGSL_RANK,
        }));
    }

    let &axis_length = expression_shape.get(axis).ok_or({
        WgpuBackendError::Validation(BackendError::AxisOutOfRange {
            operation: OPERATION,
            axis,
            rank: expression_shape.len(),
        })
    })?;
    coeus_ops::fuse::validate_fused_reduction_axis(op, axis_length)?;

    let mut expected_output_shape = expression_shape.to_vec();
    expected_output_shape[axis] = 1;
    if output_layout.shape() != expected_output_shape {
        return Err(WgpuBackendError::Validation(BackendError::ShapeMismatch {
            operation: OPERATION,
            lhs: expected_output_shape,
            rhs: output_layout.shape().to_vec(),
        }));
    }

    GpuLayoutInfo::try_from_layout(output_layout)
        .map_err(|error| WgpuBackendError::Layout(error.into()))?;
    let output_elements = checked_numel(OPERATION, output_layout.shape())?;

    let workgroups = checked_workgroup_count(OPERATION, output_elements)?;
    if workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(WgpuBackendError::ResourceLimitExceeded {
            operation: OPERATION,
            resource: "compute workgroups per dimension",
            requested: u64::from(workgroups),
            limit: u64::from(limits.max_compute_workgroups_per_dimension),
        });
    }

    Ok(ReductionDispatch {
        expression_rank: checked_u32_parameter(
            OPERATION,
            "expression rank",
            expression_shape.len(),
        )?,
        axis: checked_u32_parameter(OPERATION, "axis", axis)?,
        axis_length: checked_u32_parameter(OPERATION, "axis length", axis_length)?,
        workgroups,
    })
}

pub(crate) fn validate_output_allocation<T>(
    element_count: usize,
    limits: &wgpu::Limits,
) -> Result<u64, WgpuBackendError> {
    let bytes =
        element_count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(WgpuBackendError::Validation(BackendError::Overflow {
                operation: OPERATION,
                reason: "output buffer byte-count arithmetic overflow",
            }))?;
    let bytes = u64::try_from(bytes).map_err(|_| {
        WgpuBackendError::Validation(BackendError::Overflow {
            operation: OPERATION,
            reason: "output buffer byte count exceeds u64",
        })
    })?;
    validate_resource_bytes("output buffer bytes", bytes, limits.max_buffer_size)?;
    validate_resource_bytes(
        "output storage-binding bytes",
        bytes,
        limits.max_storage_buffer_binding_size,
    )?;
    Ok(bytes)
}

pub(super) fn validate_storage_bindings(
    input_bytes: impl IntoIterator<Item = u64>,
    output_bytes: u64,
    limits: &wgpu::Limits,
) -> Result<(), WgpuBackendError> {
    let binding_limit = limits.max_storage_buffer_binding_size;
    for bytes in input_bytes {
        validate_resource_bytes("input storage-binding bytes", bytes, binding_limit)?;
    }
    validate_resource_bytes("output storage-binding bytes", output_bytes, binding_limit)?;
    validate_resource_bytes(
        "metadata storage-binding bytes",
        METADATA_BUFFER_SIZE,
        binding_limit,
    )
}

fn validate_resource_bytes(
    resource: &'static str,
    requested: u64,
    limit: u64,
) -> Result<(), WgpuBackendError> {
    if requested > limit {
        return Err(WgpuBackendError::ResourceLimitExceeded {
            operation: OPERATION,
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

pub(super) fn validate_bindings(
    input_count: usize,
    limits: &wgpu::Limits,
) -> Result<BindingPlan, WgpuBackendError> {
    const FIXED_STORAGE_BINDINGS: usize = 3;

    let binding_count =
        input_count
            .checked_add(FIXED_STORAGE_BINDINGS)
            .ok_or(WgpuBackendError::Validation(BackendError::Overflow {
                operation: OPERATION,
                reason: "storage-binding count arithmetic overflow",
            }))?;
    let binding_count_u32 =
        checked_u32_parameter(OPERATION, "storage binding count", binding_count)?;
    let storage_limit = limits.max_storage_buffers_per_shader_stage;
    if binding_count_u32 > storage_limit {
        return Err(WgpuBackendError::ResourceLimitExceeded {
            operation: OPERATION,
            resource: "storage-buffer bindings",
            requested: u64::from(binding_count_u32),
            limit: u64::from(storage_limit),
        });
    }
    if binding_count_u32 > limits.max_bindings_per_bind_group {
        return Err(WgpuBackendError::ResourceLimitExceeded {
            operation: OPERATION,
            resource: "bind-group bindings",
            requested: u64::from(binding_count_u32),
            limit: u64::from(limits.max_bindings_per_bind_group),
        });
    }

    let layout_count = input_count
        .checked_add(1)
        .ok_or(WgpuBackendError::Validation(BackendError::Overflow {
            operation: OPERATION,
            reason: "layout-metadata count arithmetic overflow",
        }))?;
    let metadata_bytes = layout_count
        .checked_mul(std::mem::size_of::<GpuLayoutInfo>())
        .ok_or(WgpuBackendError::Validation(BackendError::Overflow {
            operation: OPERATION,
            reason: "layout-metadata byte-count arithmetic overflow",
        }))?;
    let metadata_bytes =
        u64::try_from(metadata_bytes).map_err(|_| WgpuBackendError::ResourceLimitExceeded {
            operation: OPERATION,
            resource: "layout-metadata bytes",
            requested: u64::MAX,
            limit: METADATA_BUFFER_SIZE,
        })?;
    if metadata_bytes > METADATA_BUFFER_SIZE {
        return Err(WgpuBackendError::ResourceLimitExceeded {
            operation: OPERATION,
            resource: "layout-metadata bytes",
            requested: metadata_bytes,
            limit: METADATA_BUFFER_SIZE,
        });
    }

    Ok(BindingPlan {
        output: checked_u32_parameter(OPERATION, "output binding", input_count)?,
        layouts: checked_u32_parameter(OPERATION, "layout binding", input_count + 1)?,
        axis: checked_u32_parameter(OPERATION, "axis binding", input_count + 2)?,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        validate_bindings, validate_output_allocation, validate_reduction,
        validate_storage_bindings,
    };
    use crate::backend::WgpuBackendError;
    use coeus_core::{BackendError, Layout};

    #[test]
    fn rejects_axis_outside_expression_rank() {
        let output = Layout::new(vec![2, 1].into());

        assert!(matches!(
            validate_reduction(
                &[2, 3],
                coeus_ops::ReductionOp::Sum,
                2,
                &output,
                &wgpu::Limits::default()
            ),
            Err(WgpuBackendError::Validation(BackendError::AxisOutOfRange {
                operation: "fused reduction",
                axis: 2,
                rank: 2,
            }))
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_axis_length_outside_wgsl_abi() {
        let axis_length = usize::try_from(u64::from(u32::MAX) + 1).expect("test value fits usize");
        let output = Layout::new(vec![1].into());

        assert!(matches!(
            validate_reduction(
                &[axis_length],
                coeus_ops::ReductionOp::Sum,
                0,
                &output,
                &wgpu::Limits::default()
            ),
            Err(WgpuBackendError::AbiValueOutOfRange {
                operation: "fused reduction",
                parameter: "axis length",
                value,
            }) if value == axis_length
        ));
    }

    #[test]
    fn rejects_storage_bindings_above_device_limit() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 4,
            ..wgpu::Limits::default()
        };

        assert!(matches!(
            validate_bindings(2, &limits),
            Err(WgpuBackendError::ResourceLimitExceeded {
                operation: "fused reduction",
                resource: "storage-buffer bindings",
                requested: 5,
                limit: 4,
            })
        ));
    }

    #[test]
    fn rejects_layout_metadata_above_pool_buffer_capacity() {
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: u32::MAX,
            max_bindings_per_bind_group: u32::MAX,
            ..wgpu::Limits::default()
        };

        assert!(matches!(
            validate_bindings(14, &limits),
            Err(WgpuBackendError::ResourceLimitExceeded {
                operation: "fused reduction",
                resource: "layout-metadata bytes",
                requested: 1_080,
                limit: 1_024,
            })
        ));
    }

    #[test]
    fn rejects_workgroups_above_device_limit() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..wgpu::Limits::default()
        };
        let output = Layout::new(vec![257, 1].into());

        assert!(matches!(
            validate_reduction(
                &[257, 257],
                coeus_ops::ReductionOp::Sum,
                1,
                &output,
                &limits
            ),
            Err(WgpuBackendError::ResourceLimitExceeded {
                operation: "fused reduction",
                resource: "compute workgroups per dimension",
                requested: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn rejects_output_allocation_above_device_limit() {
        let limits = wgpu::Limits {
            max_buffer_size: 16,
            max_storage_buffer_binding_size: 16,
            ..wgpu::Limits::default()
        };

        assert!(matches!(
            validate_output_allocation::<f32>(5, &limits),
            Err(WgpuBackendError::ResourceLimitExceeded {
                operation: "fused reduction",
                resource: "output buffer bytes",
                requested: 20,
                limit: 16,
            })
        ));
    }

    #[test]
    fn rejects_input_storage_binding_above_device_limit() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 16,
            ..wgpu::Limits::default()
        };

        assert!(matches!(
            validate_storage_bindings([20], 16, &limits),
            Err(WgpuBackendError::ResourceLimitExceeded {
                operation: "fused reduction",
                resource: "input storage-binding bytes",
                requested: 20,
                limit: 16,
            })
        ));
    }
}
