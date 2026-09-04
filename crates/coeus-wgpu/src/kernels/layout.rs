use thiserror::Error;

pub(crate) const MAX_WGSL_RANK: usize = 8;

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(crate) enum GpuLayoutError {
    #[error("WGPU layout rank {rank} exceeds the supported maximum {max}")]
    UnsupportedRank { rank: usize, max: usize },
    #[error("WGPU layout shape rank {shape_rank} differs from stride rank {stride_rank}")]
    RankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
    #[error("WGPU layout offset {value} exceeds the WGSL u32 range")]
    OffsetOutOfRange { value: usize },
    #[error("WGPU layout shape axis {axis} value {value} exceeds the WGSL u32 range")]
    ShapeOutOfRange { axis: usize, value: usize },
    #[error("WGPU layout stride axis {axis} value {value} exceeds the WGSL u32 range")]
    StrideOutOfRange { axis: usize, value: usize },
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLayoutInfo {
    pub offset: u32,
    pub ndim: u32,
    pub shape: [u32; 8],
    pub strides: [u32; 8],
}

impl GpuLayoutInfo {
    pub(crate) fn try_from_layout(layout: &coeus_core::Layout) -> Result<Self, GpuLayoutError> {
        let mut shape = [0u32; 8];
        let mut strides = [0u32; 8];
        let ndim = layout.ndim();
        let stride_rank = layout.strides().len();
        if ndim != stride_rank {
            return Err(GpuLayoutError::RankMismatch {
                shape_rank: ndim,
                stride_rank,
            });
        }
        if ndim > MAX_WGSL_RANK {
            return Err(GpuLayoutError::UnsupportedRank {
                rank: ndim,
                max: MAX_WGSL_RANK,
            });
        }
        let ndim = u32::try_from(ndim).expect("invariant: supported rank fits u32");
        let offset =
            u32::try_from(layout.offset()).map_err(|_| GpuLayoutError::OffsetOutOfRange {
                value: layout.offset(),
            })?;
        for (i, (&dimension, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
            shape[i] = u32::try_from(dimension).map_err(|_| GpuLayoutError::ShapeOutOfRange {
                axis: i,
                value: dimension,
            })?;
            strides[i] = u32::try_from(stride).map_err(|_| GpuLayoutError::StrideOutOfRange {
                axis: i,
                value: stride,
            })?;
        }
        Ok(Self {
            offset,
            ndim,
            shape,
            strides,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::Layout;

    #[test]
    fn converts_representable_layout_without_narrowing_loss() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![3usize, 1].into(), 4);

        let gpu = GpuLayoutInfo::try_from_layout(&layout).expect("representable layout");

        assert_eq!(gpu.offset, 4);
        assert_eq!(gpu.ndim, 2);
        assert_eq!(gpu.shape[..2], [2, 3]);
        assert_eq!(gpu.strides[..2], [3, 1]);
    }

    #[test]
    fn rejects_layouts_above_the_fixed_rank_abi() {
        let layout = Layout::new(vec![1; MAX_WGSL_RANK + 1].into());

        assert!(matches!(
            GpuLayoutInfo::try_from_layout(&layout),
            Err(GpuLayoutError::UnsupportedRank { rank, max })
                if rank == MAX_WGSL_RANK + 1 && max == MAX_WGSL_RANK
        ));
    }

    #[test]
    fn rejects_shape_and_stride_rank_mismatch() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![1usize].into(), 0);

        assert!(matches!(
            GpuLayoutInfo::try_from_layout(&layout),
            Err(GpuLayoutError::RankMismatch {
                shape_rank: 2,
                stride_rank: 1,
            })
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_dimension_values_outside_the_u32_abi() {
        let dimension = usize::try_from(u32::MAX).expect("u32 fits usize") + 1;
        let layout = Layout::new(vec![dimension].into());

        assert!(matches!(
            GpuLayoutInfo::try_from_layout(&layout),
            Err(GpuLayoutError::ShapeOutOfRange { axis: 0, value })
                if value == dimension
        ));
    }
}
