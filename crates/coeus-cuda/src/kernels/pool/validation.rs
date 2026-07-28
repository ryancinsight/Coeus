use crate::kernels::validation::{
    CUDA_BLOCK_SIZE, checked_numel, cuda_u32, launch_grid_size, layouts_fit_cuda,
};
use coeus_core::Layout;

pub(crate) fn pool_layouts_are_valid(layouts: &[&Layout], rank: usize) -> bool {
    layouts_fit_cuda(layouts)
        && layouts.iter().all(|layout| {
            layout.ndim() == rank && layout.shape().iter().all(|&dimension| dimension != 0)
        })
}

pub(crate) fn pool_prefix_matches(lhs: &Layout, rhs: &Layout) -> bool {
    lhs.shape().get(..2) == rhs.shape().get(..2)
}

pub(crate) fn pool_shapes_match(lhs: &Layout, rhs: &Layout) -> bool {
    lhs.shape() == rhs.shape()
}

pub(crate) fn checked_pool_parameters(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<[u32; 4]> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return None;
    }
    Some([
        cuda_u32(kernel_size)?,
        cuda_u32(stride)?,
        cuda_u32(padding)?,
        cuda_u32(dilation)?,
    ])
}

pub(crate) fn pool_index_arithmetic_is_valid(
    input_layout: &Layout,
    output_layout: &Layout,
    parameters: [u32; 4],
    spatial_rank: usize,
) -> bool {
    let [kernel_size, stride, padding, dilation] = parameters;
    let signed_max = i32::MAX.unsigned_abs();
    if [kernel_size, stride, padding, dilation]
        .into_iter()
        .any(|value| value > signed_max)
    {
        return false;
    }
    let Some(input_spatial) = input_layout
        .shape()
        .get(input_layout.ndim().saturating_sub(spatial_rank)..)
    else {
        return false;
    };
    let Some(output_spatial) = output_layout
        .shape()
        .get(output_layout.ndim().saturating_sub(spatial_rank)..)
    else {
        return false;
    };
    if input_spatial.len() != spatial_rank || output_spatial.len() != spatial_rank {
        return false;
    }
    let Some(window_extent) = kernel_size
        .checked_sub(1)
        .and_then(|windows| windows.checked_mul(dilation))
    else {
        return false;
    };
    input_spatial
        .iter()
        .zip(output_spatial)
        .all(|(&input, &output)| {
            let (Some(input), Some(output)) = (cuda_u32(input), cuda_u32(output)) else {
                return false;
            };
            input != 0
                && output != 0
                && input <= signed_max
                && output <= signed_max
                && output
                    .checked_sub(1)
                    .and_then(|last| last.checked_mul(stride))
                    .and_then(|base| base.checked_add(window_extent))
                    .is_some_and(|coordinate| coordinate <= signed_max)
                && input
                    .checked_sub(1)
                    .and_then(|last| last.checked_add(padding))
                    .is_some_and(|coordinate| coordinate <= signed_max)
        })
}

pub(crate) fn checked_pool_work(layout: &Layout) -> Option<(u32, u32)> {
    let numel = checked_numel(layout)?;
    Some((cuda_u32(numel)?, launch_grid_size(numel)?))
}

pub(crate) const POOL_BLOCK_SIZE: u32 = CUDA_BLOCK_SIZE;

#[cfg(test)]
mod tests {
    use super::pool_index_arithmetic_is_valid;
    use crate::kernels::validation::layout_fits_cuda_storage;
    use coeus_core::Layout;

    #[test]
    fn pool_storage_validation_rejects_undersized_and_overflowed_layouts() {
        let max_unsigned = usize::try_from(u32::MAX)
            .expect("invariant: CUDA host usize represents every u32 value");
        let strided =
            Layout::from_shape_strides(vec![1, 1, 2, 2].into(), vec![16, 16, 4, 2].into(), 3);
        let overflowed = Layout::from_shape_strides(
            vec![1, 1, 2, 2].into(),
            vec![1, 1, max_unsigned, 1].into(),
            0,
        );

        assert!(layout_fits_cuda_storage(&strided, 10, false));
        assert!(!layout_fits_cuda_storage(&strided, 9, false));
        assert!(!layout_fits_cuda_storage(&overflowed, usize::MAX, false));
    }

    #[test]
    fn pool_signed_index_validation_covers_forward_and_backward_extrema() {
        let signed_max = usize::try_from(i32::MAX.unsigned_abs())
            .expect("invariant: CUDA host usize represents every positive i32 value");
        let input = Layout::new(vec![1, 1, 8, 8].into());
        let output = Layout::new(vec![1, 1, 4, 4].into());
        let excessive_output = Layout::new(vec![1, 1, signed_max, 1].into());

        assert!(pool_index_arithmetic_is_valid(
            &input,
            &output,
            [3, 2, 1, 1],
            2
        ));
        assert!(!pool_index_arithmetic_is_valid(
            &input,
            &excessive_output,
            [3, 2, 1, 1],
            2
        ));
        assert!(!pool_index_arithmetic_is_valid(
            &input,
            &output,
            [3, 2, i32::MAX.unsigned_abs(), 1],
            2
        ));
    }
}
