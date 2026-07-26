use coeus_core::Layout;

pub(crate) const CUDA_BLOCK_SIZE: u32 = 256;

pub(crate) fn cuda_u32(value: usize) -> Option<u32> {
    u32::try_from(value).ok()
}

pub(crate) fn checked_numel(layout: &Layout) -> Option<usize> {
    layout
        .shape()
        .iter()
        .copied()
        .try_fold(1usize, |numel, dimension| numel.checked_mul(dimension))
}

pub(crate) fn checked_layout_storage_len(layout: &Layout) -> Option<usize> {
    if layout.shape().contains(&0) {
        return Some(0);
    }
    let max_offset = layout.shape().iter().zip(layout.strides()).try_fold(
        layout.offset(),
        |max_offset, (&dimension, &stride)| {
            max_offset.checked_add((dimension - 1).checked_mul(stride)?)
        },
    )?;
    max_offset.checked_add(1)
}

pub(crate) fn layouts_fit_cuda(layouts: &[&Layout]) -> bool {
    layouts.iter().all(|layout| {
        layout.ndim() <= 8
            && cuda_u32(layout.offset()).is_some()
            && layout
                .shape()
                .iter()
                .chain(layout.strides().iter())
                .all(|&value| cuda_u32(value).is_some())
    })
}

pub(crate) fn layouts_share_shape(layouts: &[&Layout]) -> bool {
    let Some((first, rest)) = layouts.split_first() else {
        return true;
    };
    rest.iter().all(|layout| layout.shape() == first.shape())
}

pub(crate) fn layout_supports_cuda_output_indexing(layout: &Layout) -> bool {
    layouts_fit_cuda(&[layout]) && layout.strides().iter().all(|&stride| stride != 0)
}

pub(crate) fn launch_grid_size(total: usize) -> Option<u32> {
    launch_grid_size_for_block(total, usize::try_from(CUDA_BLOCK_SIZE).ok()?)
}

pub(crate) fn launch_grid_size_for_block(total: usize, block_size: usize) -> Option<u32> {
    if total == 0 || block_size == 0 {
        return None;
    }
    cuda_u32(total.div_ceil(block_size))
}

#[cfg(test)]
mod tests {
    use super::{
        checked_layout_storage_len, checked_numel, launch_grid_size, launch_grid_size_for_block,
        layout_supports_cuda_output_indexing, layouts_share_shape,
    };
    use coeus_core::Layout;

    #[test]
    fn checked_numel_rejects_product_overflow() {
        let layout = Layout::new(vec![usize::MAX, 2].into());

        assert_eq!(checked_numel(&layout), None);
    }

    #[test]
    fn launch_grid_size_rejects_grid_overflow() {
        let total = usize::MAX;

        assert_eq!(launch_grid_size(total), None);
        assert_eq!(launch_grid_size(0), None);
    }

    #[test]
    fn launch_grid_size_supports_nonstandard_block_widths() {
        assert_eq!(launch_grid_size_for_block(17, 16), Some(2));
        assert_eq!(launch_grid_size_for_block(0, 16), None);
        assert_eq!(launch_grid_size_for_block(17, 0), None);
    }

    #[test]
    fn output_indexing_rejects_zero_stride_aliasing() {
        let layout = Layout::from_shape_strides(vec![2].into(), vec![0].into(), 0);

        assert!(!layout_supports_cuda_output_indexing(&layout));
    }

    #[test]
    fn shape_validation_rejects_mismatched_layouts() {
        let first = Layout::new(vec![2, 3].into());
        let second = Layout::new(vec![3, 2].into());

        assert!(!layouts_share_shape(&[&first, &second]));
    }

    #[test]
    fn layout_storage_len_accounts_for_offset_and_strides() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![4, 1].into(), 2);

        assert_eq!(checked_layout_storage_len(&layout), Some(9));
    }

    #[test]
    fn layout_storage_len_rejects_offset_overflow() {
        let layout = Layout::from_shape_strides(vec![2].into(), vec![1].into(), usize::MAX);

        assert_eq!(checked_layout_storage_len(&layout), None);
    }
}
