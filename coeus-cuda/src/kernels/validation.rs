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

pub(crate) fn launch_grid_size(total: usize) -> Option<u32> {
    let block_size = usize::try_from(CUDA_BLOCK_SIZE).ok()?;
    cuda_u32(total.div_ceil(block_size))
}

#[cfg(test)]
mod tests {
    use super::{checked_numel, launch_grid_size};
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
    }
}
