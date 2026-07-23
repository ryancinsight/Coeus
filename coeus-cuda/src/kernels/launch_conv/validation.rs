use coeus_core::Layout;

pub(super) const CUDA_BLOCK_SIZE: u32 = 256;

pub(super) fn cuda_u32(value: usize) -> Option<u32> {
    u32::try_from(value).ok()
}

pub(super) fn checked_numel(layout: &Layout) -> Option<usize> {
    layout
        .shape()
        .iter()
        .copied()
        .try_fold(1usize, |numel, dimension| numel.checked_mul(dimension))
}

pub(super) fn layouts_fit_cuda(layouts: &[&Layout]) -> bool {
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

pub(super) fn launch_grid_size(total: usize) -> Option<u32> {
    let block_size = usize::try_from(CUDA_BLOCK_SIZE).ok()?;
    cuda_u32(total.div_ceil(block_size))
}
