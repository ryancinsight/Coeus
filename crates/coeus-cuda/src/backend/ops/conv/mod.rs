mod backward;
mod forward;
mod transpose;

use coeus_core::Layout;

fn checked_numel(layout: &Layout) -> Option<usize> {
    layout
        .shape()
        .iter()
        .copied()
        .try_fold(1usize, |numel, dimension| numel.checked_mul(dimension))
}

#[inline]
fn supports_native_conv_transpose_layouts<const SPATIAL: usize>(
    input_layout: &Layout,
    weight_layout: &Layout,
    output_layout: &Layout,
) -> bool {
    let rank = SPATIAL + 2;
    if [input_layout, weight_layout, output_layout]
        .into_iter()
        .any(|layout| layout.ndim() != rank || !layout.is_contiguous() || layout.offset() != 0)
    {
        return false;
    }
    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();
    let output_shape = output_layout.shape();
    input_shape[0] == output_shape[0]
        && input_shape[1] == weight_shape[0]
        && output_shape[1] == weight_shape[1]
}
