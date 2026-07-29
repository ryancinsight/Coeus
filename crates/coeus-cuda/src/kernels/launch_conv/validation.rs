use crate::kernels::validation::{checked_numel, layout_fits_cuda_storage};
use coeus_core::Layout;

pub(super) struct OutputLayout<'layout> {
    pub(super) layout: &'layout Layout,
    pub(super) storage_len: usize,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the raw convolution boundary carries three buffer-layout pairs"
)]
pub(super) fn forward_layouts_fit_storage<const RANK: usize>(
    input_layout: &Layout,
    input_len: usize,
    weight_layout: &Layout,
    weight_len: usize,
    bias_len: Option<usize>,
    output_layout: &Layout,
    output_len: usize,
    output_elements: usize,
) -> bool {
    input_layout.ndim() == RANK
        && weight_layout.ndim() == RANK
        && output_layout.ndim() == RANK
        && checked_numel(output_layout) == Some(output_elements)
        && layout_fits_cuda_storage(input_layout, input_len, false)
        && layout_fits_cuda_storage(weight_layout, weight_len, false)
        && layout_fits_cuda_storage(output_layout, output_len, true)
        && bias_len.is_none_or(|len| {
            weight_layout
                .shape()
                .first()
                .is_some_and(|&channels| len >= channels)
        })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the backward boundary carries source and optional gradient buffer contracts"
)]
pub(super) fn backward_layouts_fit_storage<const RANK: usize>(
    grad_output_layout: &Layout,
    grad_output_len: usize,
    input_layout: &Layout,
    input_len: usize,
    weight_layout: &Layout,
    weight_len: usize,
    grad_input: Option<OutputLayout<'_>>,
    grad_weight: Option<OutputLayout<'_>>,
    grad_bias_len: Option<usize>,
) -> bool {
    grad_output_layout.ndim() == RANK
        && input_layout.ndim() == RANK
        && weight_layout.ndim() == RANK
        && layout_fits_cuda_storage(grad_output_layout, grad_output_len, false)
        && layout_fits_cuda_storage(input_layout, input_len, false)
        && layout_fits_cuda_storage(weight_layout, weight_len, false)
        && grad_input.is_none_or(|output| {
            output.layout.ndim() == RANK
                && layout_fits_cuda_storage(output.layout, output.storage_len, true)
        })
        && grad_weight.is_none_or(|output| {
            output.layout.ndim() == RANK
                && layout_fits_cuda_storage(output.layout, output.storage_len, true)
        })
        && grad_bias_len.is_none_or(|len| {
            weight_layout
                .shape()
                .first()
                .is_some_and(|&channels| len >= channels)
        })
}

#[cfg(test)]
mod tests {
    use super::{OutputLayout, backward_layouts_fit_storage, forward_layouts_fit_storage};
    use coeus_core::Layout;

    #[test]
    fn forward_rejects_each_oversized_layout_and_count_mismatch() {
        let input = Layout::new([1, 1, 4].into());
        let weight = Layout::new([2, 1, 3].into());
        let output = Layout::new([1, 2, 2].into());

        for valid in [
            forward_layouts_fit_storage::<3>(&input, 3, &weight, 6, Some(2), &output, 4, 4),
            forward_layouts_fit_storage::<3>(&input, 4, &weight, 5, Some(2), &output, 4, 4),
            forward_layouts_fit_storage::<3>(&input, 4, &weight, 6, Some(2), &output, 3, 4),
            forward_layouts_fit_storage::<3>(&input, 4, &weight, 6, Some(2), &output, 4, 5),
        ] {
            assert!(!valid);
        }
    }

    #[test]
    fn forward_rejects_rank_bias_and_writable_alias_contracts() {
        let input = Layout::new([1, 1, 4].into());
        let weight = Layout::new([2, 1, 3].into());
        let output = Layout::new([1, 2, 2].into());
        let aliased_output = Layout::from_shape_strides([1, 2, 2].into(), vec![0, 2, 1].into(), 0);

        assert!(!forward_layouts_fit_storage::<4>(
            &input,
            4,
            &weight,
            6,
            Some(2),
            &output,
            4,
            4,
        ));
        assert!(!forward_layouts_fit_storage::<3>(
            &input,
            4,
            &weight,
            6,
            Some(1),
            &output,
            4,
            4,
        ));
        assert!(!forward_layouts_fit_storage::<3>(
            &input,
            4,
            &weight,
            6,
            Some(2),
            &aliased_output,
            4,
            4,
        ));
    }

    #[test]
    fn backward_validates_requested_outputs() {
        let input = Layout::new([1, 1, 4].into());
        let weight = Layout::new([2, 1, 3].into());
        let grad_output = Layout::new([1, 2, 2].into());
        let grad_input = Layout::new([1, 1, 4].into());
        let oversized = Layout::new([1, 1, 5].into());

        assert!(backward_layouts_fit_storage::<3>(
            &grad_output,
            4,
            &input,
            4,
            &weight,
            6,
            Some(OutputLayout {
                layout: &grad_input,
                storage_len: 4,
            }),
            None,
            Some(2),
        ));
        assert!(!backward_layouts_fit_storage::<3>(
            &grad_output,
            4,
            &input,
            4,
            &weight,
            6,
            Some(OutputLayout {
                layout: &oversized,
                storage_len: 4,
            }),
            None,
            Some(2),
        ));
    }
}
