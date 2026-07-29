use crate::kernels::validation::{checked_numel, layout_fits_cuda_storage};
use coeus_core::Layout;

pub(super) struct OutputLayout<'layout> {
    pub(super) layout: &'layout Layout,
    pub(super) storage_len: usize,
}

fn output_extent(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return None;
    }
    let effective_kernel = kernel
        .checked_sub(1)?
        .checked_mul(dilation)?
        .checked_add(1)?;
    let padded_input = input.checked_add(padding.checked_mul(2)?)?;
    padded_input
        .checked_sub(effective_kernel)?
        .checked_div(stride)?
        .checked_add(1)
}

fn shapes_define_convolution<const RANK: usize>(
    input: &Layout,
    weight: &Layout,
    output: &Layout,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> bool {
    if input.ndim() != RANK || weight.ndim() != RANK || output.ndim() != RANK || RANK < 3 {
        return false;
    }
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    let output_shape = output.shape();
    input_shape[0] == output_shape[0]
        && input_shape[1] == weight_shape[1]
        && weight_shape[0] == output_shape[1]
        && (2..RANK).all(|axis| {
            output_extent(
                input_shape[axis],
                weight_shape[axis],
                stride,
                padding,
                dilation,
            ) == Some(output_shape[axis])
        })
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
    stride: usize,
    padding: usize,
    dilation: usize,
) -> bool {
    shapes_define_convolution::<RANK>(
        input_layout,
        weight_layout,
        output_layout,
        stride,
        padding,
        dilation,
    ) && checked_numel(output_layout) == Some(output_elements)
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
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: Option<OutputLayout<'_>>,
    grad_weight: Option<OutputLayout<'_>>,
    grad_bias_len: Option<usize>,
) -> bool {
    shapes_define_convolution::<RANK>(
        input_layout,
        weight_layout,
        grad_output_layout,
        stride,
        padding,
        dilation,
    ) && layout_fits_cuda_storage(grad_output_layout, grad_output_len, false)
        && layout_fits_cuda_storage(input_layout, input_len, false)
        && layout_fits_cuda_storage(weight_layout, weight_len, false)
        && grad_input.is_none_or(|output| {
            output.layout == input_layout
                && layout_fits_cuda_storage(output.layout, output.storage_len, true)
        })
        && grad_weight.is_none_or(|output| {
            output.layout == weight_layout
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
            forward_layouts_fit_storage::<3>(
                &input,
                3,
                &weight,
                6,
                Some(2),
                &output,
                4,
                4,
                1,
                0,
                1,
            ),
            forward_layouts_fit_storage::<3>(
                &input,
                4,
                &weight,
                5,
                Some(2),
                &output,
                4,
                4,
                1,
                0,
                1,
            ),
            forward_layouts_fit_storage::<3>(
                &input,
                4,
                &weight,
                6,
                Some(2),
                &output,
                3,
                4,
                1,
                0,
                1,
            ),
            forward_layouts_fit_storage::<3>(
                &input,
                4,
                &weight,
                6,
                Some(2),
                &output,
                4,
                5,
                1,
                0,
                1,
            ),
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
            1,
            0,
            1,
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
            1,
            0,
            1,
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
            1,
            0,
            1,
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
            1,
            0,
            1,
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
            1,
            0,
            1,
            Some(OutputLayout {
                layout: &oversized,
                storage_len: 4,
            }),
            None,
            Some(2),
        ));
    }

    #[test]
    fn convolution_rejects_shape_and_parameter_mismatches() {
        let input = Layout::new([1, 2, 5].into());
        let weight = Layout::new([3, 2, 3].into());
        let output = Layout::new([1, 3, 3].into());
        let wrong_channels = Layout::new([1, 2, 3].into());
        let wrong_extent = Layout::new([1, 3, 4].into());

        assert!(forward_layouts_fit_storage::<3>(
            &input,
            10,
            &weight,
            18,
            Some(3),
            &output,
            9,
            9,
            1,
            0,
            1,
        ));
        assert!(!forward_layouts_fit_storage::<3>(
            &input,
            10,
            &weight,
            18,
            Some(3),
            &wrong_channels,
            6,
            6,
            1,
            0,
            1,
        ));
        assert!(!forward_layouts_fit_storage::<3>(
            &input,
            10,
            &weight,
            18,
            Some(3),
            &wrong_extent,
            12,
            12,
            1,
            0,
            1,
        ));
        assert!(!forward_layouts_fit_storage::<3>(
            &input,
            10,
            &weight,
            18,
            Some(3),
            &output,
            9,
            9,
            0,
            0,
            1,
        ));
    }
}
