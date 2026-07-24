use crate::backend::CudaScalar;
use crate::kernels::validation::{
    checked_layout_storage_len, checked_numel, cuda_u32, layout_supports_cuda_output_indexing,
    layouts_fit_cuda,
};
use crate::storage::CudaStorage;
use coeus_core::{Layout, Storage};

#[derive(Clone, Copy)]
pub(super) struct Launch1d {
    pub(super) kernel_size: u32,
    pub(super) stride: u32,
    pub(super) padding: u32,
    pub(super) dilation: u32,
    pub(super) total: u32,
    pub(super) total_elements: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Launch2d {
    pub(super) values: [u32; 9],
    pub(super) total: u32,
    pub(super) total_elements: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Parameters1d {
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    pub(super) dilation: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Parameters2d {
    pub(super) input_rank: usize,
    pub(super) output_rank: usize,
    pub(super) values: [usize; 9],
    pub(super) width: usize,
}

fn layout_storage_is_valid<T: coeus_core::Scalar>(
    storage: &CudaStorage<T>,
    layout: &Layout,
    output: bool,
) -> bool {
    layouts_fit_cuda(&[layout])
        && (!output || layout_supports_cuda_output_indexing(layout))
        && checked_layout_storage_len(layout).is_some_and(|required| {
            required.checked_sub(1).and_then(cuda_u32).is_some() && storage.len() >= required
        })
}

pub(super) fn checked_output_dim(
    input: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Option<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return None;
    }
    let padded = input.checked_add(padding.checked_mul(2)?)?;
    let effective_kernel = dilation.checked_mul(kernel.checked_sub(1)?)?;
    padded
        .checked_sub(effective_kernel)?
        .checked_sub(1)?
        .checked_div(stride)?
        .checked_add(1)
}

pub(super) fn checked_1d_launch<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    output: &CudaStorage<T>,
    output_layout: &Layout,
    parameters: Parameters1d,
) -> Option<Launch1d> {
    if input_layout.ndim() != 3
        || output_layout.ndim() != 3
        || !layout_storage_is_valid(input, input_layout, false)
        || !layout_storage_is_valid(output, output_layout, true)
    {
        return None;
    }
    let [n, channels, input_length] = input_layout.shape() else {
        return None;
    };
    let output_length = output_layout.shape()[2];
    if [*n, *channels, *input_length, output_length]
        .into_iter()
        .any(|dimension| dimension == 0)
    {
        return None;
    }
    let [kernel_size, stride, padding, dilation] = [
        parameters.kernel_size,
        parameters.stride,
        parameters.padding,
        parameters.dilation,
    ]
    .map(cuda_u32);
    let (Some(kernel_size), Some(stride), Some(padding), Some(dilation)) =
        (kernel_size, stride, padding, dilation)
    else {
        return None;
    };
    let total_elements = checked_numel(output_layout)?;
    Some(Launch1d {
        kernel_size,
        stride,
        padding,
        dilation,
        total: cuda_u32(total_elements)?,
        total_elements,
    })
}

pub(super) fn checked_2d_launch<T: CudaScalar>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    output: &CudaStorage<T>,
    output_layout: &Layout,
    parameters: Parameters2d,
) -> Option<Launch2d> {
    if input_layout.ndim() != parameters.input_rank
        || output_layout.ndim() != parameters.output_rank
        || parameters.width == 0
        || !layout_storage_is_valid(input, input_layout, false)
        || !layout_storage_is_valid(output, output_layout, true)
    {
        return None;
    }
    let [kernel_h, kernel_w, stride_h, stride_w, _, _, dilation_h, dilation_w, _] =
        parameters.values;
    if [
        kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w,
    ]
    .into_iter()
    .any(|parameter| parameter == 0)
    {
        return None;
    }
    let [Some(kernel_h), Some(kernel_w), Some(stride_h), Some(stride_w), Some(padding_h), Some(padding_w), Some(dilation_h), Some(dilation_w), Some(width)] =
        parameters.values.map(cuda_u32)
    else {
        return None;
    };
    let total_elements = checked_numel(output_layout)?;
    Some(Launch2d {
        values: [
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            width,
        ],
        total: cuda_u32(total_elements)?,
        total_elements,
    })
}

#[cfg(test)]
mod tests {
    use super::checked_output_dim;

    #[test]
    fn checked_output_dim_matches_sliding_window_formula() {
        assert_eq!(checked_output_dim(7, 3, 1, 2, 1), Some(4));
        assert_eq!(checked_output_dim(8, 3, 0, 1, 2), Some(4));
    }

    #[test]
    fn checked_output_dim_rejects_invalid_or_unrepresentable_parameters() {
        assert_eq!(checked_output_dim(7, 0, 0, 1, 1), None);
        assert_eq!(checked_output_dim(2, 5, 0, 1, 1), None);
        assert_eq!(checked_output_dim(usize::MAX, 3, usize::MAX, 1, 1), None);
    }
}
