use crate::CudaBackendError;
use crate::kernels::layout_fits_cuda_storage;
use coeus_core::Layout;

fn require_storage_fit(
    operation: &'static str,
    role: &'static str,
    layout: &Layout,
    storage_len: usize,
    writable: bool,
) -> Result<(), CudaBackendError> {
    if layout_fits_cuda_storage(layout, storage_len, writable) {
        Ok(())
    } else {
        Err(CudaBackendError::InvalidLayout {
            operation,
            reason: role,
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation receives three buffer-layout contracts and two alias relations"
)]
pub(super) fn validate_binary_layouts(
    a_layout: &Layout,
    a_len: usize,
    b_layout: &Layout,
    b_len: usize,
    c_layout: &Layout,
    c_len: usize,
    left_aliases_output: bool,
    right_aliases_output: bool,
) -> Result<(), CudaBackendError> {
    const OPERATION: &str = "elementwise binary";

    require_storage_fit(
        OPERATION,
        "left input layout exceeds its CUDA storage",
        a_layout,
        a_len,
        false,
    )?;
    require_storage_fit(
        OPERATION,
        "right input layout exceeds its CUDA storage",
        b_layout,
        b_len,
        false,
    )?;
    require_storage_fit(
        OPERATION,
        "output layout exceeds its CUDA storage or aliases output elements",
        c_layout,
        c_len,
        true,
    )?;

    if (left_aliases_output && a_layout != c_layout)
        || (right_aliases_output && b_layout != c_layout)
    {
        return Err(CudaBackendError::InvalidLayout {
            operation: OPERATION,
            reason: "aliased input and output must use the same layout",
        });
    }

    Ok(())
}

pub(super) fn validate_unary_layouts(
    a_layout: &Layout,
    a_len: usize,
    c_layout: &Layout,
    c_len: usize,
    input_aliases_output: bool,
) -> Result<(), CudaBackendError> {
    const OPERATION: &str = "elementwise unary";

    require_storage_fit(
        OPERATION,
        "input layout exceeds its CUDA storage",
        a_layout,
        a_len,
        false,
    )?;
    require_storage_fit(
        OPERATION,
        "output layout exceeds its CUDA storage or aliases output elements",
        c_layout,
        c_len,
        true,
    )?;

    if input_aliases_output && a_layout != c_layout {
        return Err(CudaBackendError::InvalidLayout {
            operation: OPERATION,
            reason: "aliased input and output must use the same layout",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_binary_layouts, validate_unary_layouts};
    use crate::CudaBackendError;
    use coeus_core::Layout;

    #[test]
    fn binary_rejects_each_layout_that_exceeds_storage() {
        let valid = Layout::new([4].into());
        let oversized = Layout::new([5].into());

        for result in [
            validate_binary_layouts(&oversized, 4, &valid, 4, &valid, 4, false, false),
            validate_binary_layouts(&valid, 4, &oversized, 4, &valid, 4, false, false),
            validate_binary_layouts(&valid, 4, &valid, 4, &oversized, 4, false, false),
        ] {
            assert!(matches!(
                result,
                Err(CudaBackendError::InvalidLayout {
                    operation: "elementwise binary",
                    ..
                })
            ));
        }
    }

    #[test]
    fn unary_accepts_bounded_nonzero_offsets_for_strided_dispatch() {
        let view = Layout::from_shape_strides([2].into(), vec![1].into(), 2);
        let output = Layout::new([2].into());

        assert!(matches!(
            validate_unary_layouts(&view, 4, &output, 2, false),
            Ok(())
        ));
    }

    #[test]
    fn output_rejects_zero_stride_aliasing() {
        let input = Layout::new([2, 3].into());
        let output = Layout::from_shape_strides([2, 3].into(), vec![0, 1].into(), 0);

        assert!(matches!(
            validate_unary_layouts(&input, 6, &output, 3, false),
            Err(CudaBackendError::InvalidLayout {
                operation: "elementwise unary",
                reason: "output layout exceeds its CUDA storage or aliases output elements",
            })
        ));
    }

    #[test]
    fn aliased_storage_rejects_layout_remapping() {
        let source = Layout::new([2, 2].into());
        let transposed = Layout::from_shape_strides([2, 2].into(), vec![1, 2].into(), 0);

        assert!(matches!(
            validate_binary_layouts(&source, 4, &source, 4, &transposed, 4, true, false,),
            Err(CudaBackendError::InvalidLayout {
                operation: "elementwise binary",
                reason: "aliased input and output must use the same layout",
            })
        ));
        assert!(matches!(
            validate_unary_layouts(&source, 4, &transposed, 4, true),
            Err(CudaBackendError::InvalidLayout {
                operation: "elementwise unary",
                reason: "aliased input and output must use the same layout",
            })
        ));
    }

    #[test]
    fn exact_in_place_layouts_remain_valid() {
        let layout = Layout::new([4].into());

        assert!(matches!(
            validate_binary_layouts(&layout, 4, &layout, 4, &layout, 4, true, false),
            Ok(())
        ));
        assert!(matches!(
            validate_unary_layouts(&layout, 4, &layout, 4, true),
            Ok(())
        ));
    }

    #[test]
    fn empty_layouts_are_valid_without_storage() {
        let empty = Layout::new(vec![2, 0, 3].into());

        assert!(matches!(
            validate_binary_layouts(&empty, 0, &empty, 0, &empty, 0, false, false),
            Ok(())
        ));
        assert!(matches!(
            validate_unary_layouts(&empty, 0, &empty, 0, false),
            Ok(())
        ));
    }
}
