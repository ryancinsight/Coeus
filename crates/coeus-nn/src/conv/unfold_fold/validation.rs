use crate::module::ModuleError;

pub(super) fn checked_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return None;
    }

    let effective = kernel
        .checked_sub(1)
        .and_then(|extent| dilation.checked_mul(extent))
        .and_then(|extent| extent.checked_add(1))?;
    let padded = padding
        .checked_mul(2)
        .and_then(|padding| input.checked_add(padding))?;
    padded
        .checked_sub(effective)?
        .checked_div(stride)?
        .checked_add(1)
}

pub(super) fn invalid_window<E>(
    module: &'static str,
    parameter: &'static str,
    actual: Vec<usize>,
) -> ModuleError<E>
where
    E: std::error::Error + 'static,
{
    ModuleError::ShapeMismatch {
        module,
        parameter,
        expected: vec![1],
        actual,
    }
}
