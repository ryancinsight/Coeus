use crate::RotateHalfOps;
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Rotate the two equal halves of the final axis as `[-x₂, x₁]`.
///
/// # Errors
///
/// Returns the selected backend's typed failure when the rank or final extent
/// is unsupported, allocation fails, or provider dispatch fails.
pub fn rotate_half<T: Scalar, B: RotateHalfOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let storage = backend.rotate_half_storage(input.storage(), input.layout())?;
    Ok(Tensor::from_raw_parts(
        storage,
        coeus_core::Layout::new(input.shape_cloned()),
    ))
}
