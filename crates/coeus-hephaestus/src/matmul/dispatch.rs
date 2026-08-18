use super::provider::MatmulBackend;
use crate::layout::ranked;
use coeus_core::{Layout, Scalar};
use hephaestus_core::{DenseProductOps, StridedView};

const MATMUL: &str = "matmul";

/// Dispatch `c = a · b` for rank-2 operands through the backend's Hephaestus
/// dense-product seam.
///
/// The three layouts are validated and left-padded to rank 2 once here, so
/// every device provider receives the same strided operands and no backend
/// repeats the conversion.
///
/// # Errors
///
/// Returns the backend's configuration error when a layout exceeds rank 2 or
/// violates the strided-layout invariant, and the backend's dispatch error when
/// the provider rejects execution.
pub fn matmul<B, T>(
    a: &B::DeviceBuffer<T>,
    a_layout: &Layout,
    b: &B::DeviceBuffer<T>,
    b_layout: &Layout,
    c: &B::DeviceBuffer<T>,
    c_layout: &Layout,
) -> Result<(), B::Error>
where
    B: MatmulBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    let lhs = ranked::<2>(MATMUL, a_layout)?;
    let rhs = ranked::<2>(MATMUL, b_layout)?;
    let out = ranked::<2>(MATMUL, c_layout)?;

    B::Operations::default()
        .matmul_into(
            B::matmul_device(),
            StridedView::new(B::matmul_buffer(a), &lhs),
            StridedView::new(B::matmul_buffer(b), &rhs),
            StridedView::new(B::matmul_buffer(c), &out),
        )
        .map_err(|source| B::matmul_dispatch_error(MATMUL, source))
}
