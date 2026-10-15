mod avg;
mod max;
mod pool1d;
#[cfg(test)]
mod tests;

pub(crate) use avg::{avg_pool2d, avg_pool2d_backward, avg_pool3d, avg_pool3d_backward};
pub(crate) use max::{max_pool2d, max_pool2d_backward, max_pool3d, max_pool3d_backward};
pub(crate) use pool1d::{avg_pool1d, avg_pool1d_backward, max_pool1d, max_pool1d_backward};

/// Reject a dispatch layout whose rank does not match the kernel entry point.
///
/// The leto adapter dispatches on rank, so a rank-4 layout handed to the 1-D
/// entry point would silently run 2-D pooling; the historical kernels rejected
/// this before touching storage, and so do the delegations.
fn check_rank(
    operation: &'static str,
    layout: &coeus_core::Layout,
    expected: usize,
) -> Result<(), coeus_core::BackendError> {
    let actual = layout.ndim();
    if actual == expected {
        Ok(())
    } else {
        Err(coeus_core::BackendError::Storage {
            operation,
            reason: format!(
                "{operation} expects rank {expected} ([N, C, spatial...]), got rank {actual}"
            ),
        })
    }
}
