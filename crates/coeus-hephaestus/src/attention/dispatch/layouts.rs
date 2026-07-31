use crate::layout::ranked;
use coeus_core::{BackendError, Layout};
use leto::Layout as LetoLayout;

pub(super) fn tensor(
    operation: &'static str,
    layout: &Layout,
) -> Result<LetoLayout<3>, BackendError> {
    ranked::<3>(operation, layout)
}

pub(super) fn keep_mask(
    operation: &'static str,
    layout: &Layout,
) -> Result<LetoLayout<2>, BackendError> {
    ranked::<2>(operation, layout)
}
