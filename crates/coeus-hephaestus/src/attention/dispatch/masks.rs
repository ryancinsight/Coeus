use core::num::NonZeroUsize;

use coeus_core::BackendError;
use hephaestus_core::{AttentionMask, GroupedKeepMask, StridedView};
use leto::Layout;

pub(in crate::attention) fn bind<'a, B>(
    operation: &'static str,
    buffer: Option<&'a B>,
    layout: Option<&'a Layout<2>>,
    execution_batches: usize,
    is_causal: bool,
) -> Result<AttentionMask<'a, B>, BackendError> {
    let keep = match (buffer, layout) {
        (None, None) => None,
        (Some(buffer), Some(layout)) => {
            let mask_batches = layout.shape[0];
            let Some(heads_per_batch) = execution_batches
                .checked_div(mask_batches)
                .filter(|width| *width > 0 && *width * mask_batches == execution_batches)
                .and_then(NonZeroUsize::new)
            else {
                return Err(BackendError::IncompatibleBroadcast {
                    operation,
                    from: layout.shape.to_vec(),
                    to: vec![execution_batches, layout.shape[1]],
                });
            };
            Some(GroupedKeepMask::new(
                StridedView::new(buffer, layout),
                heads_per_batch,
            ))
        }
        _ => {
            return Err(BackendError::Storage {
                operation,
                reason: "key-padding mask storage and layout must be supplied together".into(),
            });
        }
    };

    Ok(match (is_causal, keep) {
        (false, None) => AttentionMask::unrestricted(),
        (true, None) => AttentionMask::causal(),
        (false, Some(keep)) => AttentionMask::keep(keep),
        (true, Some(keep)) => AttentionMask::causal_keep(keep),
    })
}
