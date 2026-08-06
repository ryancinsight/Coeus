use crate::backend::{LayoutError, WgpuBackendError};
use coeus_core::Layout;
use hephaestus_wgpu::MAX_STRIDED_RANK;
use leto::Layout as LetoLayout;

#[derive(Clone, Copy)]
pub(super) struct ProviderActivation {
    pub(super) name: &'static str,
    pub(super) parameterized: bool,
}

pub(super) fn provider_owned_activation(op: coeus_ops::UnaryOp) -> Option<ProviderActivation> {
    let (name, parameterized) = match op {
        coeus_ops::UnaryOp::Log1p => ("Log1p", false),
        coeus_ops::UnaryOp::Mish => ("Mish", false),
        coeus_ops::UnaryOp::MishGrad => ("MishGrad", false),
        coeus_ops::UnaryOp::Elu => ("Elu", false),
        coeus_ops::UnaryOp::EluGrad => ("EluGrad", false),
        coeus_ops::UnaryOp::Hardtanh(_) => ("Hardtanh", true),
        coeus_ops::UnaryOp::HardtanhGrad(_) => ("HardtanhGrad", true),
        coeus_ops::UnaryOp::Threshold(_) => ("Threshold", true),
        coeus_ops::UnaryOp::ThresholdGrad(_) => ("ThresholdGrad", true),
        coeus_ops::UnaryOp::Relu => ("Relu", false),
        coeus_ops::UnaryOp::ReluGrad => ("ReluGrad", false),
        coeus_ops::UnaryOp::Sigmoid => ("Sigmoid", false),
        _ => return None,
    };
    Some(ProviderActivation {
        name,
        parameterized,
    })
}

pub(super) fn can_route_strided_wgpu(layouts: &[&Layout], out: &Layout) -> bool {
    layouts
        .iter()
        .chain(std::iter::once(&out))
        .all(|layout| layout.ndim() <= MAX_STRIDED_RANK)
        && !out
            .shape()
            .iter()
            .zip(out.strides())
            .any(|(&dimension, &stride)| dimension > 1 && stride == 0)
}

/// Convert a dynamic Coeus layout to a const-rank Leto layout.
///
/// Shorter layouts are left-padded with size-one dimensions and zero strides
/// for provider-side broadcasting. Strides are checked before crossing the
/// signed Leto layout boundary; a narrowing cast would corrupt large strides
/// on 32-bit targets.
pub(super) fn coeus_to_leto_layout<const N: usize>(
    layout: &Layout,
) -> Result<LetoLayout<N>, WgpuBackendError> {
    let rank = layout.ndim();
    let stride_rank = layout.strides().len();
    if rank != stride_rank {
        return Err(WgpuBackendError::Layout(LayoutError::RankMismatch {
            shape_rank: rank,
            stride_rank,
        }));
    }
    if rank > N {
        return Err(WgpuBackendError::Layout(LayoutError::UnsupportedRank {
            rank,
            max: N,
        }));
    }

    let pad = N - rank;
    let mut shape = [1usize; N];
    let mut strides = [0isize; N];
    for (index, (&dimension, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
        shape[pad + index] = dimension;
        strides[pad + index] = isize::try_from(stride).map_err(|_| {
            WgpuBackendError::Layout(LayoutError::SignedStrideOutOfRange {
                axis: index,
                value: stride,
            })
        })?;
    }
    Ok(LetoLayout::new(shape, strides, layout.offset()))
}

#[cfg(test)]
mod tests {
    use super::coeus_to_leto_layout;
    use coeus_core::Layout;

    #[test]
    fn pads_layouts_for_provider_rank() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![3, 1].into(), 4);
        let converted = coeus_to_leto_layout::<3>(&layout).expect("representable layout");

        assert_eq!(converted.shape, [1, 2, 3]);
        assert_eq!(converted.strides, [0, 3, 1]);
        assert_eq!(converted.offset, 4);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn rejects_stride_values_outside_signed_provider_range() {
        let stride = usize::try_from(isize::MAX).expect("isize max fits usize") + 1;
        let layout = Layout::from_shape_strides(vec![1].into(), vec![stride].into(), 0);

        assert!(matches!(
            coeus_to_leto_layout::<1>(&layout),
            Err(crate::backend::WgpuBackendError::Layout(
                crate::backend::LayoutError::SignedStrideOutOfRange { axis: 0, value }
            )) if value == stride
        ));
    }
}
