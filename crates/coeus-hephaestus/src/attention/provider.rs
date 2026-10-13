use crate::{backend::HephaestusBackend, HephaestusProvider, HephaestusSeamBackend};
use coeus_core::{Float, Layout, Scalar};
use hephaestus_core::{AttentionOps, AttentionScalar};

/// Provider-owned scalar attention operation marker.
///
/// # Safety
///
/// Successful forward operations selected by `Operations` must initialize
/// every logical output and attention-weight element without reading prior
/// contents. Backward operations accumulate into selected gradient buffers,
/// which must be initialized before dispatch. A dispatch error may leave an
/// output partially written.
pub unsafe trait AttentionProvider<T>: HephaestusProvider
where
    T: Scalar + Float + AttentionScalar,
{
    /// Monomorphized Hephaestus operation marker selected by this provider.
    type Operations: AttentionOps<Self::Device, T> + Default;
}

/// Projects a Coeus backend's storage into one provider-owned attention path.
///
/// Device access and typed error mapping come from the shared
/// [`HephaestusSeamBackend`] seam; the default dispatch methods retain layout
/// validation, operand assembly, and provider invocation as a single
/// monomorphized implementation.
pub trait AttentionBackend<T>: HephaestusSeamBackend
where
    T: Scalar + Float + AttentionScalar,
    Self: HephaestusSeamBackend<Device = <Self::Provider as HephaestusProvider>::Device>,
{
    /// Hephaestus provider selected by this Coeus backend.
    type Provider: AttentionProvider<T>;

    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the method mirrors the AttentionOps provider boundary"
    )]
    fn dispatch_attention_forward(
        &self,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&Self::DeviceBuffer<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
        weights: &mut Self::DeviceBuffer<T>,
        weights_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        super::dispatch::forward::execute::<Self, T>(super::dispatch::forward::Forward {
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            key_padding_mask,
            key_padding_mask_layout,
            is_causal,
            scale,
            output,
            output_layout,
            weights,
            weights_layout,
        })
    }

    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the method mirrors the AttentionOps provider boundary"
    )]
    fn dispatch_attention_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        weights: &Self::DeviceBuffer<T>,
        weights_layout: &Layout,
        scale: T,
        grad_query: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
        grad_key: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
        grad_value: Option<(&mut Self::DeviceBuffer<T>, &Layout)>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        super::dispatch::backward::execute::<Self, T>(super::dispatch::backward::Backward {
            grad_output,
            grad_output_layout,
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            weights,
            weights_layout,
            scale,
            grad_query,
            grad_key,
            grad_value,
        })
    }
}

impl<P, T> AttentionBackend<T> for HephaestusBackend<P>
where
    P: AttentionProvider<T>,
    T: Scalar + Float + AttentionScalar,
{
    type Provider = P;
}
