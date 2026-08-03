use super::{MetalBackend, MetalProvider};
use coeus_core::Layout;
use coeus_hephaestus::{
    rotate_half, HephaestusBackendError, HephaestusStorage, RotateHalfProvider,
};

impl RotateHalfProvider<f32> for MetalProvider {
    type Operations = hephaestus_metal::MetalElementwiseOps;
}

impl coeus_ops::RotateHalfOps<f32> for MetalBackend {
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<f32>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        rotate_half::<MetalProvider, _>(input.buffer(), layout)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("rotate_half", source))
    }
}
