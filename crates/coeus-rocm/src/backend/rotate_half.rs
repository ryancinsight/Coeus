use super::{RocmBackend, RocmProvider};
use coeus_core::Layout;
use coeus_hephaestus::{
    rotate_half, HephaestusBackendError, HephaestusStorage, RotateHalfProvider,
};

impl RotateHalfProvider<f32> for RocmProvider {
    type Operations = hephaestus_rocm::RocmElementwiseOps;
}

impl coeus_ops::RotateHalfOps<f32> for RocmBackend {
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<f32>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        rotate_half::<RocmProvider, _>(input.buffer(), layout)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("rotate_half", source))
    }
}
