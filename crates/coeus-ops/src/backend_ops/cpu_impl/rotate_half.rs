use crate::{CpuBackend, RotateHalfOps};
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

fn provider_error(source: leto::LetoError) -> BackendError {
    BackendError::Storage {
        operation: "rotate_half",
        reason: source.to_string(),
    }
}

impl<T, B> RotateHalfOps<T> for B
where
    T: Scalar + leto_ops::RealScalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<T>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        let plan = coeus_leto::prepare_rotate_half_input(layout, input.as_slice())
            .map_err(provider_error)?;
        let output_layout = Layout::new(layout.shape_cloned());
        let mut output = self.allocate_zeroed(layout.numel());
        coeus_leto::rotate_half_into(plan, &output_layout, output.as_mut_slice())
            .map_err(provider_error)?;
        Ok(output)
    }
}
