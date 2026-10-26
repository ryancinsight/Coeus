//! CPU digamma delegates layout traversal and scalar math to its providers.
use super::super::{error::map_leto_error, CpuBackend};
use crate::DigammaOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, Layout};

// SAFETY: Leto's unary_map_into overwrites each logical output on success.
unsafe impl<T: Float + leto_ops::RealScalar, B: CpuBackend> DigammaOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn elementwise_digamma(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        coeus_cpu::elementwise_digamma_into(
            input_layout,
            input.as_slice(),
            output_layout,
            output.as_mut_slice()?,
        )
        .map_err(|error| map_leto_error("digamma", error))
    }
}
