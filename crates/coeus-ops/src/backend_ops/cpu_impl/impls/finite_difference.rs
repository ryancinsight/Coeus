use super::super::finite_difference::{self, FdScalar};
use super::super::CpuBackend;
use crate::backend_ops::traits::{
    Axis, FiniteDifference3DOps, FiniteDifference3DScheme, StaggeredPairOps,
};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout};
use leto_ops::StaggeredLeapfrog3D;

impl<T: FdScalar, B: CpuBackend> StaggeredPairOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// The prepared pair is the provider's operator itself: on the CPU the
    /// buffers are already host memory, so there is nothing to upload and the
    /// derived taps are the whole of what preparation produces.
    type StaggeredPair = StaggeredLeapfrog3D<T>;

    #[inline]
    fn prepare_staggered_pair(
        &self,
        order: usize,
        spacing: [T; 3],
    ) -> Result<Self::StaggeredPair, Self::Error> {
        finite_difference::prepare_staggered_pair(order, spacing)
    }

    #[inline]
    fn staggered_gradient(
        &self,
        pair: &Self::StaggeredPair,
        axis: Axis,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let input = input.as_slice();
        finite_difference::staggered_gradient(
            pair,
            axis,
            input,
            input_layout,
            output.as_mut_slice(),
            output_layout,
        )
    }

    #[inline]
    fn staggered_divergence(
        &self,
        pair: &Self::StaggeredPair,
        axis: Axis,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let input = input.as_slice();
        finite_difference::staggered_divergence(
            pair,
            axis,
            input,
            input_layout,
            output.as_mut_slice(),
            output_layout,
        )
    }
}

impl<T: FdScalar, B: CpuBackend> FiniteDifference3DOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn finite_difference(
        &self,
        scheme: FiniteDifference3DScheme,
        axis: Axis,
        spacing: [T; 3],
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let input = input.as_slice();
        finite_difference::finite_difference(
            scheme,
            axis,
            spacing,
            input,
            input_layout,
            output.as_mut_slice(),
            output_layout,
        )
    }
}
