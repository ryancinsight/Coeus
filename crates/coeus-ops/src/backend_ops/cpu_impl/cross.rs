use crate::backend_ops::defaults::cross::cross_fold;
use crate::{CpuBackend, CrossOps};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

impl<T, B> CrossOps<T> for B
where
    T: Scalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn cross_storage(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        dim: usize,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        let mut output = self.allocate_zeroed(a_layout.numel())?;
        cross_fold(
            a.as_slice(),
            b.as_slice(),
            a_layout,
            dim,
            output.as_mut_slice()?,
        );
        Ok(output)
    }
}
