use coeus_core::{Scalar, ComputeBackend};
use coeus_tensor::Tensor;
use crate::ops::ReduceOpTag;

/// Abstract interface for distributed process communication.
///
/// Implementations mediate synchronization, broadcasting, scattering, and reduction.
pub trait Communicator: Send + Sync + 'static {
    /// Get the rank of the current process within the process group.
    fn rank(&self) -> usize;

    /// Get the total number of processes in the process group.
    fn size(&self) -> usize;

    /// Synchronize all ranks in the process group (blocking barrier).
    fn barrier(&self);

    /// Reduce and distribute a tensor to all processes in-place.
    fn all_reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        backend: &B,
    );

    /// Broadcast a tensor from the root process rank to all other processes in-place.
    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    );

    /// Gather tensors from all processes into a slice of tensors.
    ///
    /// The length of `output` must be equal to `self.size()`.
    fn all_gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        backend: &B,
    );

    /// Reduce a tensor from all processes to a single root process.
    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    );

    /// Gather tensors from all processes into a single slice on the root process.
    fn gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        root: usize,
        backend: &B,
    );

    /// Scatter a slice of tensors from the root process to all processes in-place.
    fn scatter<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        input: &[Tensor<T, B>],
        root: usize,
        backend: &B,
    );
}
