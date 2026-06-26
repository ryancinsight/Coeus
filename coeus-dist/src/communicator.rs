use crate::ops::ReduceOpTag;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

/// Abstract interface for distributed process communication.
///
/// Implementations mediate synchronization, broadcasting, scattering, and reduction.
///
/// # Examples
///
/// Create a simulated two-rank cluster, then use [`Communicator::all_reduce`] with the
/// [`Sum`](crate::Sum) reduction tag to combine per-rank gradients in-place:
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{Communicator, LocalCommunicator, Sum};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let communicators = LocalCommunicator::create_cluster(2);
/// let mut handles = vec![];
/// for comm in communicators {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         // rank 0 -> [1.0, 2.0], rank 1 -> [2.0, 3.0]
///         let mut tensor =
///             Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
///         comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);
///         // sum across ranks: [1+2, 2+3] = [3, 5]
///         let data = tensor.as_slice();
///         assert_eq!(data[0], 3.0);
///         assert_eq!(data[1], 5.0);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
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
