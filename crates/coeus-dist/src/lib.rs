#![deny(missing_docs)]

//! Distributed training primitives for the Coeus framework.
//!
//! Provides a [`Communicator`] trait abstracting process-group communication, with
//! a thread-based [`LocalCommunicator`] for single-process verification and a socket-based
//! [`TcpCommunicator`]/[`TcpMesh`] for real multi-process runs. Gradient averaging is
//! exposed via [`synchronize_gradients`].

/// Collective communication interface implemented by all communicators.
pub mod communicator;
pub(crate) mod host_access;
/// Thread-based simulated communicator for local multi-process verification.
pub mod local;
/// Zero-sized reduction-operation tags (`Sum`, `Min`, `Max`, `Product`).
pub mod ops;
/// Socket-based communicators for real multi-process training.
pub mod tcp;

pub use communicator::Communicator;
pub use local::LocalCommunicator;
pub use ops::{Max, Min, Product, ReduceOpTag, Sum};
pub use tcp::{TcpCommunicator, TcpMesh};

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

/// Synchronize and average gradients across all ranks in a process group.
///
/// This does an in-place `all_reduce` summation of gradients, and then
/// scales each gradient by `1.0 / world_size` to obtain the average.
///
/// # Examples
///
/// With `world_size = 2`, rank 0 contributes `[1, 10]` and rank 1 contributes `[2, 11]`.
/// The summed gradient is `[3, 21]`, averaged to `[1.5, 10.5]`:
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{synchronize_gradients, Communicator, LocalCommunicator};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let communicators = LocalCommunicator::create_cluster(2);
/// let mut handles = vec![];
/// for comm in communicators {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         let x = Var::new(Tensor::zeros_on([2], &backend), true);
///         x.set_grad(Tensor::from_slice_on([2], &[rank + 1.0, rank + 10.0], &backend));
///
///         let mut params = vec![x];
///         synchronize_gradients(&mut params, &comm);
///
///         let synced_grad = params[0].grad().unwrap();
///         let data = synced_grad.as_slice();
///         // (1+2)/2 = 1.5, (10+11)/2 = 10.5
///         assert_eq!(data[0], 1.5);
///         assert_eq!(data[1], 10.5);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
pub fn synchronize_gradients<
    T: Scalar,
    B: ComputeBackend + coeus_ops::BackendOps<T> + Default,
    C: Communicator,
>(
    params: &mut [Var<T, B>],
    comm: &C,
) {
    let size = comm.size();
    if size <= 1 {
        return;
    }
    let backend = B::default();
    let scale_val = T::from_f64(1.0 / size as f64);
    let scale_tensor = Tensor::full_on([1], scale_val, &backend);

    for param in params {
        if let Some(ref g) = param.grad {
            let grad_tensor = g.write();

            // All-reduce (sum) across processes
            comm.all_reduce::<T, B, Sum>(grad_tensor, &backend);

            // Scale by 1 / world_size
            coeus_ops::mul_assign(grad_tensor, &scale_tensor, &backend)
                .expect("distributed gradient scaling backend operation");
        }
    }
}

