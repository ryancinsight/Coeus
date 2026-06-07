pub mod ops;
pub mod communicator;
pub mod mock;
pub mod tcp;
pub(crate) mod helpers;

pub use ops::{ReduceOpTag, Sum, Max, Min, Product};
pub use communicator::Communicator;
pub use mock::MockCommunicator;
pub use tcp::{TcpMesh, TcpCommunicator};

use coeus_core::{Scalar, ComputeBackend};
use coeus_tensor::Tensor;
use coeus_autograd::Var;

/// Synchronize and average gradients across all ranks in a process group.
///
/// This does an in-place `all_reduce` summation of gradients, and then
/// scales each gradient by `1.0 / world_size` to obtain the average.
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
            let mut grad_tensor = g.lock().unwrap();
            
            // All-reduce (sum) across processes
            comm.all_reduce::<T, B, Sum>(&mut grad_tensor, &backend);

            // Scale by 1 / world_size
            coeus_ops::mul_assign(&mut grad_tensor, &scale_tensor, &backend);
        }
    }
}
