use std::sync::{Arc, Barrier, Mutex};
use coeus_core::{Scalar, ComputeBackend};
use coeus_tensor::Tensor;
use crate::communicator::Communicator;
use crate::ops::ReduceOpTag;
use crate::helpers::{get_tensor_host_data, copy_host_slice_to_tensor};

/// Shared state for thread-based rank cluster simulation.
pub struct MockClusterShared {
    barrier: Barrier,
    buffers: Mutex<Vec<Option<Box<dyn std::any::Any + Send>>>>,
}

/// A thread-safe simulated communicator for local multi-process verification.
#[derive(Clone)]
pub struct MockCommunicator {
    rank: usize,
    size: usize,
    shared: Arc<MockClusterShared>,
}

impl MockCommunicator {
    /// Create a new process cluster with `world_size` simulated ranks.
    pub fn create_cluster(world_size: usize) -> Vec<Self> {
        let shared = Arc::new(MockClusterShared {
            barrier: Barrier::new(world_size),
            buffers: Mutex::new((0..world_size).map(|_| None).collect()),
        });
        (0..world_size)
            .map(|rank| Self {
                rank,
                size: world_size,
                shared: shared.clone(),
            })
            .collect()
    }
}

impl Communicator for MockCommunicator {
    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn barrier(&self) {
        self.shared.barrier.wait();
    }

    fn all_reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        backend: &B,
    ) {
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        // 1. Publish local staging data
        {
            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[self.rank] = Some(Box::new(host_data));
        }

        // 2. Barrier sync
        self.barrier();

        // 3. Perform reduction on host
        let mut reduced = vec![T::zero(); numel];
        {
            let bufs = self.shared.buffers.lock().unwrap();
            let r0_data = bufs[0].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
            reduced.copy_from_slice(r0_data);

            for r in 1..self.size {
                let r_data = bufs[r].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }
        }

        // 4. Barrier sync before clear
        self.barrier();

        // 5. Clear staging board
        if self.rank == 0 {
            let mut bufs = self.shared.buffers.lock().unwrap();
            for item in bufs.iter_mut() {
                *item = None;
            }
        }

        // 6. Barrier sync post clear
        self.barrier();

        // 7. Transfer to device
        copy_host_slice_to_tensor(&reduced, tensor, backend);
    }

    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) {
        assert!(root < self.size, "MockCommunicator broadcast root out of bounds");
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        if self.rank == root {
            let host_data = get_tensor_host_data(tensor, backend).into_owned();
            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[root] = Some(Box::new(host_data));
        }

        self.barrier();

        let mut broadcasted = vec![T::zero(); numel];
        if self.rank != root {
            let bufs = self.shared.buffers.lock().unwrap();
            let root_data = bufs[root].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
            broadcasted.copy_from_slice(root_data);
        }

        self.barrier();

        if self.rank == root {
            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[root] = None;
        }

        self.barrier();

        if self.rank != root {
            copy_host_slice_to_tensor(&broadcasted, tensor, backend);
        }
    }

    fn all_gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        backend: &B,
    ) {
        assert_eq!(output.len(), self.size, "MockCommunicator all_gather output length mismatch");
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        {
            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[self.rank] = Some(Box::new(host_data));
        }

        self.barrier();

        {
            let bufs = self.shared.buffers.lock().unwrap();
            for r in 0..self.size {
                let r_data = bufs[r].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
                copy_host_slice_to_tensor(r_data, &mut output[r], backend);
            }
        }

        self.barrier();

        if self.rank == 0 {
            let mut bufs = self.shared.buffers.lock().unwrap();
            for item in bufs.iter_mut() {
                *item = None;
            }
        }

        self.barrier();
    }

    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) {
        assert!(root < self.size, "MockCommunicator reduce root out of bounds");
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        // 1. Publish local staging data
        {
            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[self.rank] = Some(Box::new(host_data));
        }

        // 2. Barrier sync
        self.barrier();

        // 3. Perform reduction on root process
        let mut reduced = vec![T::zero(); numel];
        if self.rank == root {
            let bufs = self.shared.buffers.lock().unwrap();
            let r0_data = bufs[0].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
            reduced.copy_from_slice(r0_data);

            for r in 1..self.size {
                let r_data = bufs[r].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }
        }

        // 4. Barrier sync before clear
        self.barrier();

        // 5. Clear staging board
        if self.rank == root {
            let mut bufs = self.shared.buffers.lock().unwrap();
            for item in bufs.iter_mut() {
                *item = None;
            }
        }

        // 6. Barrier sync post clear
        self.barrier();

        // 7. Transfer to device on root
        if self.rank == root {
            copy_host_slice_to_tensor(&reduced, tensor, backend);
        }
    }

    fn gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        root: usize,
        backend: &B,
    ) {
        assert!(root < self.size, "MockCommunicator gather root out of bounds");
        if self.rank == root {
            assert_eq!(output.len(), self.size, "MockCommunicator gather output length mismatch on root");
        }
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        {
            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[self.rank] = Some(Box::new(host_data));
        }

        self.barrier();

        if self.rank == root {
            let bufs = self.shared.buffers.lock().unwrap();
            for r in 0..self.size {
                let r_data = bufs[r].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
                copy_host_slice_to_tensor(r_data, &mut output[r], backend);
            }
        }

        self.barrier();

        if self.rank == root {
            let mut bufs = self.shared.buffers.lock().unwrap();
            for item in bufs.iter_mut() {
                *item = None;
            }
        }

        self.barrier();
    }

    fn scatter<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        input: &[Tensor<T, B>],
        root: usize,
        backend: &B,
    ) {
        assert!(root < self.size, "MockCommunicator scatter root out of bounds");
        if self.rank == root {
            assert_eq!(input.len(), self.size, "MockCommunicator scatter input length mismatch on root");
        }
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        if self.rank == root {
            let mut bufs = self.shared.buffers.lock().unwrap();
            for r in 0..self.size {
                let host_data = get_tensor_host_data(&input[r], backend).into_owned();
                bufs[r] = Some(Box::new(host_data));
            }
        }

        self.barrier();

        let mut scattered = vec![T::zero(); numel];
        {
            let bufs = self.shared.buffers.lock().unwrap();
            let rank_data = bufs[self.rank].as_ref().unwrap().downcast_ref::<Vec<T>>().unwrap();
            scattered.copy_from_slice(rank_data);
        }

        self.barrier();

        if self.rank == root {
            let mut bufs = self.shared.buffers.lock().unwrap();
            for item in bufs.iter_mut() {
                *item = None;
            }
        }

        self.barrier();

        copy_host_slice_to_tensor(&scattered, tensor, backend);
    }
}
