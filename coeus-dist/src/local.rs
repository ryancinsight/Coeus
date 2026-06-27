use crate::communicator::Communicator;
use crate::helpers::{copy_host_slice_to_tensor, get_tensor_host_data};
use crate::ops::ReduceOpTag;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use std::any::type_name;
use std::sync::{Arc, Barrier, Mutex};

/// Shared state for thread-based rank cluster simulation.
pub struct LocalClusterShared {
    barrier: Barrier,
    buffers: Mutex<Vec<Option<Box<dyn std::any::Any + Send>>>>,
}

/// A thread-safe simulated communicator for local multi-process verification.
///
/// Each rank shares a single [`LocalClusterShared`] state and coordinates via barriers,
/// so a real distributed run can be reproduced inside one process with threads.
///
/// # Examples
///
/// Spawn one thread per simulated rank, reduce gradients with [`Sum`](crate::Sum),
/// and verify every rank holds the summed result:
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_dist::{Communicator, LocalCommunicator, Sum};
/// use coeus_tensor::Tensor;
/// use std::thread;
///
/// let communicators = LocalCommunicator::create_cluster(3);
/// let mut handles = vec![];
/// for comm in communicators {
///     handles.push(thread::spawn(move || {
///         let backend = SequentialBackend::new();
///         let rank = comm.rank() as f32;
///         // rank r contributes [r+1, r+2] -> [1,2], [2,3], [3,4]
///         let mut tensor =
///             Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
///         comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);
///         // sum across 3 ranks: [1+2+3, 2+3+4] = [6, 9]
///         let data = tensor.as_slice();
///         assert_eq!(data[0], 6.0);
///         assert_eq!(data[1], 9.0);
///     }));
/// }
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
#[derive(Clone)]
pub struct LocalCommunicator {
    rank: usize,
    size: usize,
    shared: Arc<LocalClusterShared>,
}

impl LocalCommunicator {
    /// Create a new process cluster with `world_size` simulated ranks.
    ///
    /// Returns one [`LocalCommunicator`] per rank; move each into its own thread to
    /// simulate independent processes that synchronize through shared barriers.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_dist::LocalCommunicator;
    /// use coeus_dist::Communicator;
    ///
    /// let comms = LocalCommunicator::create_cluster(2);
    /// assert_eq!(comms.len(), 2);
    /// assert_eq!(comms[0].rank(), 0);
    /// assert_eq!(comms[1].rank(), 1);
    /// assert_eq!(comms[0].size(), 2);
    /// ```
    pub fn create_cluster(world_size: usize) -> Vec<Self> {
        assert!(world_size > 0, "LocalCommunicator world_size must be > 0");
        let shared = Arc::new(LocalClusterShared {
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

    #[inline]
    fn slot_vec_ref<'a, T: Scalar>(
        slot: &'a Option<Box<dyn std::any::Any + Send>>,
        rank: usize,
        collective: &'static str,
    ) -> &'a Vec<T> {
        let payload = slot.as_ref().unwrap_or_else(|| {
            panic!("{collective}: missing staging payload for rank {rank}");
        });
        payload.downcast_ref::<Vec<T>>().unwrap_or_else(|| {
            panic!(
                "{collective}: staging payload type mismatch on rank {rank}; expected {}",
                type_name::<Vec<T>>()
            )
        })
    }

    #[inline]
    fn assert_numel(data_len: usize, expected_numel: usize, rank: usize, collective: &'static str) {
        assert_eq!(
            data_len, expected_numel,
            "{collective}: payload numel mismatch for rank {rank}; expected {expected_numel}, got {data_len}",
        );
    }

    #[inline]
    fn snapshot_payloads<T: Scalar>(
        bufs: &[Option<Box<dyn std::any::Any + Send>>],
        size: usize,
        numel: usize,
        collective: &'static str,
    ) -> Vec<Vec<T>> {
        let mut staged = Vec::with_capacity(size);
        for (r, slot) in bufs.iter().enumerate().take(size) {
            let r_data = Self::slot_vec_ref::<T>(slot, r, collective);
            Self::assert_numel(r_data.len(), numel, r, collective);
            staged.push(r_data.clone());
        }
        staged
    }

    #[inline]
    fn clear_staging(&self) {
        let mut bufs = self.shared.buffers.lock().unwrap();
        for item in bufs.iter_mut() {
            *item = None;
        }
    }
}

impl Communicator for LocalCommunicator {
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

        // 3. Perform reduction once on rank 0 and publish it to slot 0.
        if self.rank == 0 {
            let staged = {
                let bufs = self.shared.buffers.lock().unwrap();
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "all_reduce")
            };
            let mut reduced = staged[0].clone();
            for r_data in staged.iter().skip(1) {
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }

            let mut bufs = self.shared.buffers.lock().unwrap();
            bufs[0] = Some(Box::new(reduced));
        }

        // 4. Barrier sync to ensure reduced payload is published.
        self.barrier();

        // 5. All ranks read reduced payload.
        let reduced = {
            let bufs = self.shared.buffers.lock().unwrap();
            let reduced = Self::slot_vec_ref::<T>(&bufs[0], 0, "all_reduce");
            Self::assert_numel(reduced.len(), numel, 0, "all_reduce");
            reduced.clone()
        };

        // 6. Barrier sync before clear.
        self.barrier();

        // 7. Clear staging board
        if self.rank == 0 {
            self.clear_staging();
        }

        // 8. Barrier sync post clear
        self.barrier();

        // 9. Transfer to device
        copy_host_slice_to_tensor(&reduced, tensor, backend);
    }

    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) {
        assert!(
            root < self.size,
            "LocalCommunicator broadcast root out of bounds"
        );
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

        let mut broadcasted = Vec::new();
        if self.rank != root {
            let bufs = self.shared.buffers.lock().unwrap();
            let root_data = Self::slot_vec_ref::<T>(&bufs[root], root, "broadcast");
            Self::assert_numel(root_data.len(), numel, root, "broadcast");
            broadcasted = root_data.clone();
        }

        self.barrier();

        if self.rank == root {
            self.clear_staging();
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
        assert_eq!(
            output.len(),
            self.size,
            "LocalCommunicator all_gather output length mismatch"
        );
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

        let staged = {
            let bufs = self.shared.buffers.lock().unwrap();
            Self::snapshot_payloads::<T>(&bufs, self.size, numel, "all_gather")
        };
        for r in 0..self.size {
            copy_host_slice_to_tensor(&staged[r], &mut output[r], backend);
        }

        self.barrier();

        if self.rank == 0 {
            self.clear_staging();
        }

        self.barrier();
    }

    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) {
        assert!(
            root < self.size,
            "LocalCommunicator reduce root out of bounds"
        );
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
        let mut reduced = Vec::new();
        if self.rank == root {
            let staged = {
                let bufs = self.shared.buffers.lock().unwrap();
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "reduce")
            };
            reduced = staged[0].clone();
            for r_data in staged.iter().skip(1) {
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }
        }

        // 4. Barrier sync before clear
        self.barrier();

        // 5. Clear staging board
        if self.rank == root {
            self.clear_staging();
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
        assert!(
            root < self.size,
            "LocalCommunicator gather root out of bounds"
        );
        if self.rank == root {
            assert_eq!(
                output.len(),
                self.size,
                "LocalCommunicator gather output length mismatch on root"
            );
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
            let staged = {
                let bufs = self.shared.buffers.lock().unwrap();
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "gather")
            };
            for r in 0..self.size {
                copy_host_slice_to_tensor(&staged[r], &mut output[r], backend);
            }
        }

        self.barrier();

        if self.rank == root {
            self.clear_staging();
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
        assert!(
            root < self.size,
            "LocalCommunicator scatter root out of bounds"
        );
        if self.rank == root {
            assert_eq!(
                input.len(),
                self.size,
                "LocalCommunicator scatter input length mismatch on root"
            );
        }
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        if self.rank == root {
            let staged_inputs = input
                .iter()
                .enumerate()
                .take(self.size)
                .map(|(r, in_tensor)| {
                    assert_eq!(
                        in_tensor.numel(),
                        numel,
                        "LocalCommunicator scatter input numel mismatch on root at rank {}",
                        r
                    );
                    get_tensor_host_data(in_tensor, backend).into_owned()
                })
                .collect::<Vec<Vec<T>>>();

            let mut bufs = self.shared.buffers.lock().unwrap();
            for (r, host_data) in staged_inputs.into_iter().enumerate() {
                bufs[r] = Some(Box::new(host_data));
            }
        }

        self.barrier();

        let scattered;
        {
            let bufs = self.shared.buffers.lock().unwrap();
            let rank_data = Self::slot_vec_ref::<T>(&bufs[self.rank], self.rank, "scatter");
            Self::assert_numel(rank_data.len(), numel, self.rank, "scatter");
            scattered = rank_data.clone();
        }

        self.barrier();

        if self.rank == root {
            self.clear_staging();
        }

        self.barrier();

        copy_host_slice_to_tensor(&scattered, tensor, backend);
    }
}
