use coeus_core::Scalar;
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
///         let Ok(()) = comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);
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
        let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
        for item in bufs.iter_mut() {
            *item = None;
        }
    }
}

mod collectives;
