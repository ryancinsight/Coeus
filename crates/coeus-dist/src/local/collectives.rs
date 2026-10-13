//! The [`Communicator`] collectives of [`LocalCommunicator`]: staged through
//! the cluster's shared buffers and synchronized by its barrier.

use super::LocalCommunicator;
use crate::communicator::{CollectiveError, Communicator};
use crate::host_access::{copy_host_slice_to_tensor, get_tensor_host_data};
use crate::ops::ReduceOpTag;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;
use std::convert::Infallible;

impl Communicator for LocalCommunicator {
    type Error = Infallible;

    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn barrier(&self) -> Result<(), Infallible> {
        self.shared.barrier.wait();
        Ok(())
    }

    fn all_reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        backend: &B,
    ) -> Result<(), CollectiveError<Infallible, B::Error>> {
        let numel = tensor.numel();
        if numel == 0 {
            return Ok(());
        }

        let host_data = get_tensor_host_data(tensor, backend)
            .map_err(CollectiveError::Backend)?
            .into_owned();

        // 1. Publish local staging data
        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        // 2. Barrier sync
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 3. Perform reduction once on rank 0 and publish it to slot 0.
        if self.rank == 0 {
            let staged = {
                let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "all_reduce")
            };
            let mut reduced = staged[..numel].to_vec();
            for r_data in staged.chunks_exact(numel).skip(1) {
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }

            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[0] = Some(Box::new(reduced));
        }

        // 4. Barrier sync to ensure reduced payload is published.
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 5. All ranks read reduced payload.
        let reduced = {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            let reduced = Self::slot_vec_ref::<T>(&bufs[0], 0, "all_reduce");
            Self::assert_numel(reduced.len(), numel, 0, "all_reduce");
            reduced.clone()
        };

        // 6. Barrier sync before clear.
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 7. Clear staging board
        if self.rank == 0 {
            self.clear_staging();
        }

        // 8. Barrier sync post clear
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 9. Transfer to device
        copy_host_slice_to_tensor(&reduced, tensor, backend).map_err(CollectiveError::Backend)?;
        Ok(())
    }

    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) -> Result<(), CollectiveError<Infallible, B::Error>> {
        assert!(
            root < self.size,
            "LocalCommunicator broadcast root out of bounds"
        );
        let numel = tensor.numel();
        if numel == 0 {
            return Ok(());
        }

        if self.rank == root {
            let host_data = get_tensor_host_data(tensor, backend)
                .map_err(CollectiveError::Backend)?
                .into_owned();
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[root] = Some(Box::new(host_data));
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        let mut broadcasted = Vec::new();
        if self.rank != root {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            let root_data = Self::slot_vec_ref::<T>(&bufs[root], root, "broadcast");
            Self::assert_numel(root_data.len(), numel, root, "broadcast");
            broadcasted = root_data.clone();
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        if self.rank == root {
            self.clear_staging();
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        if self.rank != root {
            copy_host_slice_to_tensor(&broadcasted, tensor, backend)
                .map_err(CollectiveError::Backend)?;
        }
        Ok(())
    }

    fn all_gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        backend: &B,
    ) -> Result<(), CollectiveError<Infallible, B::Error>> {
        assert_eq!(
            output.len(),
            self.size,
            "LocalCommunicator all_gather output length mismatch"
        );
        let numel = tensor.numel();
        for (r, out) in output.iter().enumerate() {
            assert_eq!(
                out.numel(),
                numel,
                "LocalCommunicator all_gather output numel mismatch at rank {}",
                r
            );
        }
        if numel == 0 {
            return Ok(());
        }

        let host_data = get_tensor_host_data(tensor, backend)
            .map_err(CollectiveError::Backend)?
            .into_owned();

        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        let staged = {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            Self::snapshot_payloads::<T>(&bufs, self.size, numel, "all_gather")
        };
        for (row, out) in staged.chunks_exact(numel).zip(output.iter_mut()) {
            copy_host_slice_to_tensor(row, out, backend).map_err(CollectiveError::Backend)?;
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        if self.rank == 0 {
            self.clear_staging();
        }

        self.barrier().map_err(CollectiveError::Communicator)?;
        Ok(())
    }

    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) -> Result<(), CollectiveError<Infallible, B::Error>> {
        assert!(
            root < self.size,
            "LocalCommunicator reduce root out of bounds"
        );
        let numel = tensor.numel();
        if numel == 0 {
            return Ok(());
        }

        let host_data = get_tensor_host_data(tensor, backend)
            .map_err(CollectiveError::Backend)?
            .into_owned();

        // 1. Publish local staging data
        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        // 2. Barrier sync
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 3. Perform reduction on root process
        let mut reduced = Vec::new();
        if self.rank == root {
            let staged = {
                let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "reduce")
            };
            reduced = staged[..numel].to_vec();
            for r_data in staged.chunks_exact(numel).skip(1) {
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }
        }

        // 4. Barrier sync before clear
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 5. Clear staging board
        if self.rank == root {
            self.clear_staging();
        }

        // 6. Barrier sync post clear
        self.barrier().map_err(CollectiveError::Communicator)?;

        // 7. Transfer to device on root
        if self.rank == root {
            copy_host_slice_to_tensor(&reduced, tensor, backend)
                .map_err(CollectiveError::Backend)?;
        }
        Ok(())
    }

    fn gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        root: usize,
        backend: &B,
    ) -> Result<(), CollectiveError<Infallible, B::Error>> {
        assert!(
            root < self.size,
            "LocalCommunicator gather root out of bounds"
        );
        let numel = tensor.numel();
        if self.rank == root {
            assert_eq!(
                output.len(),
                self.size,
                "LocalCommunicator gather output length mismatch on root"
            );
            for (r, out) in output.iter().enumerate() {
                assert_eq!(
                    out.numel(),
                    numel,
                    "LocalCommunicator gather output numel mismatch on root at rank {}",
                    r
                );
            }
        }
        if numel == 0 {
            return Ok(());
        }

        let host_data = get_tensor_host_data(tensor, backend)
            .map_err(CollectiveError::Backend)?
            .into_owned();

        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        if self.rank == root {
            let staged = {
                let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "gather")
            };
            for (row, out) in staged.chunks_exact(numel).zip(output.iter_mut()) {
                copy_host_slice_to_tensor(row, out, backend).map_err(CollectiveError::Backend)?;
            }
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        if self.rank == root {
            self.clear_staging();
        }

        self.barrier().map_err(CollectiveError::Communicator)?;
        Ok(())
    }

    fn scatter<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        input: &[Tensor<T, B>],
        root: usize,
        backend: &B,
    ) -> Result<(), CollectiveError<Infallible, B::Error>> {
        assert!(
            root < self.size,
            "LocalCommunicator scatter root out of bounds"
        );
        let numel = tensor.numel();
        if self.rank == root {
            assert_eq!(
                input.len(),
                self.size,
                "LocalCommunicator scatter input length mismatch on root"
            );
            for (r, in_tensor) in input.iter().enumerate() {
                assert_eq!(
                    in_tensor.numel(),
                    numel,
                    "LocalCommunicator scatter input numel mismatch on root at rank {}",
                    r
                );
            }
        }
        if numel == 0 {
            return Ok(());
        }

        if self.rank == root {
            // Every input tensor was asserted to numel above, so one
            // contiguous buffer at a fixed stride replaces a per-rank Vec.
            let mut staged_flat = Vec::with_capacity(self.size * numel);
            for in_tensor in input.iter().take(self.size) {
                staged_flat.extend(
                    get_tensor_host_data(in_tensor, backend)
                        .map_err(CollectiveError::Backend)?
                        .into_owned(),
                );
            }

            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            for (r, row) in staged_flat.chunks_exact(numel).enumerate() {
                bufs[r] = Some(Box::new(row.to_vec()));
            }
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        let scattered;
        {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            let rank_data = Self::slot_vec_ref::<T>(&bufs[self.rank], self.rank, "scatter");
            Self::assert_numel(rank_data.len(), numel, self.rank, "scatter");
            scattered = rank_data.clone();
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        if self.rank == root {
            self.clear_staging();
        }

        self.barrier().map_err(CollectiveError::Communicator)?;

        copy_host_slice_to_tensor(&scattered, tensor, backend).map_err(CollectiveError::Backend)?;
        Ok(())
    }
}
