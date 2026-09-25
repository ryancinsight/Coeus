//! The [`Communicator`] collectives of [`LocalCommunicator`]: staged through
//! the cluster's shared buffers and synchronized by its barrier.

use super::LocalCommunicator;
use crate::communicator::Communicator;
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
    ) -> Result<(), Infallible> {
        let numel = tensor.numel();
        if numel == 0 {
            return Ok(());
        }

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        // 1. Publish local staging data
        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        // 2. Barrier sync
        self.barrier();

        // 3. Perform reduction once on rank 0 and publish it to slot 0.
        if self.rank == 0 {
            let staged = {
                let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
                Self::snapshot_payloads::<T>(&bufs, self.size, numel, "all_reduce")
            };
            let mut reduced = staged[0].clone();
            for r_data in staged.iter().skip(1) {
                for i in 0..numel {
                    reduced[i] = Op::apply(reduced[i], r_data[i]);
                }
            }

            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[0] = Some(Box::new(reduced));
        }

        // 4. Barrier sync to ensure reduced payload is published.
        self.barrier();

        // 5. All ranks read reduced payload.
        let reduced = {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
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
        Ok(())
    }

    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) -> Result<(), Infallible> {
        assert!(
            root < self.size,
            "LocalCommunicator broadcast root out of bounds"
        );
        let numel = tensor.numel();
        if numel == 0 {
            return Ok(());
        }

        if self.rank == root {
            let host_data = get_tensor_host_data(tensor, backend).into_owned();
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[root] = Some(Box::new(host_data));
        }

        self.barrier();

        let mut broadcasted = Vec::new();
        if self.rank != root {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
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
        Ok(())
    }

    fn all_gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        backend: &B,
    ) -> Result<(), Infallible> {
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

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        self.barrier();

        let staged = {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
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
        Ok(())
    }

    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) -> Result<(), Infallible> {
        assert!(
            root < self.size,
            "LocalCommunicator reduce root out of bounds"
        );
        let numel = tensor.numel();
        if numel == 0 {
            return Ok(());
        }

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        // 1. Publish local staging data
        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        // 2. Barrier sync
        self.barrier();

        // 3. Perform reduction on root process
        let mut reduced = Vec::new();
        if self.rank == root {
            let staged = {
                let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
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
        Ok(())
    }

    fn gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        root: usize,
        backend: &B,
    ) -> Result<(), Infallible> {
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

        let host_data = get_tensor_host_data(tensor, backend).into_owned();

        {
            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            bufs[self.rank] = Some(Box::new(host_data));
        }

        self.barrier();

        if self.rank == root {
            let staged = {
                let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
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
        Ok(())
    }

    fn scatter<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        input: &[Tensor<T, B>],
        root: usize,
        backend: &B,
    ) -> Result<(), Infallible> {
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
            let staged_inputs = input
                .iter()
                .enumerate()
                .take(self.size)
                .map(|(_, in_tensor)| get_tensor_host_data(in_tensor, backend).into_owned())
                .collect::<Vec<Vec<T>>>();

            let mut bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
            for (r, host_data) in staged_inputs.into_iter().enumerate() {
                bufs[r] = Some(Box::new(host_data));
            }
        }

        self.barrier();

        let scattered;
        {
            let bufs = self.shared.buffers.lock().expect("invariant: no prior holder of the local-cluster staging lock panicked while holding it");
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
        Ok(())
    }
}
