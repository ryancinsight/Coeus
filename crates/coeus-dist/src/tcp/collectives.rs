use super::error::TcpMeshError;
use super::mesh::TcpMesh;
use crate::communicator::Communicator;
use crate::host_access::{
    copy_host_slice_to_tensor, get_tensor_host_data, recv_slice_data, recv_tensor_data,
    with_tensor_host_bytes,
};
use crate::ops::ReduceOpTag;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

/// A socket-based communicator for distributed training.
///
/// Construction is infallible: mesh setup failures surface as a
/// [`TcpMeshError`] from [`TcpMesh::new`] or
/// [`TcpMesh::create_loopback_cluster`] before a communicator exists.
///
/// A collective that fails on any peer link poisons every link of this rank
/// and returns the [`TcpMeshError`]. Closing the links ends the waits of the
/// peers still in the collective, so every surviving rank returns a typed
/// error within its I/O deadline, and every later collective on this
/// communicator returns [`TcpMeshError::LinkPoisoned`].
pub struct TcpCommunicator {
    mesh: TcpMesh,
}

impl TcpCommunicator {
    /// Create a new TcpCommunicator wrapping an established TcpMesh.
    pub fn new(mesh: TcpMesh) -> Self {
        Self { mesh }
    }

    /// Gracefully close the underlying mesh's peer streams and stop its
    /// dedicated runtime.
    ///
    /// Delegates to [`TcpMesh::shutdown`]; every owner calls this before the
    /// communicator is dropped.
    pub fn shutdown(&mut self) {
        self.mesh.shutdown();
    }

    /// Run the steps of one collective, poisoning every link if any step
    /// fails: the collective's streams are then at unknown frame positions.
    fn collective<R>(
        &self,
        steps: impl FnOnce() -> Result<R, TcpMeshError>,
    ) -> Result<R, TcpMeshError> {
        steps().inspect_err(|_| self.mesh.poison_all_links())
    }

    #[inline]
    fn assert_numel(
        collective: &'static str,
        index: usize,
        actual_numel: usize,
        expected_numel: usize,
    ) {
        assert_eq!(
            actual_numel, expected_numel,
            "{collective} numel mismatch at rank index {index}: expected {expected_numel}, got {actual_numel}",
        );
    }

    #[inline]
    fn assert_root(root: usize, size: usize) {
        assert!(root < size, "collective root out of bounds");
    }

    /// The element count as sent on the wire.
    #[inline]
    fn wire_numel(numel: usize) -> u64 {
        // usize is at most 64 bits on every supported target.
        numel as u64
    }

    #[inline]
    fn recv_numel_from(&self, peer: usize) -> Result<u64, TcpMeshError> {
        let mut peer_numel_bytes = [0u8; 8];
        self.mesh.recv(peer, &mut peer_numel_bytes)?;
        Ok(u64::from_le_bytes(peer_numel_bytes))
    }

    /// The error for `peer` announcing `received` elements where this rank
    /// holds `expected`.
    fn numel_mismatch(&self, peer: usize, expected: u64, received: u64) -> TcpMeshError {
        TcpMeshError::NumelMismatch {
            rank: self.mesh.rank(),
            peer,
            address: self.mesh.peer_address(peer),
            expected,
            received,
        }
    }

    /// Agree on the element count with `root` before a rooted collective.
    ///
    /// Every rank sends its count to the root, which answers each with
    /// status 1 (all counts equal its own) or 0 (some count differs). A
    /// mismatch is an error on every rank rather than a desynchronized byte
    /// stream: the root reports the first differing peer, the others report
    /// the root's status 0.
    fn rooted_numel_handshake(
        &self,
        rank: usize,
        size: usize,
        root: usize,
        numel: usize,
    ) -> Result<(), TcpMeshError> {
        let local = Self::wire_numel(numel);
        if rank == root {
            let mut mismatch = None;
            for other in (0..size).filter(|&other| other != root) {
                let received = self.recv_numel_from(other)?;
                if mismatch.is_none() && received != local {
                    mismatch = Some((other, received));
                }
            }
            let status = [u8::from(mismatch.is_none())];
            for other in (0..size).filter(|&other| other != root) {
                self.mesh.send(other, &status)?;
            }
            if let Some((other, received)) = mismatch {
                return Err(self.numel_mismatch(other, local, received));
            }
        } else {
            self.mesh.send(root, &local.to_le_bytes())?;
            let mut status = [0u8; 1];
            self.mesh.recv(root, &mut status)?;
            match status[0] {
                1 => {}
                0 => {
                    return Err(TcpMeshError::PeerReportedMismatch {
                        rank,
                        peer: root,
                        address: self.mesh.peer_address(root),
                    });
                }
                status => {
                    return Err(TcpMeshError::InvalidStatus {
                        rank,
                        peer: root,
                        address: self.mesh.peer_address(root),
                        status,
                    });
                }
            }
        }
        Ok(())
    }

    /// Agree on the element count with every peer, pair by pair, before an
    /// all-gather; the lower rank of each pair sends first.
    fn pairwise_numel_handshake(
        &self,
        rank: usize,
        size: usize,
        numel: usize,
    ) -> Result<(), TcpMeshError> {
        let local = Self::wire_numel(numel);
        for other in (0..size).filter(|&other| other != rank) {
            let received = if rank < other {
                self.mesh.send(other, &local.to_le_bytes())?;
                self.recv_numel_from(other)?
            } else {
                let received = self.recv_numel_from(other)?;
                self.mesh.send(other, &local.to_le_bytes())?;
                received
            };
            if received != local {
                return Err(self.numel_mismatch(other, local, received));
            }
        }
        Ok(())
    }
}

impl Communicator for TcpCommunicator {
    type Error = TcpMeshError;

    #[inline]
    fn rank(&self) -> usize {
        self.mesh.rank()
    }

    #[inline]
    fn size(&self) -> usize {
        self.mesh.size()
    }

    fn barrier(&self) -> Result<(), TcpMeshError> {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        if size <= 1 {
            return Ok(());
        }
        self.collective(|| {
            let mut byte = [0u8; 1];
            if rank == 0 {
                for other in 1..size {
                    self.mesh.recv(other, &mut byte)?;
                }
                for other in 1..size {
                    self.mesh.send(other, &[1])?;
                }
            } else {
                self.mesh.send(0, &[1])?;
                self.mesh.recv(0, &mut byte)?;
            }
            Ok(())
        })
    }

    fn all_reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        backend: &B,
    ) -> Result<(), TcpMeshError> {
        self.reduce::<T, B, Op>(tensor, 0, backend)?;
        self.broadcast(tensor, 0, backend)
    }

    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) -> Result<(), TcpMeshError> {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        let numel = tensor.numel();
        if size <= 1 {
            return Ok(());
        }
        self.collective(|| {
            // Exchange expected payload lengths first so rank-shape mismatches
            // fail fast instead of desynchronizing the byte stream.
            self.rooted_numel_handshake(rank, size, root, numel)?;
            if numel == 0 {
                return Ok(());
            }
            if rank == root {
                with_tensor_host_bytes(tensor, backend, |slice| {
                    (0..size)
                        .filter(|&other| other != root)
                        .try_for_each(|other| self.mesh.send(other, slice))
                })
            } else {
                recv_tensor_data(tensor, backend, |slice| self.mesh.recv(root, slice))
            }
        })
    }

    fn all_gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        backend: &B,
    ) -> Result<(), TcpMeshError> {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        assert_eq!(output.len(), size, "all_gather output length mismatch");
        let numel = tensor.numel();
        for (idx, out) in output.iter().enumerate().take(size) {
            Self::assert_numel("all_gather output", idx, out.numel(), numel);
        }
        self.collective(|| {
            self.pairwise_numel_handshake(rank, size, numel)?;
            if numel == 0 {
                return Ok(());
            }

            let self_host_data = get_tensor_host_data(tensor, backend);
            copy_host_slice_to_tensor(&self_host_data, &mut output[rank], backend);

            with_tensor_host_bytes(tensor, backend, |send_raw_slice| {
                for (other, out_tensor) in output.iter_mut().enumerate().take(size) {
                    if other == rank {
                        continue;
                    }
                    if rank < other {
                        self.mesh.send(other, send_raw_slice)?;
                        recv_tensor_data(out_tensor, backend, |slice| {
                            self.mesh.recv(other, slice)
                        })?;
                    } else {
                        recv_tensor_data(out_tensor, backend, |slice| {
                            self.mesh.recv(other, slice)
                        })?;
                        self.mesh.send(other, send_raw_slice)?;
                    }
                }
                Ok(())
            })
        })
    }

    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) -> Result<(), TcpMeshError> {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        if size <= 1 {
            return Ok(());
        }
        let numel = tensor.numel();
        self.collective(|| {
            self.rooted_numel_handshake(rank, size, root, numel)?;
            if numel == 0 {
                return Ok(());
            }

            if rank == root {
                let mut reduced = get_tensor_host_data(tensor, backend).into_owned();
                let mut incoming = vec![T::zero(); numel];
                for other in (0..size).filter(|&other| other != root) {
                    recv_slice_data(&mut incoming, |slice| self.mesh.recv(other, slice))?;
                    for (acc, &value) in reduced.iter_mut().zip(&incoming) {
                        *acc = Op::apply(*acc, value);
                    }
                }
                copy_host_slice_to_tensor(&reduced, tensor, backend);
                Ok(())
            } else {
                with_tensor_host_bytes(tensor, backend, |slice| self.mesh.send(root, slice))
            }
        })
    }

    fn gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        root: usize,
        backend: &B,
    ) -> Result<(), TcpMeshError> {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        let numel = tensor.numel();
        if rank == root {
            assert_eq!(output.len(), size, "gather output length mismatch on root");
            for (idx, out) in output.iter().enumerate().take(size) {
                Self::assert_numel("gather output", idx, out.numel(), numel);
            }
        }
        self.collective(|| {
            self.rooted_numel_handshake(rank, size, root, numel)?;
            if numel == 0 {
                return Ok(());
            }

            if rank == root {
                let self_host_data = get_tensor_host_data(tensor, backend);
                copy_host_slice_to_tensor(&self_host_data, &mut output[root], backend);
                for (other, out_tensor) in output.iter_mut().enumerate().take(size) {
                    if other != root {
                        recv_tensor_data(out_tensor, backend, |slice| {
                            self.mesh.recv(other, slice)
                        })?;
                    }
                }
                Ok(())
            } else {
                with_tensor_host_bytes(tensor, backend, |slice| self.mesh.send(root, slice))
            }
        })
    }

    fn scatter<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        input: &[Tensor<T, B>],
        root: usize,
        backend: &B,
    ) -> Result<(), TcpMeshError> {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        let numel = tensor.numel();
        if rank == root {
            assert_eq!(input.len(), size, "scatter input length mismatch on root");
            for (idx, in_tensor) in input.iter().enumerate().take(size) {
                Self::assert_numel("scatter input", idx, in_tensor.numel(), numel);
            }
        }
        self.collective(|| {
            self.rooted_numel_handshake(rank, size, root, numel)?;
            if numel == 0 {
                return Ok(());
            }

            if rank == root {
                let self_host_data = get_tensor_host_data(&input[root], backend);
                copy_host_slice_to_tensor(&self_host_data, tensor, backend);
                for (other, in_tensor) in input.iter().enumerate().take(size) {
                    if other != root {
                        with_tensor_host_bytes(in_tensor, backend, |slice| {
                            self.mesh.send(other, slice)
                        })?;
                    }
                }
                Ok(())
            } else {
                recv_tensor_data(tensor, backend, |slice| self.mesh.recv(root, slice))
            }
        })
    }
}
