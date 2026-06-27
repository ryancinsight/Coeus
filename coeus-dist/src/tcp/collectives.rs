use super::mesh::TcpMesh;
use crate::communicator::Communicator;
use crate::helpers::{
    copy_host_slice_to_tensor, get_tensor_host_data, recv_slice_data, recv_tensor_data,
    with_tensor_host_bytes,
};
use crate::ops::ReduceOpTag;
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

/// A socket-based communicator for distributed training.
pub struct TcpCommunicator {
    mesh: TcpMesh,
}

impl TcpCommunicator {
    /// Create a new TcpCommunicator wrapping a TcpMesh.
    pub fn new(mesh: TcpMesh) -> Self {
        Self { mesh }
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
}

impl Communicator for TcpCommunicator {
    #[inline]
    fn rank(&self) -> usize {
        self.mesh.rank()
    }

    #[inline]
    fn size(&self) -> usize {
        self.mesh.size()
    }

    fn barrier(&self) {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        if size <= 1 {
            return;
        }
        if rank == 0 {
            let mut byte = [0u8; 1];
            for other in 1..size {
                self.mesh.recv(other, &mut byte);
            }
            for other in 1..size {
                self.mesh.send(other, &[1]);
            }
        } else {
            self.mesh.send(0, &[1]);
            let mut byte = [0u8; 1];
            self.mesh.recv(0, &mut byte);
        }
    }

    fn all_reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        backend: &B,
    ) {
        self.reduce::<T, B, Op>(tensor, 0, backend);
        self.broadcast(tensor, 0, backend);
    }

    fn broadcast<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        if size <= 1 {
            return;
        }
        if tensor.numel() == 0 {
            return;
        }

        if rank == root {
            with_tensor_host_bytes(tensor, backend, |slice| {
                for other in 0..size {
                    if other != root {
                        self.mesh.send(other, slice);
                    }
                }
            });
        } else {
            recv_tensor_data(tensor, backend, |slice| {
                self.mesh.recv(root, slice);
            });
        }
    }

    fn all_gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        backend: &B,
    ) {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        assert_eq!(output.len(), size, "all_gather output length mismatch");
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }
        for (idx, out) in output.iter().enumerate().take(size) {
            Self::assert_numel("all_gather output", idx, out.numel(), numel);
        }

        let self_host_data = get_tensor_host_data(tensor, backend);
        copy_host_slice_to_tensor(&self_host_data, &mut output[rank], backend);

        with_tensor_host_bytes(tensor, backend, |send_raw_slice| {
            for (other, out_tensor) in output.iter_mut().enumerate().take(size) {
                if other == rank {
                    continue;
                }
                if rank < other {
                    self.mesh.send(other, send_raw_slice);

                    recv_tensor_data(out_tensor, backend, |slice| {
                        self.mesh.recv(other, slice);
                    });
                } else {
                    recv_tensor_data(out_tensor, backend, |slice| {
                        self.mesh.recv(other, slice);
                    });

                    self.mesh.send(other, send_raw_slice);
                }
            }
        });
    }

    fn reduce<T: Scalar, B: ComputeBackend, Op: ReduceOpTag>(
        &self,
        tensor: &mut Tensor<T, B>,
        root: usize,
        backend: &B,
    ) {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        if size <= 1 {
            return;
        }
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        if rank == root {
            let mut reduced = get_tensor_host_data(tensor, backend).into_owned();
            let mut incoming = vec![T::zero(); numel];

            for other in 0..size {
                if other != root {
                    recv_slice_data(&mut incoming, |slice| {
                        self.mesh.recv(other, slice);
                    });
                    for i in 0..numel {
                        reduced[i] = Op::apply(reduced[i], incoming[i]);
                    }
                }
            }
            copy_host_slice_to_tensor(&reduced, tensor, backend);
        } else {
            with_tensor_host_bytes(tensor, backend, |slice| {
                self.mesh.send(root, slice);
            });
        }
    }

    fn gather<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &Tensor<T, B>,
        output: &mut [Tensor<T, B>],
        root: usize,
        backend: &B,
    ) {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        if rank == root {
            assert_eq!(output.len(), size, "gather output length mismatch on root");
            for (idx, out) in output.iter().enumerate().take(size) {
                Self::assert_numel("gather output", idx, out.numel(), numel);
            }
            let self_host_data = get_tensor_host_data(tensor, backend);
            copy_host_slice_to_tensor(&self_host_data, &mut output[root], backend);

            for (other, out_tensor) in output.iter_mut().enumerate().take(size) {
                if other != root {
                    recv_tensor_data(out_tensor, backend, |slice| {
                        self.mesh.recv(other, slice);
                    });
                }
            }
        } else {
            with_tensor_host_bytes(tensor, backend, |slice| {
                self.mesh.send(root, slice);
            });
        }
    }

    fn scatter<T: Scalar, B: ComputeBackend>(
        &self,
        tensor: &mut Tensor<T, B>,
        input: &[Tensor<T, B>],
        root: usize,
        backend: &B,
    ) {
        let rank = self.mesh.rank();
        let size = self.mesh.size();
        Self::assert_root(root, size);
        let numel = tensor.numel();
        if numel == 0 {
            return;
        }

        if rank == root {
            assert_eq!(input.len(), size, "scatter input length mismatch on root");
            for (idx, in_tensor) in input.iter().enumerate().take(size) {
                Self::assert_numel("scatter input", idx, in_tensor.numel(), numel);
            }
            let self_host_data = get_tensor_host_data(&input[root], backend);
            copy_host_slice_to_tensor(&self_host_data, tensor, backend);

            for (other, in_tensor) in input.iter().enumerate().take(size) {
                if other != root {
                    with_tensor_host_bytes(in_tensor, backend, |slice| {
                        self.mesh.send(other, slice);
                    });
                }
            }
        } else {
            recv_tensor_data(tensor, backend, |slice| {
                self.mesh.recv(root, slice);
            });
        }
    }
}
