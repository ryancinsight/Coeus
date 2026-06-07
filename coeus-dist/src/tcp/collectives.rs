use coeus_core::{Scalar, ComputeBackend};
use coeus_tensor::Tensor;
use crate::communicator::Communicator;
use crate::ops::ReduceOpTag;
use crate::helpers::{
    get_tensor_host_data, copy_host_slice_to_tensor, with_tensor_host_bytes,
    recv_tensor_data, recv_slice_data,
};
use super::mesh::TcpMesh;

/// A socket-based communicator for distributed training.
pub struct TcpCommunicator {
    mesh: TcpMesh,
}

impl TcpCommunicator {
    /// Create a new TcpCommunicator wrapping a TcpMesh.
    pub fn new(mesh: TcpMesh) -> Self {
        Self { mesh }
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
        if tensor.numel() == 0 {
            return;
        }

        output[rank] = tensor.clone();

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
        if tensor.numel() == 0 {
            return;
        }

        if rank == root {
            assert_eq!(output.len(), size, "gather output length mismatch on root");
            output[root] = tensor.clone();

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
        if tensor.numel() == 0 {
            return;
        }

        if rank == root {
            assert_eq!(input.len(), size, "scatter input length mismatch on root");
            *tensor = input[root].clone();

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

