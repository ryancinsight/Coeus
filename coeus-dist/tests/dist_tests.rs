use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_dist::{
    synchronize_gradients, Communicator, LocalCommunicator, Max, Min, Product, Sum,
    TcpCommunicator, TcpMesh,
};
use coeus_tensor::Tensor;
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn assert_any_thread_panicked(handles: Vec<thread::JoinHandle<()>>, message: &str) {
    let total = handles.len();
    let (tx, rx) = mpsc::channel();
    for handle in handles {
        let tx = tx.clone();
        thread::spawn(move || {
            let _ = tx.send(handle.join().is_err());
        });
    }
    drop(tx);

    let deadline = Instant::now() + Duration::from_secs(20);
    let mut completed = 0usize;
    let mut panicked = false;
    while completed < total {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok(did_panic) => {
                completed += 1;
                panicked |= did_panic;
                if panicked {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    assert!(panicked, "{}", message);
}

#[test]
fn test_local_all_reduce() {
    let world_size = 3;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);

            comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);

            let data = tensor.as_slice();
            assert_eq!(data[0], 6.0);
            assert_eq!(data[1], 9.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_broadcast() {
    let world_size = 4;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let mut tensor = if comm.rank() == 2 {
                Tensor::from_slice_on([2], &[42.0f32, 100.0], &backend)
            } else {
                Tensor::zeros_on([2], &backend)
            };

            comm.broadcast(&mut tensor, 2, &backend);

            let data = tensor.as_slice();
            assert_eq!(data[0], 42.0);
            assert_eq!(data[1], 100.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_all_gather() {
    let world_size = 3;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let tensor = Tensor::from_slice_on([1], &[rank * 10.0], &backend);

            let mut output = vec![
                Tensor::zeros_on([1], &backend),
                Tensor::zeros_on([1], &backend),
                Tensor::zeros_on([1], &backend),
            ];

            comm.all_gather(&tensor, &mut output, &backend);

            assert_eq!(output[0].as_slice()[0], 0.0);
            assert_eq!(output[1].as_slice()[0], 10.0);
            assert_eq!(output[2].as_slice()[0], 20.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_gradient_synchronization() {
    let world_size = 2;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let x = Var::new(Tensor::zeros_on([2], &backend), true);
            let grad_val = Tensor::from_slice_on([2], &[rank + 1.0, rank + 10.0], &backend);
            x.set_grad(grad_val);

            let mut params = vec![x];
            synchronize_gradients(&mut params, &comm);

            let synced_grad = params[0].grad().unwrap();
            let data = synced_grad.as_slice();
            assert_eq!(data[0], 1.5);
            assert_eq!(data[1], 10.5);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_reduce() {
    let world_size = 3;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);

            comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend);

            if comm.rank() == 1 {
                let data = tensor.as_slice();
                assert_eq!(data[0], 6.0);
                assert_eq!(data[1], 9.0);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_gather() {
    let world_size = 3;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let tensor = Tensor::from_slice_on([1], &[rank * 10.0], &backend);

            let mut output = if comm.rank() == 2 {
                vec![
                    Tensor::zeros_on([1], &backend),
                    Tensor::zeros_on([1], &backend),
                    Tensor::zeros_on([1], &backend),
                ]
            } else {
                vec![]
            };

            comm.gather(&tensor, &mut output, 2, &backend);

            if comm.rank() == 2 {
                assert_eq!(output[0].as_slice()[0], 0.0);
                assert_eq!(output[1].as_slice()[0], 10.0);
                assert_eq!(output[2].as_slice()[0], 20.0);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_scatter() {
    let world_size = 3;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let mut tensor = Tensor::zeros_on([1], &backend);

            let input = if comm.rank() == 0 {
                vec![
                    Tensor::from_slice_on([1], &[100.0], &backend),
                    Tensor::from_slice_on([1], &[200.0], &backend),
                    Tensor::from_slice_on([1], &[300.0], &backend),
                ]
            } else {
                vec![]
            };

            comm.scatter(&mut tensor, &input, 0, &backend);

            let rank = comm.rank() as f32;
            assert_eq!(tensor.as_slice()[0], (rank + 1.0) * 100.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
#[should_panic(expected = "LocalCommunicator scatter input numel mismatch on root at rank 0")]
fn test_local_scatter_mismatched_input_numel_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::zeros_on([2], &backend);
    let input = vec![Tensor::from_slice_on([1], &[3.0f32], &backend)];
    comm.scatter(&mut tensor, &input, 0, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator world_size must be > 0")]
fn test_local_create_cluster_zero_world_size_panics() {
    let _ = LocalCommunicator::create_cluster(0);
}

#[test]
#[should_panic(expected = "LocalCommunicator broadcast root out of bounds")]
fn test_local_broadcast_root_out_of_bounds_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    comm.broadcast(&mut tensor, 1, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator reduce root out of bounds")]
fn test_local_reduce_root_out_of_bounds_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator gather root out of bounds")]
fn test_local_gather_root_out_of_bounds_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 1, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator scatter root out of bounds")]
fn test_local_scatter_root_out_of_bounds_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::zeros_on([1], &backend);
    let input = vec![Tensor::from_slice_on([1], &[1.0f32], &backend)];
    comm.scatter(&mut tensor, &input, 1, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator all_gather output length mismatch")]
fn test_local_all_gather_zero_numel_output_len_mismatch_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.all_gather(&tensor, &mut output, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator all_gather output numel mismatch at rank 0")]
fn test_local_all_gather_mismatched_output_numel_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.all_gather(&tensor, &mut output, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator all_gather output numel mismatch at rank 0")]
fn test_local_all_gather_zero_numel_output_numel_mismatch_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.all_gather(&tensor, &mut output, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator gather output length mismatch on root")]
fn test_local_gather_zero_numel_output_len_mismatch_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.gather(&tensor, &mut output, 0, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator gather output numel mismatch on root at rank 0")]
fn test_local_gather_mismatched_output_numel_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 0, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator gather output numel mismatch on root at rank 0")]
fn test_local_gather_zero_numel_output_numel_mismatch_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 0, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator scatter input length mismatch on root")]
fn test_local_scatter_zero_numel_input_len_mismatch_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let input: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.scatter(&mut tensor, &input, 0, &backend);
}

#[test]
#[should_panic(expected = "LocalCommunicator scatter input numel mismatch on root at rank 0")]
fn test_local_scatter_zero_numel_input_numel_mismatch_panics() {
    let comm = LocalCommunicator::create_cluster(1).remove(0);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let input = vec![Tensor::zeros_on([1], &backend)];
    comm.scatter(&mut tensor, &input, 0, &backend);
}

#[test]
fn test_local_all_reduce_sliced() {
    let world_size = 2;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;

            let parent = Tensor::from_slice_on([4], &[0.0, rank + 1.0, rank + 2.0, 0.0], &backend);
            let mut slice = parent.slice(&[(1, 3)]);
            comm.all_reduce::<f32, _, Sum>(&mut slice, &backend);

            let data = slice.as_slice();
            assert_eq!(data[0], 3.0);
            assert_eq!(data[1], 5.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_broadcast_sliced() {
    let world_size = 2;
    let communicators = LocalCommunicator::create_cluster(world_size);
    let mut handles = vec![];

    for comm in communicators {
        let handle = thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank();

            let parent = if rank == 0 {
                Tensor::from_slice_on([2, 2], &[1.0f32, 2.0, 3.0, 4.0], &backend)
            } else {
                Tensor::zeros_on([2, 2], &backend)
            };

            let mut view = parent.t();
            comm.broadcast(&mut view, 0, &backend);

            let contig = view.to_contiguous();
            let data = contig.as_slice();
            assert_eq!(data[0], 1.0);
            assert_eq!(data[1], 3.0);
            assert_eq!(data[2], 2.0);
            assert_eq!(data[3], 4.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

struct PortAllocatorLock {
    path: PathBuf,
}

impl Drop for PortAllocatorLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn port_allocator_lock() -> PortAllocatorLock {
    let path = std::env::temp_dir().join("coeus-dist-tcp-port-counter.lock");
    let start = Instant::now();

    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return PortAllocatorLock { path },
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
                ) =>
            {
                if start.elapsed() > Duration::from_secs(20) {
                    let metadata = fs::metadata(&path)
                        .expect("TCP port allocator lock exists but metadata could not be read");
                    let modified = metadata
                        .modified()
                        .expect("TCP port allocator lock modified time could not be read");
                    if modified.elapsed().unwrap_or(Duration::ZERO) > Duration::from_secs(120) {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    panic!(
                        "timed out waiting for TCP port allocator lock at {}",
                        path.display()
                    );
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!(
                "failed to acquire TCP port allocator lock at {}: {error}",
                path.display()
            ),
        }
    }
}

fn get_free_ports(count: usize) -> Vec<std::net::SocketAddr> {
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};

    assert!(count > 0, "port count must be positive");

    let _lock = port_allocator_lock();
    let counter_path = std::env::temp_dir().join("coeus-dist-tcp-port-counter.txt");
    let mut start = fs::read_to_string(&counter_path)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(30_000);

    for _ in 0..4096 {
        if start < 30_000 || start + count >= u16::MAX as usize {
            start = 30_000;
            continue;
        }

        let mut addrs = Vec::with_capacity(count);
        let mut listeners = Vec::with_capacity(count);
        let mut all_bound = true;

        for offset in 0..count {
            let port = (start + offset) as u16;
            let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
            match TcpListener::bind(addr) {
                Ok(listener) => {
                    addrs.push(addr);
                    listeners.push(listener);
                }
                Err(_) => {
                    all_bound = false;
                    break;
                }
            }
        }

        if all_bound {
            let next = start + count + 1;
            fs::write(
                &counter_path,
                if next + count < u16::MAX as usize {
                    next.to_string()
                } else {
                    "30000".to_string()
                },
            )
            .expect("failed to persist TCP test port counter");
            drop(listeners);
            return addrs;
        }

        start += count + 1;
    }

    panic!("failed to reserve {count} local TCP test ports after multiple attempts");
}

#[test]
fn test_tcp_all_reduce() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor =
                Tensor::from_slice_on([2], &[(rank + 1) as f32, (rank + 2) as f32], &backend);
            comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);

            let data = tensor.as_slice();
            assert_eq!(data[0], 3.0);
            assert_eq!(data[1], 5.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_all_reduce_mismatched_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend)
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend)
            };
            comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP all_reduce with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_all_reduce_zero_numel_mismatched_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::<f32, _>::zeros_on([0], &backend)
            } else {
                Tensor::zeros_on([1], &backend)
            };
            comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP all_reduce zero-numel with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_broadcast() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[10.0f32, 20.0], &backend)
            } else {
                Tensor::zeros_on([2], &backend)
            };

            comm.broadcast(&mut tensor, 0, &backend);

            let data = tensor.as_slice();
            assert_eq!(data[0], 10.0);
            assert_eq!(data[1], 20.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_broadcast_mismatched_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[10.0f32, 20.0], &backend)
            } else {
                Tensor::zeros_on([1], &backend)
            };

            comm.broadcast(&mut tensor, 0, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP broadcast with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_all_gather() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = Tensor::from_slice_on([1], &[(rank * 100) as f32], &backend);
            let mut output = vec![
                Tensor::zeros_on([1], &backend),
                Tensor::zeros_on([1], &backend),
            ];

            comm.all_gather(&tensor, &mut output, &backend);

            assert_eq!(output[0].as_slice()[0], 0.0);
            assert_eq!(output[1].as_slice()[0], 100.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_all_gather_mismatched_peer_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend)
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend)
            };
            let mut output = if rank == 0 {
                vec![
                    Tensor::zeros_on([2], &backend),
                    Tensor::zeros_on([2], &backend),
                ]
            } else {
                vec![
                    Tensor::zeros_on([1], &backend),
                    Tensor::zeros_on([1], &backend),
                ]
            };

            comm.all_gather(&tensor, &mut output, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP all_gather with mismatched peer tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_all_gather_zero_numel_mismatched_peer_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = if rank == 0 {
                Tensor::<f32, _>::zeros_on([0], &backend)
            } else {
                Tensor::zeros_on([1], &backend)
            };
            let mut output = if rank == 0 {
                vec![
                    Tensor::zeros_on([0], &backend),
                    Tensor::zeros_on([0], &backend),
                ]
            } else {
                vec![
                    Tensor::zeros_on([1], &backend),
                    Tensor::zeros_on([1], &backend),
                ]
            };

            comm.all_gather(&tensor, &mut output, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP all_gather zero-numel with mismatched peer tensor numel should panic on at least one rank"
    );
}

#[test]
fn test_tcp_barrier() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);

            comm.barrier();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_reduce() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor =
                Tensor::from_slice_on([2], &[(rank + 1) as f32, (rank + 2) as f32], &backend);
            comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend);

            if rank == 1 {
                let data = tensor.as_slice();
                assert_eq!(data[0], 3.0);
                assert_eq!(data[1], 5.0);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_reduce_mismatched_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend)
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend)
            };

            comm.reduce::<f32, _, Sum>(&mut tensor, 0, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP reduce with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_gather() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = Tensor::from_slice_on([1], &[(rank * 100) as f32], &backend);
            let mut output = if rank == 1 {
                vec![
                    Tensor::zeros_on([1], &backend),
                    Tensor::zeros_on([1], &backend),
                ]
            } else {
                vec![]
            };

            comm.gather(&tensor, &mut output, 1, &backend);

            if rank == 1 {
                assert_eq!(output[0].as_slice()[0], 0.0);
                assert_eq!(output[1].as_slice()[0], 100.0);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_gather_mismatched_peer_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();
            let tensor = if rank == 1 {
                Tensor::from_slice_on([2], &[11.0f32, 12.0], &backend)
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend)
            };
            let mut output = if rank == 1 {
                vec![
                    Tensor::zeros_on([2], &backend),
                    Tensor::zeros_on([2], &backend),
                ]
            } else {
                vec![]
            };

            comm.gather(&tensor, &mut output, 1, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP gather with mismatched peer tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_gather_zero_numel_mismatched_peer_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();
            let tensor = if rank == 1 {
                Tensor::<f32, _>::zeros_on([0], &backend)
            } else {
                Tensor::zeros_on([1], &backend)
            };
            let mut output = if rank == 1 {
                vec![
                    Tensor::zeros_on([0], &backend),
                    Tensor::zeros_on([0], &backend),
                ]
            } else {
                vec![]
            };

            comm.gather(&tensor, &mut output, 1, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP gather zero-numel with mismatched peer tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_scatter() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);

    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        let handle = thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = Tensor::zeros_on([1], &backend);
            let input = if rank == 0 {
                vec![
                    Tensor::from_slice_on([1], &[100.0], &backend),
                    Tensor::from_slice_on([1], &[200.0], &backend),
                ]
            } else {
                vec![]
            };

            comm.scatter(&mut tensor, &input, 0, &backend);

            assert_eq!(tensor.as_slice()[0], (rank + 1) as f32 * 100.0);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_scatter_mismatched_target_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::zeros_on([2], &backend)
            } else {
                Tensor::zeros_on([1], &backend)
            };
            let input = if rank == 0 {
                vec![
                    Tensor::from_slice_on([2], &[100.0f32, 101.0], &backend),
                    Tensor::from_slice_on([2], &[200.0f32, 201.0], &backend),
                ]
            } else {
                vec![]
            };

            comm.scatter(&mut tensor, &input, 0, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP scatter with mismatched target tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_scatter_zero_numel_mismatched_target_numel_panics() {
    let world_size = 2;
    let addresses = get_free_ports(world_size);
    let mut handles = vec![];

    for rank in 0..world_size {
        let addrs = addresses.clone();
        handles.push(thread::spawn(move || {
            let mesh = TcpMesh::new(rank, world_size, &addrs);
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::<f32, _>::zeros_on([0], &backend)
            } else {
                Tensor::zeros_on([1], &backend)
            };
            let input = if rank == 0 {
                vec![
                    Tensor::zeros_on([0], &backend),
                    Tensor::zeros_on([0], &backend),
                ]
            } else {
                vec![]
            };

            comm.scatter(&mut tensor, &input, 0, &backend);
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP scatter zero-numel with mismatched target tensor numel should panic on at least one rank",
    );
}

#[test]
#[should_panic(expected = "all_gather output numel mismatch")]
fn test_tcp_all_gather_mismatched_output_numel_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.all_gather(&tensor, &mut output, &backend);
}

#[test]
#[should_panic(expected = "all_gather output length mismatch")]
fn test_tcp_all_gather_zero_numel_output_len_mismatch_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.all_gather(&tensor, &mut output, &backend);
}

#[test]
#[should_panic(expected = "all_gather output numel mismatch")]
fn test_tcp_all_gather_zero_numel_output_numel_mismatch_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.all_gather(&tensor, &mut output, &backend);
}

#[test]
#[should_panic(expected = "scatter input numel mismatch")]
fn test_tcp_scatter_mismatched_input_numel_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let mut tensor = Tensor::zeros_on([2], &backend);
    let input = vec![Tensor::from_slice_on([1], &[3.0f32], &backend)];
    comm.scatter(&mut tensor, &input, 0, &backend);
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_broadcast_root_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    comm.broadcast(&mut tensor, 1, &backend);
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_reduce_root_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend);
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_gather_root_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([1], &[1.0f32], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 1, &backend);
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_scatter_root_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::zeros_on([1], &backend);
    let input = vec![Tensor::from_slice_on([1], &[1.0f32], &backend)];
    comm.scatter(&mut tensor, &input, 1, &backend);
}

#[test]
#[should_panic(expected = "gather output length mismatch on root")]
fn test_tcp_gather_zero_numel_output_len_mismatch_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.gather(&tensor, &mut output, 0, &backend);
}

#[test]
#[should_panic(expected = "gather output numel mismatch")]
fn test_tcp_gather_mismatched_output_numel_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 0, &backend);
}

#[test]
#[should_panic(expected = "gather output numel mismatch")]
fn test_tcp_gather_zero_numel_output_numel_mismatch_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let mut output = vec![Tensor::zeros_on([1], &backend)];
    comm.gather(&tensor, &mut output, 0, &backend);
}

#[test]
#[should_panic(expected = "scatter input length mismatch on root")]
fn test_tcp_scatter_zero_numel_input_len_mismatch_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let input: Vec<Tensor<f32, SequentialBackend>> = vec![];
    comm.scatter(&mut tensor, &input, 0, &backend);
}

#[test]
#[should_panic(expected = "scatter input numel mismatch")]
fn test_tcp_scatter_zero_numel_input_numel_mismatch_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend);
    let input = vec![Tensor::zeros_on([1], &backend)];
    comm.scatter(&mut tensor, &input, 0, &backend);
}

#[test]
#[should_panic(expected = "send peer must differ from local rank")]
fn test_tcp_mesh_send_self_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    mesh.send(0, &[1u8]);
}

#[test]
#[should_panic(expected = "recv peer must differ from local rank")]
fn test_tcp_mesh_recv_self_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let mut byte = [0u8; 1];
    mesh.recv(0, &mut byte);
}

#[test]
#[should_panic(expected = "send peer out of bounds")]
fn test_tcp_mesh_send_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    mesh.send(1, &[1u8]);
}

#[test]
#[should_panic(expected = "recv peer out of bounds")]
fn test_tcp_mesh_recv_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let mesh = TcpMesh::new(0, 1, &addresses);
    let mut byte = [0u8; 1];
    mesh.recv(1, &mut byte);
}

#[test]
#[should_panic(expected = "rank must be less than world size")]
fn test_tcp_mesh_new_rank_out_of_bounds_panics() {
    let addresses = get_free_ports(1);
    let _mesh = TcpMesh::new(1, 1, &addresses);
}

#[test]
#[should_panic(expected = "world size must be > 0")]
fn test_tcp_mesh_new_zero_world_size_panics() {
    let addresses: Vec<std::net::SocketAddr> = vec![];
    let _mesh = TcpMesh::new(0, 0, &addresses);
}

#[test]
#[should_panic(expected = "addresses list length must match world size")]
fn test_tcp_mesh_new_addresses_len_mismatch_panics() {
    let addresses = get_free_ports(1);
    let _mesh = TcpMesh::new(0, 2, &addresses);
}

// ── all_reduce with Max / Min / Product reduce ops ──
//
// world_size = 3, rank r contributes [r+1, r+2] -> ranks [1,2], [2,3], [3,4].
// Only Sum was previously exercised; these cover the other ReduceOpTag tags.

#[test]
fn test_local_all_reduce_max() {
    let communicators = LocalCommunicator::create_cluster(3);
    let mut handles = vec![];
    for comm in communicators {
        handles.push(thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
            comm.all_reduce::<f32, _, Max>(&mut tensor, &backend);
            assert_eq!(tensor.as_slice(), &[3.0, 4.0]);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_all_reduce_min() {
    let communicators = LocalCommunicator::create_cluster(3);
    let mut handles = vec![];
    for comm in communicators {
        handles.push(thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
            comm.all_reduce::<f32, _, Min>(&mut tensor, &backend);
            assert_eq!(tensor.as_slice(), &[1.0, 2.0]);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_all_reduce_product() {
    let communicators = LocalCommunicator::create_cluster(3);
    let mut handles = vec![];
    for comm in communicators {
        handles.push(thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend);
            comm.all_reduce::<f32, _, Product>(&mut tensor, &backend);
            // [1*2*3, 2*3*4] = [6, 24]
            assert_eq!(tensor.as_slice(), &[6.0, 24.0]);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_local_reduce_product_to_root() {
    // `reduce` (not all_reduce) leaves the product only on the root rank.
    let communicators = LocalCommunicator::create_cluster(3);
    let mut handles = vec![];
    for comm in communicators {
        handles.push(thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank();
            let mut tensor =
                Tensor::from_slice_on([2], &[rank as f32 + 1.0, rank as f32 + 2.0], &backend);
            comm.reduce::<f32, _, Product>(&mut tensor, 0, &backend);
            if rank == 0 {
                assert_eq!(tensor.as_slice(), &[6.0, 24.0]);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
