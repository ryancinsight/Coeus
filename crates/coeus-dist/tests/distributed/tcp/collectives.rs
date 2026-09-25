//! TCP loopback collective contracts.

use super::super::support::loopback_meshes;
use coeus_core::SequentialBackend;
use coeus_dist::Communicator;
use coeus_dist::Product;
use coeus_dist::Sum;
use coeus_dist::TcpCommunicator;
use coeus_tensor::Tensor;
use std::thread;

#[test]
fn test_tcp_all_reduce() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor =
                Tensor::from_slice_on([2], &[(rank + 1) as f32, (rank + 2) as f32], &backend);
            comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend)
                .unwrap();

            let data = tensor.as_slice();
            assert_eq!(data[0], 3.0);
            assert_eq!(data[1], 5.0);
            comm.shutdown();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_broadcast() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[10.0f32, 20.0], &backend)
            } else {
                Tensor::zeros_on([2], &backend)
            };

            comm.broadcast(&mut tensor, 0, &backend).unwrap();

            let data = tensor.as_slice();
            assert_eq!(data[0], 10.0);
            assert_eq!(data[1], 20.0);
            comm.shutdown();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_all_gather() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = Tensor::from_slice_on([1], &[(rank * 100) as f32], &backend);
            let mut output = vec![
                Tensor::zeros_on([1], &backend),
                Tensor::zeros_on([1], &backend),
            ];

            comm.all_gather(&tensor, &mut output, &backend).unwrap();

            assert_eq!(output[0].as_slice()[0], 0.0);
            assert_eq!(output[1].as_slice()[0], 100.0);
            comm.shutdown();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_barrier() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for mesh in meshes {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);

            comm.barrier().unwrap();
            comm.shutdown();
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
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor =
                Tensor::from_slice_on([2], &[(rank + 1) as f32, (rank + 2) as f32], &backend);
            comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend)
                .unwrap();

            if rank == 1 {
                let data = tensor.as_slice();
                assert_eq!(data[0], 3.0);
                assert_eq!(data[1], 5.0);
            }
            comm.shutdown();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

/// A peer contributing `i32::MAX` to an integer `Sum` reduce wraps on the
/// root instead of overflow-panicking it (`1_i32.wrapping_add(i32::MAX) ==
/// i32::MIN`), matching ADR 0075's wrapping-integer-reduction decision. The
/// peer value is not malformed protocol data — it is a legitimate, merely
/// extreme, tensor element a hostile or careless peer can hold.
#[test]
fn test_tcp_reduce_sum_wraps_on_integer_overflow_from_a_peer() {
    let meshes = loopback_meshes(2);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();
            let local = if rank == 0 { 1_i32 } else { i32::MAX };
            let mut tensor = Tensor::from_slice_on([1], &[local], &backend);
            comm.reduce::<i32, _, Sum>(&mut tensor, 0, &backend)
                .unwrap();
            if rank == 0 {
                assert_eq!(tensor.as_slice()[0], i32::MIN);
            }
            comm.shutdown();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

/// The `Product` counterpart: `2_i32.wrapping_mul(i32::MAX)` wraps instead of
/// panicking under overflow checks. See
/// [`test_tcp_reduce_sum_wraps_on_integer_overflow_from_a_peer`].
#[test]
fn test_tcp_reduce_product_wraps_on_integer_overflow_from_a_peer() {
    let meshes = loopback_meshes(2);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();
            let local = if rank == 0 { 2_i32 } else { i32::MAX };
            let mut tensor = Tensor::from_slice_on([1], &[local], &backend);
            comm.reduce::<i32, _, Product>(&mut tensor, 0, &backend)
                .unwrap();
            if rank == 0 {
                assert_eq!(tensor.as_slice()[0], 2_i32.wrapping_mul(i32::MAX));
            }
            comm.shutdown();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_gather() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
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

            comm.gather(&tensor, &mut output, 1, &backend).unwrap();

            if rank == 1 {
                assert_eq!(output[0].as_slice()[0], 0.0);
                assert_eq!(output[1].as_slice()[0], 100.0);
            }
            comm.shutdown();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_tcp_scatter() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
            let mut comm = TcpCommunicator::new(mesh);
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

            comm.scatter(&mut tensor, &input, 0, &backend).unwrap();

            assert_eq!(tensor.as_slice()[0], (rank + 1) as f32 * 100.0);
            comm.shutdown();
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}
