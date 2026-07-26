//! TCP loopback collective contracts.

use super::super::support::loopback_meshes;
use coeus_core::SequentialBackend;
use coeus_dist::Communicator;
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
fn test_tcp_broadcast() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
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
fn test_tcp_all_gather() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
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
fn test_tcp_barrier() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for mesh in meshes {
        let handle = thread::spawn(move || {
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
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
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
fn test_tcp_gather() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
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
fn test_tcp_scatter() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        let handle = thread::spawn(move || {
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
