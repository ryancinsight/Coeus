//! TCP collective invalid-input and panic contracts.

use super::super::super::support::assert_any_thread_panicked;
use super::super::super::support::loopback_meshes;
use super::super::super::support::single_rank_tcp_mesh;
use super::super::super::support::spawn_maybe_panicking;
use coeus_core::SequentialBackend;
use coeus_dist::Communicator;
use coeus_dist::Sum;
use coeus_dist::TcpCommunicator;
use coeus_tensor::Tensor;

#[test]
fn test_tcp_all_reduce_mismatched_numel_panics() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend).expect("construct tensor")
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend).expect("construct tensor")
            };
            drop(comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend));
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
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor")
            } else {
                Tensor::zeros_on([1], &backend).expect("construct tensor")
            };
            drop(comm.all_reduce::<f32, _, Sum>(&mut tensor, &backend));
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP all_reduce zero-numel with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_broadcast_mismatched_numel_panics() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[10.0f32, 20.0], &backend).expect("construct tensor")
            } else {
                Tensor::zeros_on([1], &backend).expect("construct tensor")
            };

            drop(comm.broadcast(&mut tensor, 0, &backend));
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP broadcast with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_all_gather_mismatched_peer_numel_panics() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend).expect("construct tensor")
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend).expect("construct tensor")
            };
            let mut output = if rank == 0 {
                vec![
                    Tensor::zeros_on([2], &backend).expect("construct tensor"),
                    Tensor::zeros_on([2], &backend).expect("construct tensor"),
                ]
            } else {
                vec![
                    Tensor::zeros_on([1], &backend).expect("construct tensor"),
                    Tensor::zeros_on([1], &backend).expect("construct tensor"),
                ]
            };

            drop(comm.all_gather(&tensor, &mut output, &backend));
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
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let tensor = if rank == 0 {
                Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor")
            } else {
                Tensor::zeros_on([1], &backend).expect("construct tensor")
            };
            let mut output = if rank == 0 {
                vec![
                    Tensor::zeros_on([0], &backend).expect("construct tensor"),
                    Tensor::zeros_on([0], &backend).expect("construct tensor"),
                ]
            } else {
                vec![
                    Tensor::zeros_on([1], &backend).expect("construct tensor"),
                    Tensor::zeros_on([1], &backend).expect("construct tensor"),
                ]
            };

            drop(comm.all_gather(&tensor, &mut output, &backend));
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP all_gather zero-numel with mismatched peer tensor numel should panic on at least one rank"
    );
}

#[test]
fn test_tcp_reduce_mismatched_numel_panics() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);

    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend).expect("construct tensor")
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend).expect("construct tensor")
            };

            drop(comm.reduce::<f32, _, Sum>(&mut tensor, 0, &backend));
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP reduce with mismatched tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_gather_mismatched_peer_numel_panics() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();
            let tensor = if rank == 1 {
                Tensor::from_slice_on([2], &[11.0f32, 12.0], &backend).expect("construct tensor")
            } else {
                Tensor::from_slice_on([1], &[3.0f32], &backend).expect("construct tensor")
            };
            let mut output = if rank == 1 {
                vec![
                    Tensor::zeros_on([2], &backend).expect("construct tensor"),
                    Tensor::zeros_on([2], &backend).expect("construct tensor"),
                ]
            } else {
                vec![]
            };

            drop(comm.gather(&tensor, &mut output, 1, &backend));
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
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();
            let tensor = if rank == 1 {
                Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor")
            } else {
                Tensor::zeros_on([1], &backend).expect("construct tensor")
            };
            let mut output = if rank == 1 {
                vec![
                    Tensor::zeros_on([0], &backend).expect("construct tensor"),
                    Tensor::zeros_on([0], &backend).expect("construct tensor"),
                ]
            } else {
                vec![]
            };

            drop(comm.gather(&tensor, &mut output, 1, &backend));
        }));
    }

    assert_any_thread_panicked(
        handles,
        "TCP gather zero-numel with mismatched peer tensor numel should panic on at least one rank",
    );
}

#[test]
fn test_tcp_scatter_mismatched_target_numel_panics() {
    let world_size = 2;
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::zeros_on([2], &backend).expect("construct tensor")
            } else {
                Tensor::zeros_on([1], &backend).expect("construct tensor")
            };
            let input = if rank == 0 {
                vec![
                    Tensor::from_slice_on([2], &[100.0f32, 101.0], &backend).expect("construct tensor"),
                    Tensor::from_slice_on([2], &[200.0f32, 201.0], &backend).expect("construct tensor"),
                ]
            } else {
                vec![]
            };

            drop(comm.scatter(&mut tensor, &input, 0, &backend));
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
    let meshes = loopback_meshes(world_size);
    let mut handles = vec![];

    for (rank, mesh) in meshes.into_iter().enumerate() {
        handles.push(spawn_maybe_panicking(move || {
            let comm = TcpCommunicator::new(mesh);
            let backend = SequentialBackend::new();

            let mut tensor = if rank == 0 {
                Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor")
            } else {
                Tensor::zeros_on([1], &backend).expect("construct tensor")
            };
            let input = if rank == 0 {
                vec![
                    Tensor::zeros_on([0], &backend).expect("construct tensor"),
                    Tensor::zeros_on([0], &backend).expect("construct tensor"),
                ]
            } else {
                vec![]
            };

            drop(comm.scatter(&mut tensor, &input, 0, &backend));
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
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend).expect("construct tensor");
    let mut output = vec![Tensor::zeros_on([1], &backend).expect("construct tensor")];
    drop(comm.all_gather(&tensor, &mut output, &backend));
}

#[test]
#[should_panic(expected = "all_gather output length mismatch")]
fn test_tcp_all_gather_zero_numel_output_len_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor");
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    drop(comm.all_gather(&tensor, &mut output, &backend));
}

#[test]
#[should_panic(expected = "all_gather output numel mismatch")]
fn test_tcp_all_gather_zero_numel_output_numel_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let tensor = Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor");
    let mut output = vec![Tensor::zeros_on([1], &backend).expect("construct tensor")];
    drop(comm.all_gather(&tensor, &mut output, &backend));
}

#[test]
#[should_panic(expected = "scatter input numel mismatch")]
fn test_tcp_scatter_mismatched_input_numel_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();

    let mut tensor = Tensor::zeros_on([2], &backend).expect("construct tensor");
    let input = vec![Tensor::from_slice_on([1], &[3.0f32], &backend).expect("construct tensor")];
    drop(comm.scatter(&mut tensor, &input, 0, &backend));
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_broadcast_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend).expect("construct tensor");
    drop(comm.broadcast(&mut tensor, 1, &backend));
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_reduce_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::from_slice_on([1], &[1.0f32], &backend).expect("construct tensor");
    drop(comm.reduce::<f32, _, Sum>(&mut tensor, 1, &backend));
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_gather_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([1], &[1.0f32], &backend).expect("construct tensor");
    let mut output = vec![Tensor::zeros_on([1], &backend).expect("construct tensor")];
    drop(comm.gather(&tensor, &mut output, 1, &backend));
}

#[test]
#[should_panic(expected = "collective root out of bounds")]
fn test_tcp_scatter_root_out_of_bounds_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::zeros_on([1], &backend).expect("construct tensor");
    let input = vec![Tensor::from_slice_on([1], &[1.0f32], &backend).expect("construct tensor")];
    drop(comm.scatter(&mut tensor, &input, 1, &backend));
}

#[test]
#[should_panic(expected = "gather output length mismatch on root")]
fn test_tcp_gather_zero_numel_output_len_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor");
    let mut output: Vec<Tensor<f32, SequentialBackend>> = vec![];
    drop(comm.gather(&tensor, &mut output, 0, &backend));
}

#[test]
#[should_panic(expected = "gather output numel mismatch")]
fn test_tcp_gather_mismatched_output_numel_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::from_slice_on([2], &[1.0f32, 2.0], &backend).expect("construct tensor");
    let mut output = vec![Tensor::zeros_on([1], &backend).expect("construct tensor")];
    drop(comm.gather(&tensor, &mut output, 0, &backend));
}

#[test]
#[should_panic(expected = "gather output numel mismatch")]
fn test_tcp_gather_zero_numel_output_numel_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let tensor = Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor");
    let mut output = vec![Tensor::zeros_on([1], &backend).expect("construct tensor")];
    drop(comm.gather(&tensor, &mut output, 0, &backend));
}

#[test]
#[should_panic(expected = "scatter input length mismatch on root")]
fn test_tcp_scatter_zero_numel_input_len_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor");
    let input: Vec<Tensor<f32, SequentialBackend>> = vec![];
    drop(comm.scatter(&mut tensor, &input, 0, &backend));
}

#[test]
#[should_panic(expected = "scatter input numel mismatch")]
fn test_tcp_scatter_zero_numel_input_numel_mismatch_panics() {
    let mesh = single_rank_tcp_mesh();
    let comm = TcpCommunicator::new(mesh);
    let backend = SequentialBackend::new();
    let mut tensor = Tensor::<f32, _>::zeros_on([0], &backend).expect("construct tensor");
    let input = vec![Tensor::zeros_on([1], &backend).expect("construct tensor")];
    drop(comm.scatter(&mut tensor, &input, 0, &backend));
}
