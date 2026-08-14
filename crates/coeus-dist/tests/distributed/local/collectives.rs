//! Local in-process collective and gradient contracts.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_dist::synchronize_gradients;
use coeus_dist::Communicator;
use coeus_dist::LocalCommunicator;
use coeus_dist::Sum;
use coeus_tensor::Tensor;
use std::thread;

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
            synchronize_gradients(&mut params, &comm).expect("valid distributed gradient layout");

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
