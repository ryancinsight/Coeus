//! Local reduction-operator contracts.

use coeus_core::SequentialBackend;
use coeus_dist::Communicator;
use coeus_dist::LocalCommunicator;
use coeus_dist::Max;
use coeus_dist::Min;
use coeus_dist::Product;
use coeus_tensor::Tensor;
use std::thread;

#[test]
fn test_local_all_reduce_max() {
    let communicators = LocalCommunicator::create_cluster(3);
    let mut handles = vec![];
    for comm in communicators {
        handles.push(thread::spawn(move || {
            let backend = SequentialBackend::new();
            let rank = comm.rank() as f32;
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend).expect("construct tensor");
            comm.all_reduce::<f32, _, Max>(&mut tensor, &backend).expect("collective operation");
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
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend).expect("construct tensor");
            comm.all_reduce::<f32, _, Min>(&mut tensor, &backend).expect("collective operation");
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
            let mut tensor = Tensor::from_slice_on([2], &[rank + 1.0, rank + 2.0], &backend).expect("construct tensor");
            comm.all_reduce::<f32, _, Product>(&mut tensor, &backend).expect("collective operation");
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
                Tensor::from_slice_on([2], &[rank as f32 + 1.0, rank as f32 + 2.0], &backend).expect("construct tensor");
            comm.reduce::<f32, _, Product>(&mut tensor, 0, &backend).expect("collective operation");
            if rank == 0 {
                assert_eq!(tensor.as_slice(), &[6.0, 24.0]);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
