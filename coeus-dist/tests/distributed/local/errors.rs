//! Local in-process invalid-input and panic contracts.

use coeus_core::SequentialBackend;
use coeus_dist::Communicator;
use coeus_dist::LocalCommunicator;
use coeus_dist::Sum;
use coeus_tensor::Tensor;

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
