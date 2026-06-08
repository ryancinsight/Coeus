use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_cuda::CudaBackend;

#[test]
fn test_cuda_max_pool2d() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_data);
    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);

    let mut output_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 2, 2]);
    let mut output_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 2, 2]);

    let (out_seq_storage, out_seq_layout) = output_seq.storage_mut_and_layout();
    coeus_ops::BackendOps::max_pool2d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        out_seq_storage,
        out_seq_layout,
    );

    let (out_cuda_storage, out_cuda_layout) = output_cuda.storage_mut_and_layout();
    coeus_ops::BackendOps::max_pool2d(
        &cuda_b,
        input_cuda.storage(),
        input_cuda.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        out_cuda_storage,
        out_cuda_layout,
    );

    let output_cuda_on_cpu = output_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(output_cuda_on_cpu.as_slice(), output_seq.as_slice());
    }

    // Now test backward
    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &grad_out_data);
    let grad_out_cuda = grad_out_seq.to_backend_on(&seq, &cuda_b);

    let mut grad_in_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 4, 4]);
    let mut grad_in_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 4, 4]);

    let (gi_seq_storage, gi_seq_layout) = grad_in_seq.storage_mut_and_layout();
    coeus_ops::BackendOps::max_pool2d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        input_seq.storage(),
        input_seq.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        gi_seq_storage,
        gi_seq_layout,
    );

    let (gi_cuda_storage, gi_cuda_layout) = grad_in_cuda.storage_mut_and_layout();
    coeus_ops::BackendOps::max_pool2d_backward(
        &cuda_b,
        grad_out_cuda.storage(),
        grad_out_cuda.layout(),
        input_cuda.storage(),
        input_cuda.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        gi_cuda_storage,
        gi_cuda_layout,
    );

    let grad_in_cuda_on_cpu = grad_in_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(grad_in_cuda_on_cpu.as_slice(), grad_in_seq.as_slice());
    }
}

#[test]
fn test_cuda_avg_pool2d() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_data);
    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);

    let mut output_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 2, 2]);
    let mut output_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 2, 2]);

    let (out_seq_storage, out_seq_layout) = output_seq.storage_mut_and_layout();
    coeus_ops::BackendOps::avg_pool2d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        out_seq_storage,
        out_seq_layout,
    );

    let (out_cuda_storage, out_cuda_layout) = output_cuda.storage_mut_and_layout();
    coeus_ops::BackendOps::avg_pool2d(
        &cuda_b,
        input_cuda.storage(),
        input_cuda.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        out_cuda_storage,
        out_cuda_layout,
    );

    let output_cuda_on_cpu = output_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(output_cuda_on_cpu.as_slice(), output_seq.as_slice());
    }

    // Now test backward
    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &grad_out_data);
    let grad_out_cuda = grad_out_seq.to_backend_on(&seq, &cuda_b);

    let mut grad_in_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 4, 4]);
    let mut grad_in_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 4, 4]);

    let (gi_seq_storage, gi_seq_layout) = grad_in_seq.storage_mut_and_layout();
    coeus_ops::BackendOps::avg_pool2d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        gi_seq_storage,
        gi_seq_layout,
    );

    let (gi_cuda_storage, gi_cuda_layout) = grad_in_cuda.storage_mut_and_layout();
    coeus_ops::BackendOps::avg_pool2d_backward(
        &cuda_b,
        grad_out_cuda.storage(),
        grad_out_cuda.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        gi_cuda_storage,
        gi_cuda_layout,
    );

    let grad_in_cuda_on_cpu = grad_in_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(grad_in_cuda_on_cpu.as_slice(), grad_in_seq.as_slice());
    }
}
