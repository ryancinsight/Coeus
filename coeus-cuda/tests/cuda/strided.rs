use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_tensor::{Tensor, Transpose};

#[test]
fn test_cuda_strided_ops() {
    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        let seq = SequentialBackend::new();
        let cuda_b = CudaBackend::new();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![10.0f32, 20.0, 30.0];

        let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
        let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 1], &b_data);

        let a_seq_t = a_seq.transpose();

        let a_cuda_t = a_seq_t.to_backend_on(&seq, &cuda_b);
        let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);

        assert_eq!(a_cuda_t.shape(), &[3, 2]);
        assert_eq!(b_cuda.shape(), &[3, 1]);

        let c_cuda = coeus_ops::add(&a_cuda_t, &b_cuda, &cuda_b);
        let c_seq = c_cuda.to_backend_on(&cuda_b, &seq);

        let c_expected = coeus_ops::add(&a_seq_t, &b_seq, &seq);

        for (i, (&res, &exp)) in c_seq
            .as_slice()
            .iter()
            .zip(c_expected.as_slice().iter())
            .enumerate()
        {
            assert!(
                (res - exp).abs() < 1e-4f32,
                "Strided add mismatch at {}: {} vs {}",
                i,
                res,
                exp
            );
        }

        let u_cuda = coeus_ops::relu(&a_cuda_t, &cuda_b);
        let u_seq = u_cuda.to_backend_on(&cuda_b, &seq);
        let u_expected = coeus_ops::relu(&a_seq_t, &seq);

        for (i, (&res, &exp)) in u_seq
            .as_slice()
            .iter()
            .zip(u_expected.as_slice().iter())
            .enumerate()
        {
            assert!(
                (res - exp).abs() < 1e-4f32,
                "Strided relu mismatch at {}: {} vs {}",
                i,
                res,
                exp
            );
        }
    }
}
