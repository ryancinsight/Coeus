use super::*;
use coeus_core::{ComputeBackend, Layout};
use coeus_ops::{ReductionOp, ReductionOps};
use eunomia::{Bf16, F16};

#[test]
fn test_cuda_parity_sum_axis0() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(
        &coeus_ops::sum_axis(&to_gpu(&x, &s, &c), 0, &c).expect("valid CUDA sum axis"),
        &c,
        &s,
    );
    assert_parity_tol("sum_axis0", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_sum_axis1() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(
        &coeus_ops::sum_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA sum axis"),
        &c,
        &s,
    );
    assert_parity_tol("sum_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_mean_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s).expect("valid CPU mean axis");
    let gpu = to_cpu(
        &coeus_ops::mean_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA mean axis"),
        &c,
        &s,
    );
    assert_parity_tol("mean_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_max_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s).expect("valid CPU max axis");
    let gpu = to_cpu(
        &coeus_ops::max_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA max axis"),
        &c,
        &s,
    );
    assert_parity_tol("max_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_min_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s).expect("valid CPU min axis");
    let gpu = to_cpu(
        &coeus_ops::min_axis(&to_gpu(&x, &s, &c), 0, &c).expect("valid CUDA min axis"),
        &c,
        &s,
    );
    assert_parity_tol("min_axis0", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_prod_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = vec![1.0f32, -2.0, 3.0, 4.0, 0.5, 6.0];
    let x = Tensor::from_slice(vec![2, 3], &data);
    let cpu = coeus_ops::prod_axis(&x, 1, &s).expect("valid CPU product axis");
    let gpu = to_cpu(
        &coeus_ops::prod_axis(&to_gpu(&x, &s, &c), 1, &c).expect("valid CUDA product axis"),
        &c,
        &s,
    );
    assert_parity_tol("prod_axis1", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_rank_one_sum() {
    let Some((s, c)) = backends() else {
        return;
    };
    let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
    let cpu = coeus_ops::sum_axis(&input, 0, &s).expect("valid CPU rank-one sum");
    let gpu = to_cpu(
        &coeus_ops::sum_axis(&to_gpu(&input, &s, &c), 0, &c).expect("valid CUDA rank-one sum"),
        &c,
        &s,
    );

    assert_eq!(gpu.shape(), &[1]);
    assert_parity_tol("rank-one-sum", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_rank_one_scan() {
    let Some((s, c)) = backends() else {
        return;
    };
    let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
    let cpu = coeus_ops::cumsum(&input, 0);
    let gpu = to_cpu(&coeus_ops::cumsum(&to_gpu(&input, &s, &c), 0), &c, &s);

    assert_eq!(gpu.shape(), &[4]);
    assert_parity_tol("rank-one-scan", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_reduction_rejects_unsupported_rank() {
    let Some((s, c)) = backends() else {
        return;
    };
    let input = Tensor::from_slice(vec![2, 2, 2], &[1.0f32; 8]);
    let gpu_input = to_gpu(&input, &s, &c);

    let error = match coeus_ops::sum_axis(&gpu_input, 1, &c) {
        Ok(_) => panic!("rank-three CUDA reduction unexpectedly succeeded"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        coeus_cuda::CudaBackendError::UnsupportedRank {
            operation: "reduction",
            rank: 3,
            max_rank: 2,
        }
    ));
}

#[test]
fn test_cuda_parity_cumulative_scans() {
    let Some((s, c)) = backends() else {
        return;
    };
    let data = (1..=6).map(|value| value as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![2, 3], &data);
    let gpu_input = to_gpu(&x, &s, &c);

    let cpu_prefix = coeus_ops::cumsum(&x, 1);
    let gpu_prefix = to_cpu(&coeus_ops::cumsum(&gpu_input, 1), &c, &s);
    assert_parity_tol(
        "cumsum-axis1",
        cpu_prefix.as_slice(),
        gpu_prefix.as_slice(),
        CUDA_TOL,
    );

    let cpu_suffix = coeus_ops::suffix_sum(&x, 0);
    let gpu_suffix = to_cpu(&coeus_ops::suffix_sum(&gpu_input, 0), &c, &s);
    assert_parity_tol(
        "suffix-sum-axis0",
        cpu_suffix.as_slice(),
        gpu_suffix.as_slice(),
        CUDA_TOL,
    );

    let cpu_prefix_product = coeus_ops::cumprod(&x, 1, &s);
    let gpu_prefix_product = to_cpu(&coeus_ops::cumprod(&gpu_input, 1, &c), &c, &s);
    assert_parity_tol(
        "cumprod-axis1",
        cpu_prefix_product.as_slice(),
        gpu_prefix_product.as_slice(),
        CUDA_TOL,
    );

    let cpu_suffix_product = coeus_ops::suffix_prod(&x, 0, &s);
    let gpu_suffix_product = to_cpu(&coeus_ops::suffix_prod(&gpu_input, 0, &c), &c, &s);
    assert_parity_tol(
        "suffix-prod-axis0",
        cpu_suffix_product.as_slice(),
        gpu_suffix_product.as_slice(),
        CUDA_TOL,
    );
}

// Matmul.

// ── half-precision identity proof ──────────────────────────────────────
//
// All six reduction identities (sum/prod/max/min/cumsum/cumprod) execute on
// device for both half formats. Inputs are small integers, so every result
// is exact in F16 and Bf16 and the claim is bitwise equality — a wrong
// identity literal would surface as garbage, not rounding.
macro_rules! test_halves_reduction_identities {
    ($ty:ty, $s:expr, $c:expr) => {{
        let flat: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
        let data: Vec<$ty> = flat.iter().map(|&v| <$ty>::from_f32(v)).collect();
        let in_layout = Layout::new(vec![3, 4].into());
        // `reduce` keeps the reduced axis as a size-1 dim rather than squeezing.
        let axis_layout = Layout::new(vec![3, 1].into());
        let cpu_in = Tensor::<$ty, SequentialBackend>::from_slice(vec![3, 4], &data);
        let gpu_in = cpu_in.to_backend_on(&$s, &$c);

        let run_reduce = |op: ReductionOp, expected: &[f32]| {
            let mut cpu_out = $s.allocate_zeroed::<$ty>(3);
            ReductionOps::reduce(
                &$s,
                op,
                cpu_in.storage(),
                cpu_in.layout(),
                1,
                &mut cpu_out,
                &axis_layout,
            )
            .expect("CPU halves reduce dispatch");
            let mut gpu_buf = $c.allocate_zeroed::<$ty>(3);
            ReductionOps::reduce(
                &$c,
                op,
                gpu_in.storage(),
                gpu_in.layout(),
                1,
                &mut gpu_buf,
                &axis_layout,
            )
            .expect("CUDA halves reduce dispatch");
            let expected: Vec<$ty> = expected.iter().map(|&v| <$ty>::from_f32(v)).collect();
            assert_eq!(
                Tensor::<$ty, SequentialBackend>::from_raw_parts(cpu_out, axis_layout.clone())
                    .as_slice(),
                expected.as_slice(),
                "{op:?} CPU halves reference",
            );
            assert_eq!(
                Tensor::<$ty, CudaBackend>::from_raw_parts(gpu_buf, axis_layout.clone())
                    .to_backend_on(&$c, &$s)
                    .as_slice(),
                expected.as_slice(),
                "{op:?} CUDA halves parity",
            );
        };
        run_reduce(ReductionOp::Sum, &[4.0, 5.0, 6.0]);
        run_reduce(ReductionOp::Max, &[1.0, 2.0, 2.0]);
        run_reduce(ReductionOp::Min, &[1.0, 1.0, 1.0]);

        // Product splits out of `reduce` (it needs `FloatElement`, which the
        // shared entry point deliberately does not require).
        {
            let mut cpu_out = $s.allocate_zeroed::<$ty>(3);
            ReductionOps::reduce_prod(
                &$s,
                cpu_in.storage(),
                cpu_in.layout(),
                1,
                &mut cpu_out,
                &axis_layout,
            )
            .expect("CPU halves prod dispatch");
            let mut gpu_buf = $c.allocate_zeroed::<$ty>(3);
            ReductionOps::reduce_prod(
                &$c,
                gpu_in.storage(),
                gpu_in.layout(),
                1,
                &mut gpu_buf,
                &axis_layout,
            )
            .expect("CUDA halves prod dispatch");
            let expected: Vec<$ty> = [1.0, 2.0, 4.0]
                .iter()
                .map(|&v| <$ty>::from_f32(v))
                .collect();
            assert_eq!(
                Tensor::<$ty, SequentialBackend>::from_raw_parts(cpu_out, axis_layout.clone())
                    .as_slice(),
                expected.as_slice(),
                "Prod CPU halves reference",
            );
            assert_eq!(
                Tensor::<$ty, CudaBackend>::from_raw_parts(gpu_buf, axis_layout.clone())
                    .to_backend_on(&$c, &$s)
                    .as_slice(),
                expected.as_slice(),
                "Prod CUDA halves parity",
            );
        }

        let run_scan = |cumsum: bool, expected: &[f32]| {
            let mut cpu_out = $s.allocate_zeroed::<$ty>(12);
            let mut gpu_buf = $c.allocate_zeroed::<$ty>(12);
            if cumsum {
                ReductionOps::cumsum(
                    &$s,
                    cpu_in.storage(),
                    cpu_in.layout(),
                    1,
                    &mut cpu_out,
                    &in_layout,
                )
                .expect("CPU halves cumsum dispatch");
                ReductionOps::cumsum(
                    &$c,
                    gpu_in.storage(),
                    gpu_in.layout(),
                    1,
                    &mut gpu_buf,
                    &in_layout,
                )
                .expect("CUDA halves cumsum dispatch");
            } else {
                ReductionOps::cumprod(
                    &$s,
                    cpu_in.storage(),
                    cpu_in.layout(),
                    1,
                    &mut cpu_out,
                    &in_layout,
                )
                .expect("CPU halves cumprod dispatch");
                ReductionOps::cumprod(
                    &$c,
                    gpu_in.storage(),
                    gpu_in.layout(),
                    1,
                    &mut gpu_buf,
                    &in_layout,
                )
                .expect("CUDA halves cumprod dispatch");
            }
            let expected: Vec<$ty> = expected.iter().map(|&v| <$ty>::from_f32(v)).collect();
            assert_eq!(
                Tensor::<$ty, SequentialBackend>::from_raw_parts(cpu_out, in_layout.clone())
                    .as_slice(),
                expected.as_slice(),
                "scan CPU halves reference",
            );
            assert_eq!(
                Tensor::<$ty, CudaBackend>::from_raw_parts(gpu_buf, in_layout.clone())
                    .to_backend_on(&$c, &$s)
                    .as_slice(),
                expected.as_slice(),
                "scan CUDA halves parity",
            );
        };
        run_scan(
            true,
            &[1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 4.0, 5.0, 2.0, 4.0, 5.0, 6.0],
        );
        run_scan(
            false,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
        );
    }};
}

#[test]
fn test_cuda_parity_halves_reduction_identities_f16() {
    let Some((s, c)) = backends() else {
        return;
    };
    test_halves_reduction_identities!(F16, s, c);
}

#[test]
fn test_cuda_parity_halves_reduction_identities_bf16() {
    let Some((s, c)) = backends() else {
        return;
    };
    test_halves_reduction_identities!(Bf16, s, c);
}
