use coeus_core::{ComputeBackend, Layout, SequentialBackend};
use coeus_ops::{ReductionOp, ReductionOps};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;
use eunomia::F16;

use super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_sum_axis0() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 0, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 0, &wgpu()).expect("valid WGPU sum axis"));
    assert_parity("sum_axis0", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sum_axis1() {
    let s = seq();
    let data = (0..12).map(|x| x as f32).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::sum_axis(&x, 1, &s).expect("valid CPU sum axis");
    let gpu = to_cpu(&coeus_ops::sum_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU sum axis"));
    assert_parity("sum_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mean_axis() {
    let s = seq();
    let data = (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>();
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::mean_axis(&x, 1, &s).expect("valid CPU mean axis");
    let gpu = to_cpu(&coeus_ops::mean_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU mean axis"));
    assert_parity("mean_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_max_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 2.0, 0.5, 7.0, 3.0, 5.0, 9.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::max_axis(&x, 1, &s).expect("valid CPU max axis");
    let gpu = to_cpu(&coeus_ops::max_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU max axis"));
    assert_parity("max_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_min_axis() {
    let s = seq();
    let data = vec![
        3.0f32, 1.0, 4.0, 1.5, 2.0, 8.0, 0.2, 0.5, 7.0, 3.0, 5.0, -1.0,
    ];
    let x = Tensor::from_slice(vec![3, 4], &data);
    let cpu = coeus_ops::min_axis(&x, 0, &s).expect("valid CPU min axis");
    let gpu = to_cpu(&coeus_ops::min_axis(&to_gpu(&x), 0, &wgpu()).expect("valid WGPU min axis"));
    assert_parity("min_axis0", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_prod_axis() {
    let s = seq();
    let data = vec![1.0f32, -2.0, 3.0, 4.0, 0.5, 6.0];
    let x = Tensor::from_slice(vec![2, 3], &data);
    let cpu = coeus_ops::prod_axis(&x, 1, &s).expect("valid CPU product axis");
    let gpu =
        to_cpu(&coeus_ops::prod_axis(&to_gpu(&x), 1, &wgpu()).expect("valid WGPU product axis"));
    assert_parity("prod_axis1", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_rank_one_sum() {
    let s = seq();
    let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
    let cpu = coeus_ops::sum_axis(&input, 0, &s).expect("valid CPU rank-one sum");
    let gpu =
        to_cpu(&coeus_ops::sum_axis(&to_gpu(&input), 0, &wgpu()).expect("valid WGPU rank-one sum"));

    assert_eq!(gpu.shape(), &[1]);
    assert_parity("rank-one-sum", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_rank_one_scan() {
    let input = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]);
    let cpu = coeus_ops::cumsum(&input, 0);
    let gpu = to_cpu(&coeus_ops::cumsum(&to_gpu(&input), 0));

    assert_eq!(gpu.shape(), &[4]);
    assert_parity("rank-one-scan", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_reduction_rejects_unsupported_rank() {
    let input = Tensor::from_slice(vec![2, 2, 2], &[1.0f32; 8]);
    let gpu_input = to_gpu(&input);

    let error = match coeus_ops::sum_axis(&gpu_input, 1, &wgpu()) {
        Ok(_) => panic!("rank-three WGPU reduction unexpectedly succeeded"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        coeus_wgpu::WgpuBackendError::Validation(coeus_core::BackendError::UnsupportedRank {
            operation: "reduction",
            rank: 3,
            max_rank: 2,
        })
    ));
}

// ── half-precision identity proof ──────────────────────────────────────
//
// All six reduction identities (sum/prod/max/min/cumsum/cumprod) execute on
// device for F16. Inputs are small integers, so every result is exact and
// the claim is bitwise equality — a wrong identity literal would surface as
// garbage, not rounding. Bf16 is absent by WGSL design: the shading language
// has no bf16 type, so no `DialectScalar<Wgsl>` can exist for it.
macro_rules! test_f16_reduction_identities {
    ($s:expr, $c:expr) => {{
        let flat: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
        let data: Vec<F16> = flat.iter().map(|&v| F16::from_f32(v)).collect();
        let in_layout = Layout::new(vec![3, 4].into());
        // `reduce` keeps the reduced axis as a size-1 dim rather than squeezing.
        let axis_layout = Layout::new(vec![3, 1].into());
        let cpu_in = Tensor::<F16, SequentialBackend>::from_slice(vec![3, 4], &data);
        let gpu_in = cpu_in.to_backend_on(&$s, &$c);

        let run_reduce = |op: ReductionOp, expected: &[f32]| {
            let mut cpu_out = $s.allocate_zeroed::<F16>(3);
            ReductionOps::reduce(
                &$s,
                op,
                cpu_in.storage(),
                cpu_in.layout(),
                1,
                &mut cpu_out,
                &axis_layout,
            )
            .expect("CPU F16 reduce dispatch");
            let mut gpu_buf = $c.allocate_zeroed::<F16>(3);
            ReductionOps::reduce(
                &$c,
                op,
                gpu_in.storage(),
                gpu_in.layout(),
                1,
                &mut gpu_buf,
                &axis_layout,
            )
            .expect("WGPU F16 reduce dispatch");
            let expected: Vec<F16> = expected.iter().map(|&v| F16::from_f32(v)).collect();
            assert_eq!(
                Tensor::<F16, SequentialBackend>::from_raw_parts(cpu_out, axis_layout.clone())
                    .as_slice(),
                expected.as_slice(),
                "{op:?} CPU F16 reference",
            );
            assert_eq!(
                Tensor::<F16, WgpuBackend>::from_raw_parts(gpu_buf, axis_layout.clone())
                    .to_backend_on(&$c, &$s)
                    .as_slice(),
                expected.as_slice(),
                "{op:?} WGPU F16 parity",
            );
        };
        run_reduce(ReductionOp::Sum, &[4.0, 5.0, 6.0]);
        run_reduce(ReductionOp::Max, &[1.0, 2.0, 2.0]);
        run_reduce(ReductionOp::Min, &[1.0, 1.0, 1.0]);

        // Product splits out of `reduce` (it needs `FloatElement`, which the
        // shared entry point deliberately does not require).
        {
            let mut cpu_out = $s.allocate_zeroed::<F16>(3);
            ReductionOps::reduce_prod(
                &$s,
                cpu_in.storage(),
                cpu_in.layout(),
                1,
                &mut cpu_out,
                &axis_layout,
            )
            .expect("CPU F16 prod dispatch");
            let mut gpu_buf = $c.allocate_zeroed::<F16>(3);
            ReductionOps::reduce_prod(
                &$c,
                gpu_in.storage(),
                gpu_in.layout(),
                1,
                &mut gpu_buf,
                &axis_layout,
            )
            .expect("WGPU F16 prod dispatch");
            let expected: Vec<F16> = [1.0, 2.0, 4.0].iter().map(|&v| F16::from_f32(v)).collect();
            assert_eq!(
                Tensor::<F16, SequentialBackend>::from_raw_parts(cpu_out, axis_layout.clone())
                    .as_slice(),
                expected.as_slice(),
                "Prod CPU F16 reference",
            );
            assert_eq!(
                Tensor::<F16, WgpuBackend>::from_raw_parts(gpu_buf, axis_layout.clone())
                    .to_backend_on(&$c, &$s)
                    .as_slice(),
                expected.as_slice(),
                "Prod WGPU F16 parity",
            );
        }

        let run_scan = |cumsum: bool, expected: &[f32]| {
            let mut cpu_out = $s.allocate_zeroed::<F16>(12);
            let mut gpu_buf = $c.allocate_zeroed::<F16>(12);
            if cumsum {
                ReductionOps::cumsum(
                    &$s,
                    cpu_in.storage(),
                    cpu_in.layout(),
                    1,
                    &mut cpu_out,
                    &in_layout,
                )
                .expect("CPU F16 cumsum dispatch");
                ReductionOps::cumsum(
                    &$c,
                    gpu_in.storage(),
                    gpu_in.layout(),
                    1,
                    &mut gpu_buf,
                    &in_layout,
                )
                .expect("WGPU F16 cumsum dispatch");
            } else {
                ReductionOps::cumprod(
                    &$s,
                    cpu_in.storage(),
                    cpu_in.layout(),
                    1,
                    &mut cpu_out,
                    &in_layout,
                )
                .expect("CPU F16 cumprod dispatch");
                ReductionOps::cumprod(
                    &$c,
                    gpu_in.storage(),
                    gpu_in.layout(),
                    1,
                    &mut gpu_buf,
                    &in_layout,
                )
                .expect("WGPU F16 cumprod dispatch");
            }
            let expected: Vec<F16> = expected.iter().map(|&v| F16::from_f32(v)).collect();
            assert_eq!(
                Tensor::<F16, SequentialBackend>::from_raw_parts(cpu_out, in_layout.clone())
                    .as_slice(),
                expected.as_slice(),
                "scan CPU F16 reference",
            );
            assert_eq!(
                Tensor::<F16, WgpuBackend>::from_raw_parts(gpu_buf, in_layout.clone())
                    .to_backend_on(&$c, &$s)
                    .as_slice(),
                expected.as_slice(),
                "scan WGPU F16 parity",
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
fn test_wgpu_parity_f16_reduction_identities() {
    if !crate::availability::device_supports_f16("coeus-wgpu-f16-reduction-test") {
        return;
    }
    let s = seq();
    let c = wgpu();
    test_f16_reduction_identities!(s, c);
}
