use super::*;
use coeus_core::{ComputeBackend, Layout};
use coeus_ops::ScalarPowerOps;

/// pow spans orders of magnitude, so the claim is a relative epsilon bound
/// rather than the absolute tolerance of the arithmetic tests.
macro_rules! test_pow_parity {
    ($ty:ty, $eps:expr, $s:expr, $c:expr) => {{
        let bases: Vec<$ty> = vec![0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0];
        let layout = Layout::new(vec![bases.len()].into());
        for exponent in [0.5 as $ty, 2.5 as $ty] {
            let cpu_in = Tensor::<$ty, SequentialBackend>::from_slice(vec![bases.len()], &bases);
            let gpu_in = cpu_in.to_backend_on(&$s, &$c);
            let mut cpu_out = $s.allocate_zeroed::<$ty>(bases.len());
            ScalarPowerOps::elementwise_pow_scalar(
                &$s,
                cpu_in.storage(),
                cpu_in.layout(),
                exponent,
                &mut cpu_out,
                &layout,
            )
            .expect("CPU pow dispatch");
            let mut gpu_buf = $c.allocate_zeroed::<$ty>(bases.len());
            ScalarPowerOps::elementwise_pow_scalar(
                &$c,
                gpu_in.storage(),
                gpu_in.layout(),
                exponent,
                &mut gpu_buf,
                &layout,
            )
            .expect("CUDA pow dispatch");
            let expected =
                Tensor::<$ty, SequentialBackend>::from_raw_parts(cpu_out, layout.clone())
                    .as_slice()
                    .to_vec();
            let actual = Tensor::<$ty, CudaBackend>::from_raw_parts(gpu_buf, layout.clone())
                .to_backend_on(&$c, &$s)
                .as_slice()
                .to_vec();
            for (i, (&c, &g)) in expected.iter().zip(actual.iter()).enumerate() {
                let bound = 64.0 * $eps * c.abs().max(1.0);
                assert!(
                    (c - g).abs() <= bound,
                    "pow({exponent})[{i}]: cpu={c} gpu={g} bound={bound:e}"
                );
            }
        }
    }};
}

#[test]
fn test_cuda_parity_pow_f32() {
    let Some((s, c)) = backends() else {
        return;
    };
    test_pow_parity!(f32, f32::EPSILON, s, c);
}

#[test]
fn test_cuda_parity_pow_f64() {
    let Some((s, c)) = backends() else {
        return;
    };
    test_pow_parity!(f64, f64::EPSILON, s, c);
}
