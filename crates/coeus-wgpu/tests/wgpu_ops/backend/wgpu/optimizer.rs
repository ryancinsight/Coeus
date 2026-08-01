// On-device optimizer step parity against the CPU reference.
//
// Each provider-owned stateful update (`sgd`, `adam`, `rmsprop`, `adamw`, `adagrad`)
// runs the same step on `WgpuBackend` and `SequentialBackend` with identical
// inputs and asserts element-wise agreement on the updated parameter and
// optimizer state. The updates are element-wise (no cross-element reduction),
// so the device result matches the CPU `f32` arithmetic within tight roundoff.

use coeus_core::SequentialBackend;
use coeus_ops::OptimizerOps;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

/// Element-wise fused-update tolerance: device f32 vs CPU f32 over the same
/// straight-line arithmetic (mul/add/sqrt/div, no reduction reorder).
const TOL: f32 = 1e-5;

fn assert_close(label: &str, gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (g - c).abs() < TOL,
            "{label} mismatch at {i}: GPU={g}, CPU={c}",
        );
    }
}

const SHAPE: [usize; 2] = [2, 3];
const P: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const G: [f32; 6] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6];
const S1: [f32; 6] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06];
const S2: [f32; 6] = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16];

fn pair<const N: usize>(
    seq: &SequentialBackend,
    wgpu: &WgpuBackend,
    data: &[f32; N],
) -> (Tensor<f32, SequentialBackend>, Tensor<f32, WgpuBackend>) {
    let cpu = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), data);
    let gpu = cpu.to_backend_on(seq, wgpu);
    (cpu, gpu)
}

#[test]
fn test_wgpu_sgd_step() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (mut p_c, mut p_g) = pair(&seq, &wgpu, &P);
    let (g_c, g_g) = pair(&seq, &wgpu, &G);
    let (mut vel_c, mut vel_g) = pair(&seq, &wgpu, &S1);
    let (lr, momentum) = (0.05f32, 0.9f32);

    {
        let (p, pl) = p_c.storage_mut_and_layout();
        let (vel, vl) = vel_c.storage_mut_and_layout();
        seq.sgd_step(p, pl, g_c.storage(), g_c.layout(), vel, vl, lr, momentum)
            .expect("CPU SGD step");
    }
    {
        let (p, pl) = p_g.storage_mut_and_layout();
        let (vel, vl) = vel_g.storage_mut_and_layout();
        wgpu.sgd_step(p, pl, g_g.storage(), g_g.layout(), vel, vl, lr, momentum)
            .expect("WGPU SGD step");
    }
    assert_close(
        "sgd_p",
        p_g.to_backend_on(&wgpu, &seq).as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "sgd_velocity",
        vel_g.to_backend_on(&wgpu, &seq).as_slice(),
        vel_c.as_slice(),
    );
}

#[test]
fn test_wgpu_sgd_ranks_zero_through_eight() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    for rank in 0..=8 {
        let shape = vec![1; rank];
        let mut p_c = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[2.0]);
        let g_c = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[1.0]);
        let mut v_c = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &[0.0]);
        let mut p_g = p_c.to_backend_on(&seq, &wgpu);
        let g_g = g_c.to_backend_on(&seq, &wgpu);
        let mut v_g = v_c.to_backend_on(&seq, &wgpu);
        let pl = p_c.layout().clone();
        let gl = g_c.layout().clone();
        let vl = v_c.layout().clone();

        seq.sgd_step(
            p_c.storage_mut(),
            &pl,
            g_c.storage(),
            &gl,
            v_c.storage_mut(),
            &vl,
            0.1,
            0.0,
        )
        .unwrap_or_else(|error| panic!("rank-{rank} CPU SGD failed: {error}"));
        wgpu.sgd_step(
            p_g.storage_mut(),
            &pl,
            g_g.storage(),
            &gl,
            v_g.storage_mut(),
            &vl,
            0.1,
            0.0,
        )
        .unwrap_or_else(|error| panic!("rank-{rank} WGPU SGD failed: {error}"));

        assert_close(
            &format!("rank-{rank} parameter"),
            p_g.to_backend_on(&wgpu, &seq).as_slice(),
            p_c.as_slice(),
        );
        assert_close(
            &format!("rank-{rank} velocity"),
            v_g.to_backend_on(&wgpu, &seq).as_slice(),
            v_c.as_slice(),
        );
    }
}

#[test]
fn test_wgpu_adam_step() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (mut p_c, mut p_g) = pair(&seq, &wgpu, &P);
    let (g_c, g_g) = pair(&seq, &wgpu, &G);
    let (mut m_c, mut m_g) = pair(&seq, &wgpu, &S1);
    let (mut v_c, mut v_g) = pair(&seq, &wgpu, &S2);
    let (lr, beta1, beta2, eps, t) = (0.05f32, 0.9f32, 0.99f32, 1e-6f32, 3usize);

    {
        let (p, pl) = p_c.storage_mut_and_layout();
        let (m, ml) = m_c.storage_mut_and_layout();
        let (v, vl) = v_c.storage_mut_and_layout();
        seq.adam_step(
            p,
            pl,
            g_c.storage(),
            g_c.layout(),
            m,
            ml,
            v,
            vl,
            lr,
            beta1,
            beta2,
            eps,
            t,
        )
        .expect("CPU Adam step");
    }
    {
        let (p, pl) = p_g.storage_mut_and_layout();
        let (m, ml) = m_g.storage_mut_and_layout();
        let (v, vl) = v_g.storage_mut_and_layout();
        wgpu.adam_step(
            p,
            pl,
            g_g.storage(),
            g_g.layout(),
            m,
            ml,
            v,
            vl,
            lr,
            beta1,
            beta2,
            eps,
            t,
        )
        .expect("WGPU Adam step");
    }
    assert_close(
        "adam_p",
        p_g.to_backend_on(&wgpu, &seq).as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "adam_m",
        m_g.to_backend_on(&wgpu, &seq).as_slice(),
        m_c.as_slice(),
    );
    assert_close(
        "adam_v",
        v_g.to_backend_on(&wgpu, &seq).as_slice(),
        v_c.as_slice(),
    );
}

#[test]
fn test_wgpu_rmsprop_step() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (mut p_c, mut p_g) = pair(&seq, &wgpu, &P);
    let (g_c, g_g) = pair(&seq, &wgpu, &G);
    let (mut v_c, mut v_g) = pair(&seq, &wgpu, &S1);
    let (lr, alpha, eps) = (0.05f32, 0.99f32, 1e-6f32);

    {
        let (p, pl) = p_c.storage_mut_and_layout();
        let (v, vl) = v_c.storage_mut_and_layout();
        seq.rmsprop_step(p, pl, g_c.storage(), g_c.layout(), v, vl, lr, alpha, eps)
            .expect("CPU RMSProp step");
    }
    {
        let (p, pl) = p_g.storage_mut_and_layout();
        let (v, vl) = v_g.storage_mut_and_layout();
        wgpu.rmsprop_step(p, pl, g_g.storage(), g_g.layout(), v, vl, lr, alpha, eps)
            .expect("WGPU RMSProp step");
    }
    assert_close(
        "rmsprop_p",
        p_g.to_backend_on(&wgpu, &seq).as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "rmsprop_v",
        v_g.to_backend_on(&wgpu, &seq).as_slice(),
        v_c.as_slice(),
    );
}

#[test]
fn test_wgpu_adagrad_step() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (mut p_c, mut p_g) = pair(&seq, &wgpu, &P);
    let (g_c, g_g) = pair(&seq, &wgpu, &G);
    let (mut h_c, mut h_g) = pair(&seq, &wgpu, &S1);
    let (lr, eps) = (0.05f32, 1e-6f32);

    {
        let (p, pl) = p_c.storage_mut_and_layout();
        let (h, hl) = h_c.storage_mut_and_layout();
        seq.adagrad_step(p, pl, g_c.storage(), g_c.layout(), h, hl, lr, eps)
            .expect("CPU AdaGrad step");
    }
    {
        let (p, pl) = p_g.storage_mut_and_layout();
        let (h, hl) = h_g.storage_mut_and_layout();
        wgpu.adagrad_step(p, pl, g_g.storage(), g_g.layout(), h, hl, lr, eps)
            .expect("WGPU AdaGrad step");
    }
    assert_close(
        "adagrad_p",
        p_g.to_backend_on(&wgpu, &seq).as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "adagrad_history",
        h_g.to_backend_on(&wgpu, &seq).as_slice(),
        h_c.as_slice(),
    );
}

#[test]
fn test_wgpu_adamw_step() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (mut p_c, mut p_g) = pair(&seq, &wgpu, &P);
    let (g_c, g_g) = pair(&seq, &wgpu, &G);
    let (mut m_c, mut m_g) = pair(&seq, &wgpu, &S1);
    let (mut v_c, mut v_g) = pair(&seq, &wgpu, &S2);
    let (lr, beta1, beta2, eps, wd, t) = (0.05f32, 0.9f32, 0.99f32, 1e-6f32, 0.02f32, 3usize);

    {
        let (p, pl) = p_c.storage_mut_and_layout();
        let (m, ml) = m_c.storage_mut_and_layout();
        let (v, vl) = v_c.storage_mut_and_layout();
        seq.adamw_step(
            p,
            pl,
            g_c.storage(),
            g_c.layout(),
            m,
            ml,
            v,
            vl,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            t,
        )
        .expect("CPU AdamW step");
    }
    {
        let (p, pl) = p_g.storage_mut_and_layout();
        let (m, ml) = m_g.storage_mut_and_layout();
        let (v, vl) = v_g.storage_mut_and_layout();
        wgpu.adamw_step(
            p,
            pl,
            g_g.storage(),
            g_g.layout(),
            m,
            ml,
            v,
            vl,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            t,
        )
        .expect("WGPU AdamW step");
    }
    assert_close(
        "adamw_p",
        p_g.to_backend_on(&wgpu, &seq).as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "adamw_m",
        m_g.to_backend_on(&wgpu, &seq).as_slice(),
        m_c.as_slice(),
    );
    assert_close(
        "adamw_v",
        v_g.to_backend_on(&wgpu, &seq).as_slice(),
        v_c.as_slice(),
    );
}
