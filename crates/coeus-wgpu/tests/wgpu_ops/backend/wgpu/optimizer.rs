// On-device optimizer step parity against the CPU reference.
//
// Each fused optimizer kernel (`sgd`, `adam`, `rmsprop`, `adamw`, `adagrad`)
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
    let cpu = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), data)
        .expect("construct tensor");
    let gpu = cpu.to_backend_on(seq, wgpu).expect("transfer tensor");
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
        let (p, pl) = p_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (vel, vl) = vel_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        seq.sgd_step(p, pl, g_c.storage(), g_c.layout(), vel, vl, lr, momentum)
            .expect("execute CPU SGD");
    }
    {
        let (p, pl) = p_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (vel, vl) = vel_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        wgpu.sgd_step(p, pl, g_g.storage(), g_g.layout(), vel, vl, lr, momentum)
            .expect("execute WGPU SGD");
    }
    assert_close(
        "sgd_p",
        p_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "sgd_velocity",
        vel_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        vel_c.as_slice(),
    );
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
        let (p, pl) = p_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (m, ml) = m_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (v, vl) = v_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
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
        .expect("execute CPU Adam");
    }
    {
        let (p, pl) = p_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (m, ml) = m_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (v, vl) = v_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
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
        .expect("execute WGPU Adam");
    }
    assert_close(
        "adam_p",
        p_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "adam_m",
        m_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        m_c.as_slice(),
    );
    assert_close(
        "adam_v",
        v_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
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
        let (p, pl) = p_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (v, vl) = v_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        seq.rmsprop_step(p, pl, g_c.storage(), g_c.layout(), v, vl, lr, alpha, eps)
            .expect("execute CPU RMSProp");
    }
    {
        let (p, pl) = p_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (v, vl) = v_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        wgpu.rmsprop_step(p, pl, g_g.storage(), g_g.layout(), v, vl, lr, alpha, eps)
            .expect("execute WGPU RMSProp");
    }
    assert_close(
        "rmsprop_p",
        p_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "rmsprop_v",
        v_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
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
        let (p, pl) = p_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (h, hl) = h_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        seq.adagrad_step(p, pl, g_c.storage(), g_c.layout(), h, hl, lr, eps)
            .expect("execute CPU Adagrad");
    }
    {
        let (p, pl) = p_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (h, hl) = h_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        wgpu.adagrad_step(p, pl, g_g.storage(), g_g.layout(), h, hl, lr, eps)
            .expect("execute WGPU Adagrad");
    }
    assert_close(
        "adagrad_p",
        p_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "adagrad_history",
        h_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
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
        let (p, pl) = p_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (m, ml) = m_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (v, vl) = v_c
            .storage_mut_and_layout()
            .expect("access tensor storage");
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
        .expect("execute CPU AdamW");
    }
    {
        let (p, pl) = p_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (m, ml) = m_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
        let (v, vl) = v_g
            .storage_mut_and_layout()
            .expect("access tensor storage");
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
        .expect("execute WGPU AdamW");
    }
    assert_close(
        "adamw_p",
        p_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        p_c.as_slice(),
    );
    assert_close(
        "adamw_m",
        m_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        m_c.as_slice(),
    );
    assert_close(
        "adamw_v",
        v_g.to_backend_on(&wgpu, &seq).expect("transfer tensor").as_slice(),
        v_c.as_slice(),
    );
}
