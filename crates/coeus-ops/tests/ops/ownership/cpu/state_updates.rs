use super::accumulated_outputs::{
    attention_preserves_output_clones, convolution_preserves_output_clones,
    pooling_preserves_output_clones, windows_preserve_output_clones,
};
use super::optimizer::adam_preserves_all_state_clones;
use super::staggered::staggered_preserves_clones;
use coeus_core::{Float, MoiraiBackend, SequentialBackend};
use coeus_ops::{ConvOps, OptimizerOps, PoolOps, UnfoldFoldOps};

fn cpu_updates<T: Float>(one: T)
where
    SequentialBackend: OptimizerOps<T> + ConvOps<T> + PoolOps<T> + UnfoldFoldOps<T>,
    MoiraiBackend: OptimizerOps<T> + ConvOps<T> + PoolOps<T> + UnfoldFoldOps<T>,
{
    let sequential = SequentialBackend::new();
    let parallel = MoiraiBackend::new();
    adam_preserves_all_state_clones(&sequential, one);
    adam_preserves_all_state_clones(&parallel, one);
    convolution_preserves_output_clones(&sequential, one);
    convolution_preserves_output_clones(&parallel, one);
    pooling_preserves_output_clones(&sequential, one);
    pooling_preserves_output_clones(&parallel, one);
    windows_preserve_output_clones(&sequential, one);
    windows_preserve_output_clones(&parallel, one);
}

#[test]
fn cpu_updates_preserve_all_output_clones() {
    cpu_updates(1.0_f32);
    cpu_updates(1.0_f64);
    cpu_updates(eunomia::F16::from_bits(0x3c00));
    cpu_updates(eunomia::Bf16::from_bits(0x3f80));
    attention_preserves_output_clones(&SequentialBackend::new(), 1.0_f32);
    attention_preserves_output_clones(&SequentialBackend::new(), 1.0_f64);
    attention_preserves_output_clones(&MoiraiBackend::new(), 1.0_f32);
    attention_preserves_output_clones(&MoiraiBackend::new(), 1.0_f64);
}

#[test]
fn cpu_staggered_preserves_output_clones() {
    staggered_preserves_clones(&SequentialBackend::new(), 1.0_f32);
    staggered_preserves_clones(&SequentialBackend::new(), 1.0_f64);
    staggered_preserves_clones(&MoiraiBackend::new(), 1.0_f32);
    staggered_preserves_clones(&MoiraiBackend::new(), 1.0_f64);
}
