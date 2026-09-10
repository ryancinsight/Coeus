//! Cached topological sort: fixtures shared by the two case families.

use super::*;
use crate::autodiff_cache::CacheConfig;
use crate::grad_buffer::GradBuffer;
use crate::var::Var;
use coeus_core::{BackendError, MoiraiBackend};
use coeus_tensor::Tensor;

struct TestNode {
    name: &'static str,
    output_grad: Arc<GradBuffer<f32, MoiraiBackend>>,
    inputs: Vec<Var<f32, MoiraiBackend>>,
}

struct BudgetConfig {
    metadata_memory: usize,
    plan_memory: usize,
    max_entries: usize,
}

impl CacheConfig for BudgetConfig {
    fn max_cache_entries(&self) -> usize {
        self.max_entries
    }

    fn max_metadata_memory(&self) -> usize {
        self.metadata_memory
    }

    fn max_plan_memory(&self) -> usize {
        self.plan_memory
    }
}

struct GenerationConfig {
    generation: std::sync::atomic::AtomicU32,
}

impl CacheConfig for GenerationConfig {
    fn generation(&self) -> u32 {
        self.generation.load(std::sync::atomic::Ordering::Relaxed)
    }
}

struct PurgeIntervalConfig {
    interval: u64,
}

impl CacheConfig for PurgeIntervalConfig {
    fn plan_purge_interval(&self) -> u64 {
        self.interval
    }
}

impl BackwardNode<f32, MoiraiBackend> for TestNode {
    fn op_name(&self) -> &'static str {
        self.name
    }

    fn output_grad(&self) -> &Arc<GradBuffer<f32, MoiraiBackend>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<f32, MoiraiBackend>] {
        &self.inputs
    }

    fn backward(
        &self,
        _grad_out: &Tensor<f32, MoiraiBackend>,
        _input_grads: &[Option<Arc<GradBuffer<f32, MoiraiBackend>>>],
    ) -> Result<(), BackendError> {
        Ok(())
    }
}

fn test_graph() -> Arc<dyn BackwardNode<f32, MoiraiBackend>> {
    let leaf = Var::new(Tensor::zeros([1]), true);
    let child: Arc<dyn BackwardNode<f32, MoiraiBackend>> = Arc::new(TestNode {
        name: "child",
        output_grad: Arc::new(GradBuffer::new(Tensor::zeros([1]))),
        inputs: vec![leaf],
    });
    let child_output = Var::with_creator(
        Tensor::zeros([1]),
        Some(Arc::clone(child.output_grad())),
        child,
    );

    Arc::new(TestNode {
        name: "root",
        output_grad: Arc::new(GradBuffer::new(Tensor::zeros([1]))),
        inputs: vec![child_output],
    })
}

mod budgets;
mod plans;
