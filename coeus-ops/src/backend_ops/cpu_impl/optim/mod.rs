mod adam_adamw;
mod sgd_adagrad_rmsprop;

pub use adam_adamw::{adam_step, adamw_step};
pub use sgd_adagrad_rmsprop::{adagrad_step, rmsprop_step, sgd_step};

use coeus_core::Backend;

/// Element count at or below which an optimizer update runs sequentially instead
/// of dispatching across the Moirai pool.
///
/// Optimizer steps are light, memory-bandwidth-bound element-wise updates (a few
/// FLOPs per element), so for small parameter tensors a single auto-vectorized
/// loop beats the thread dispatch/join cost. This is deliberately the op-layer's
/// threshold: `coeus-ops` knows the per-element work is cheap, so it raises the
/// bar above Moirai's general-purpose `Adaptive` threshold (1024) before handing
/// off to the pool.
///
/// Note: benchmark evidence (`coeus-optim/benches/optim_bench.rs`) indicates the
/// parallel path stays slower than sequential well past this value for these
/// ops; tuning `SEQUENTIAL_THRESHOLD` upward is tracked as a follow-up requiring a
/// derived sequential-vs-parallel crossover measurement across sizes.
const SEQUENTIAL_THRESHOLD: usize = 4096;

/// Apply `f(i)` for every `i in 0..numel`, sequentially below
/// [`SEQUENTIAL_THRESHOLD`] (no dispatch/join overhead) or across Moirai's pool
/// above it. The same per-element closure serves both regimes, so each optimizer
/// keeps a single update body rather than duplicating a sequential and a parallel
/// variant.
#[inline]
fn dispatch<B, F>(backend: &B, numel: usize, f: F)
where
    B: Backend,
    F: Fn(usize) + Send + Sync + 'static,
{
    if numel <= SEQUENTIAL_THRESHOLD {
        for i in 0..numel {
            f(i);
        }
    } else {
        backend.parallel_for(0, numel, f);
    }
}
