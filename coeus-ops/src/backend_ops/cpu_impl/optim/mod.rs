mod adam_adamw;
mod sgd_adagrad_rmsprop;

pub use adam_adamw::{adam_step, adamw_step};
pub use sgd_adagrad_rmsprop::{adagrad_step, rmsprop_step, sgd_step};

use coeus_core::Backend;

/// Element count at or below which an optimizer update runs sequentially instead
/// of dispatching across the Moirai pool.
///
/// Optimizer steps are light, memory-bandwidth-bound element-wise updates, so the
/// parallel path's fixed thread dispatch/join cost (~20 µs, measured) dominates
/// until the tensor is large. This is deliberately the op-layer's threshold:
/// `coeus-ops` knows the per-element work is cheap, so it sets the bar far above
/// Moirai's general-purpose `Adaptive` threshold (1024) before handing to the pool.
///
/// Derived from `coeus-optim/benches/optim_bench.rs` sequential-vs-parallel
/// crossover measurements (f32, MoiraiBackend). Per-op sequential wins hold up to
/// roughly SGD ~425K elements, Adam ~130K (heavier per element → lower crossover).
/// A single shared threshold must not regress the heaviest op, so it sits below the
/// *minimum* crossover (Adam's) with margin: at 65_536 both are decisively
/// sequential (SGD 10.7 vs 38 µs; Adam 33 vs 57 µs, non-overlapping CIs) — a 16×
/// lift over the previous 4096 across the common small/medium range. Above it both
/// dispatch to the pool (parallel wins for SGD by ~1M, Adam by ~260K).
const SEQUENTIAL_THRESHOLD: usize = 65_536;

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
