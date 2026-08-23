# Computation Graph Caching - Quick Start Guide

## What Is It?

Computation graph caching in Coeus Autodiff caches graph metadata for repeated structures and reuses topology plans when the same live graph root is retained. Measure your workload rather than assuming a fixed percentage improvement.

## How It Works

1. **First backward**: The graph is traversed once
   - Topological order and graph metadata are collected
   - A structural fingerprint is recorded
   - A weak-root-scoped topology plan is retained when the root remains alive

2. **Later backwards**:
   - The same live root can reuse its topology plan
   - Separately rebuilt graphs share metadata only and still perform a live traversal
   - Cache statistics and snapshots distinguish these two paths

## Key Features

✅ **Automatic**: No code changes required
✅ **Thread-safe**: Integrated via thread-local cache
✅ **Correct**: Gradients unchanged, cache never affects results
✅ **Configurable**: Custom cache policies via `CacheConfig` trait
✅ **Observable**: Statistics tracking (hits, misses, memory)
✅ **Performant**: LRU eviction, generation-based invalidation

## Basic Usage

Your existing code automatically benefits from caching:

```rust
use coeus_autograd::{Var, add, mul, sum};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

fn solver_iteration(x: &Var<f32, MoiraiBackend>) -> Result<(), Box<dyn std::error::Error>> {
    let y = add(x, x);
    let z = mul(&y, x);
    let loss = sum(&z);
    loss.backward()?;
    x.zero_grad();
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([3], &[1.0, 2.0, 3.0]),
        true
    );
    
    // Each rebuilt loss has a new root: metadata is reusable, but its live
    // topology plan is intentionally not shared across graph instances.
    solver_iteration(&x)?;
    
    // Repeated structure produces metadata hits; retain one graph root when
    // topology-plan reuse is required.
    for _ in 1..100 {
        solver_iteration(&x)?;
    }
    
    Ok(())
}
```

## Monitoring Cache Performance

```rust
use coeus_autograd::get_backward_cache;

let cache = get_backward_cache();
let stats = cache.stats();

println!("Cache hits: {}", stats.hits);
println!("Cache misses: {}", stats.misses);
println!("Hit rate: {:.1}%", stats.hit_rate());
println!("Memory used: {} bytes", stats.memory_bytes);
println!("Metadata residency: {} entries, {} evictions", stats.metadata_entries, stats.metadata_evictions);
println!("Plan residency: {} entries, {} bytes", stats.plan_entries, stats.plan_memory_bytes);
// Monotonic high-water marks; reset with reset_stats().
println!("Plan peak: {} entries, {} bytes", stats.peak_plan_entries, stats.peak_plan_memory_bytes);
println!("Plan reuse: {:.1}%", stats.plan_hit_rate());
println!("Plan lookups: {}", stats.total_plan_ops());
```

## Example: Newton's Method Solver

```rust
use coeus_autograd::{Var, mul, add, sub, sum, get_backward_cache};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

fn compute_residual(x: &Var<f32, MoiraiBackend>) -> Var<f32, MoiraiBackend> {
    // f(x) = x^2 - 2 (finding sqrt(2))
    let x2 = mul(x, x);
    sub(&x2, &Var::new(Tensor::from_slice([1], &[2.0]), false))
}

fn newton_solver(x0: f32, max_iters: usize, tol: f32) -> Result<(), Box<dyn std::error::Error>> {
    let x = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([1], &[x0]),
        true
    );
    
    for iter in 0..max_iters {
        // Forward
        let residual = compute_residual(&x);
        let loss = sum(&residual);
        
        // Backward; this rebuilt graph can share metadata, but not a live
        // topology plan with the previous iteration.
        loss.backward()?;
        
        if iter > 0 && iter % 10 == 0 {
            let cache = get_backward_cache();
            let stats = cache.stats();
            println!(
                "Iter {}: loss={:?}, hit_rate={:.1}%, plan_reuse={:.1}%",
                iter,
                loss.tensor.as_slice()[0],
                stats.hit_rate(),
                stats.plan_hit_rate()
            );
        }
        
        x.zero_grad();
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    newton_solver(1.0, 100, 1e-6)?;
    Ok(())
}
```

## Running Benchmarks

### Simple Repeated Graph Benchmark

```bash
cargo test -p coeus-autograd --test cache_benchmarks \
    benches::simple_repeated_graph -- --ignored --nocapture
```

Expected output:
```
Iteration 0: Initial (expected miss)
  Time: 1.2ms
  Cache hits: 0, misses: 1

Iteration 10: With cache
  Time: 0.8ms
  Cache hits: 10, misses: 1
  Hit rate: 90.9%

=== Cache Performance Summary ===
First iteration (no cache): 1.20 ms
Avg subsequent (with cache): 0.80 ms  Time reduction: workload-dependent
```

### Deep Computation Graph Benchmark

```bash
cargo test -p coeus-autograd --test cache_benchmarks \
    benches::deep_computation_graph -- --ignored --nocapture
```

Expected output:
```
=== Deep Graph Cache Performance ===
First iteration: 3.50 ms
Avg subsequent: 2.40 ms  Time reduction: workload-dependent
```

### All Benchmarks

```bash
cargo test -p coeus-autograd --test cache_benchmarks -- --ignored --nocapture
```

## Integration Tests

Verify cache correctness:

```bash
cargo test -p coeus-autograd --test cache_integration
```

Tests cover:
- ✅ Gradient correctness with cache enabled
- ✅ Cache hits on repeated iterations
- ✅ Shape discrimination (different shapes don't incorrectly hit)
- ✅ Complex graph gradient accuracy
- ✅ Thread-local isolation
- ✅ Statistics tracking

## Custom Cache Configuration

```rust
use coeus_autograd::{CacheConfig, ComputeGraphCache};
use std::sync::Arc;

struct MyConfig {
    generation: u32,
}

impl CacheConfig for MyConfig {
    fn max_cache_entries(&self) -> usize {
        2048  // More graphs in memory
    }
    
    fn max_entry_memory(&self) -> usize {
        2 * 1024 * 1024  // 2MB per entry
    }

    fn max_metadata_memory(&self) -> usize {
        64 * 1024 * 1024  // 64MB across metadata entries
    }

    fn max_plan_memory(&self) -> usize {
        128 * 1024 * 1024  // 128MB across topology plans
    }

    fn plan_purge_interval(&self) -> u64 {
        32  // Full expired-plan scan every 32 ops once the table is large
    }
    
    fn is_enabled(&self) -> bool {
        true
    }
    
    fn generation(&self) -> u32 {
        self.generation
    }
}

fn main() {
    let config = Arc::new(MyConfig { generation: 0 });
    let cache = ComputeGraphCache::with_config(config);
    
    // Use custom-configured cache
    // Note: Currently only the default thread-local cache is used
    // This is for future extensibility
}
```

## Disabling Cache (for Comparison)

```rust
use coeus_autograd::CacheConfig;

struct NoCacheConfig;

impl CacheConfig for NoCacheConfig {
    fn is_enabled(&self) -> bool {
        false
    }
}

// Create instances without caching for benchmarking
```

## Performance Tips

### Maximize Cache Hits

1. **Use consistent shapes** across iterations
2. **Avoid dynamic graph creation** in solver loop
3. **Batch operations** together
4. **Increase `max_cache_entries`** if memory available

### Minimize Memory Usage

1. **Lower `max_cache_entries`** if needed
2. **Reduce `max_entry_memory`** per cached graph
3. **Monitor with `cache.stats().memory_bytes`**
4. **Set `max_metadata_memory` and `max_plan_memory` independently** when metadata and topology-plan residency need separate limits

## Troubleshooting

### Cache Misses Instead of Hits

Possible causes:
- **Different shapes per iteration**: Cache is discriminated by shape
- **Dynamic graph**: Graph changes between iterations
- **One-off computation**: No reuse opportunities

Check with:
```rust
let stats = get_backward_cache().stats();
println!("Hit rate: {:.1}%", stats.hit_rate());
```

### Memory Growth

Cache has automatic LRU eviction. If concerned:
- Monitor `stats.memory_bytes`
- Check `cache.size()` for entry count
- Adjust `max_cache_entries` in config

### Correctness Issues

Cache should never affect gradient accuracy. If gradients differ:
1. Disable cache: `impl CacheConfig` with `is_enabled = false`
2. Run with both cache enabled/disabled
3. Compare gradients (they should be identical)
4. Report issue with reproducible example

## Documentation

- **Full Design**: See `docs/autodiff_caching.md`
- **Implementation Details**: See `docs/cache_implementation_summary.md`
- **API Reference**: See `coeus-autograd` crate documentation

## Performance Expectations

There is no universal speedup guarantee. Same-root workloads can avoid repeated topology traversal, while dynamically rebuilt graphs still traverse their live nodes and primarily benefit from metadata accounting. Use the ignored benchmarks in `tests/cache_benchmarks.rs` and inspect `plan_hits`, `plan_misses`, and snapshot residency for workload-specific measurements.

The first backward on a retained graph builds its plan; later backwards can reuse it.

## FAQ

**Q: Do I need to change my code?**
A: No. Cache integration is automatic. Retain the same graph root when you want topology-plan reuse; rebuilt graphs receive metadata reuse only.

**Q: Is cache thread-safe?**
A: Yes! Each thread has its own cache via thread-local storage.

**Q: Can I disable cache for debugging?**
A: Implement `CacheConfig` with `is_enabled() = false`.

**Q: What if my graph changes every iteration?**
A: The cache records misses and does not reuse an incompatible live plan. Monitor `plan_hits` and `plan_misses` to confirm whether your workload benefits.

**Q: How much memory does cache use?**
A: Typically <50MB for default 1024 entries, configurable.

**Q: Are gradients affected by cache?**
A: Never! Cache only stores metadata, not values.

## Next Steps

1. Run the benchmarks to see performance gains
2. Monitor cache statistics in your solver
3. Configure cache based on your workload
4. Report any issues or feature requests

Happy accelerated autodiff! 🚀
