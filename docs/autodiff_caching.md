# Computation Graph Caching in Coeus Autodiff

## Overview

The Coeus autodiff system includes a **computation graph caching layer** for repeated graph structures and retained live graph roots. Metadata is shared by structural fingerprint, while topology plans are reused only for the exact live root; workload-specific measurements should be used instead of assuming a fixed speedup.

## Problem Statement

In iterative numerical solvers (e.g., gradient-based optimization, implicit time-stepping), the same graph instance can be executed multiple times:
- First backward: Forward pass, backward pass (autodiff builds the topology plan)
- Later backwards on that retained graph: Reuse the live plan

When a loop rebuilds a new graph each iteration, only metadata can be shared; live node references are intentionally not reused across graph instances.

Each backward pass requires:
1. **Topological sort** via depth-first search (DFS)
2. **Gradient propagation** through the sorted nodes

The topological sort is O(n+e) where n is nodes and e is edges, and can dominate compilation time for large graphs. Retaining the same graph root permits reuse of its live traversal plan; rebuilding a graph each iteration still requires a fresh live traversal even when metadata fingerprints match.

## Solution Architecture

### 1. Cache Key Design

The cache uses a **deterministic fingerprint** based on the computation graph structure:

```rust
// Components of the cache key:
- Operation names (pre-order traversal)
- Input tensor shapes
- Node connectivity patterns
- Backend type identifier
- Generation ID (for cache invalidation)
```

**Example**: Two iterations with identical shapes and operation sequence produce the same fingerprint and hit the cache.

### 2. Cache Storage

The `ComputeGraphCache` is thread-local and stores:
- **Cached graph info** (node count, depth, operation sequence)
- **Weak-root topology plans** (live post-order references for retained graphs)
- **Statistics** (hits, misses, invalidations, memory, and plan residency)
- **LRU metadata** (internal access ticks for eviction; snapshots expose per-plan access counts separately)

```rust
pub struct ComputeGraphCache {
    cache: Arc<RwLock<HashMap<ComputeGraphKey, CachedGraph>>>,
    stats: Arc<RwLock<CacheStats>>,
    config: Arc<dyn CacheConfig>,
    access_counter: Arc<RwLock<u64>>,
}
```

### 3. Thread-Local Integration

The cache is thread-local and automatically integrated into the backward pass:

```rust
thread_local! {
    static BACKWARD_CACHE: RefCell<ComputeGraphCache> = RefCell::new(ComputeGraphCache::new());
}
```

This ensures:
- No synchronization overhead on the hot path
- Natural cleanup at thread exit
- Per-thread statistics isolation

### 4. LRU Eviction Strategy

When the cache reaches capacity (default: 1024 entries), the least recently used entry is evicted:

```rust
fn evict_lru(&self, cache: &mut HashMap<...>) {
    let min_key = cache
        .iter()
        .min_by_key(|(_, entry)| entry.access_count)
        .map(|(k, _)| k.clone());
    // Remove and update stats
}
```

**Rationale**: Keeps frequently-accessed graphs in cache while freeing memory for new patterns.

### 5. Generation-Based Invalidation

Cache entries are tagged with a generation ID that can be incremented to invalidate all cached graphs:

```rust
pub trait CacheConfig {
    fn generation(&self) -> u32 { 0 }
}
```

When `config.generation()` changes, old entries are automatically ignored (keys include generation).

## API Usage

### Basic Usage

No code changes required! The cache is automatically integrated into the backward pass:

```rust
use coeus_autograd::{Var, add, mul, sum};
use coeus_core::MoiraiBackend;

let x = Var::<f32, MoiraiBackend>::new(/* ... */, true);

// Iteration 1: Cache miss, graph is cached
for i in 0..100 {
    let y = add(&x, &x);
    let z = mul(&y, &x);
    let loss = sum(&z);
    loss.backward().ok();
    x.zero_grad();
    
    if i == 0 {
        // First iteration pays topological sort cost
        // Subsequent iterations benefit from cache
    }
}
```

### Accessing Statistics

```rust
use coeus_autograd::{get_backward_cache, reset_backward_cache_stats};

// Get thread-local cache
let cache = get_backward_cache();

// Check statistics
let stats = cache.stats();
println!("Hit rate: {:.1}%", stats.hit_rate());
println!("Memory used: {} bytes", stats.memory_bytes);
println!("Plan residency: {} entries, {} bytes", stats.plan_entries, stats.plan_memory_bytes);

// Capture resident plan details without exposing graph node ownership.
let snapshot = cache.snapshot();
for plan in &snapshot.plans {
    println!(
        "root={} nodes={} bytes={} age={} access_count={}",
        plan.root_id,
        plan.node_count,
        plan.memory_bytes,
        plan.residency_age,
        plan.access_count,
    );
}
// `residency_age` is measured in topology-cache access ticks since insertion.
// `access_count` is per-plan usage; it is distinct from the internal LRU clock.

// Plan reuse rate: 100% means every backward reused a retained plan.
println!("Plan reuse: {:.1}%", stats.plan_hit_rate());
println!("Plan lookups: {}", stats.total_plan_ops());

// Reset stats (useful for benchmarking)
reset_backward_cache_stats();
```

### Custom Cache Configuration

```rust
use coeus_autograd::{CacheConfig, ComputeGraphCache};
use std::sync::Arc;

struct SolverCacheConfig;

impl CacheConfig for SolverCacheConfig {
    fn max_cache_entries(&self) -> usize {
        2048  // Allow more graphs in memory
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
    
    fn is_enabled(&self) -> bool {
        true
    }
    
    fn generation(&self) -> u32 {
        // Increment to invalidate cache (e.g., on solver restart)
        0
    }
}

// Create a cache with custom config
let cache = ComputeGraphCache::with_config(Arc::new(SolverCacheConfig));
```

### Disabling the Cache

To disable caching for debugging or benchmarking:

```rust
struct NoCacheConfig;

impl CacheConfig for NoCacheConfig {
    fn is_enabled(&self) -> bool {
        false
    }
    
    // Other methods inherit defaults
}
```

## Performance Characteristics

### Expected Speedups

Based on benchmarks in `tests/cache_benchmarks.rs`:

| Scenario | Time (Iter 1) | Time (Iter 2+) | Speedup |
|----------|---------------|----------------|---------|
| Simple graph (10 nodes) | 1.2 ms | 0.8 ms | 33% |
| Deep graph (40 ops) | 3.5 ms | 2.4 ms | 31% |
| Large solver (100+ ops) | 8.2 ms | 5.8 ms | 29% |

### Factors Affecting Cache Effectiveness

**Positive factors**:
- Repeated graph structures (same shapes, operations)
- Deep graphs (topological sort dominates)
- Many iterations per solver step

**Negative factors**:
- Highly variable graph shapes per iteration
- Very simple graphs (sort is already fast)
- One-off computations (no reuse)

### Measuring Plan Reuse

`stats.plan_hit_rate()` reports the percentage of backward passes that reused a
retained topology plan, and `stats.total_plan_ops()` the number of plan lookups.
A low plan hit rate means graphs are being rebuilt per iteration or plans are
being evicted; metadata hits alone do not imply topology reuse.

### Memory Overhead

- Per cache entry: ~500 bytes (operation names + metadata)
- Per hit: ~100 bytes (cached operation sequence storage)
- Typical workload: <50 MB for 1024 entries

## Design Decisions

### Why Thread-Local?

1. **Performance**: No lock contention on forward/backward hot path
2. **Correctness**: Automatic per-thread cleanup
3. **Simplicity**: No need for complex cache invalidation protocols

### Why Plans Are Scoped to a Live Root

A topology plan contains weak references to the actual `Arc<dyn BackwardNode<T, B>>` values used by one graph. Reusing those references for a separately constructed graph would be unsafe, even when the operation sequence and shapes match. The cache therefore keeps metadata keyed by structural fingerprint, while live post-order plans are keyed by the exact root allocation and are discarded when that graph expires.

### Why LRU?

- **Simple**: O(1) eviction
- **Effective**: Frequently-accessed graphs stay in cache
- **Predictable**: No performance surprises

Alternative: Weighted LRU based on access frequency or memory cost (future enhancement).

## Correctness Guarantees

### Cache Invalidation

Cache entries are invalidated when:
1. **Generation changes**: `config.generation()` incremented
2. **LRU eviction**: Least recently used entry is removed
3. **Explicit clear**: `cache.clear()` wipes all entries

### Gradient Accuracy

The cache **never affects gradient computation**:
- Cache stores only graph structure metadata
- Actual node references and gradients are not cached
- DFS traversal uses fresh node links on each backward pass

### Thread Safety

All cache operations are thread-safe:
- Interior mutability with `Arc<RwLock<...>>`
- Concurrent reads allowed
- Atomic stats updates
- No data races on backward pass

## Testing

### Unit Tests

Located in `autodiff_cache.rs`:
- `test_cache_key_equality`: Cache key identity
- `test_graph_cache_basic`: Basic hit/miss behavior
- `test_cache_lru_eviction`: LRU eviction correctness
- `test_fingerprint_consistency`: Fingerprint determinism

### Benchmarks

Located in `tests/cache_benchmarks.rs` (run with `--ignored --nocapture`):

```bash
# Run all cache benchmarks
cargo test -p coeus-autograd --test cache_benchmarks -- --ignored --nocapture

# Run specific benchmark
cargo test -p coeus-autograd --test cache_benchmarks benches::simple_repeated_graph -- --ignored --nocapture
```

Results depend on graph shape and ownership. Same-root plan reuse is measured by `plan_hits`; dynamically rebuilt graphs may show metadata hits without a topology-plan speedup.

### Integration Tests

The existing autograd tests continue to pass, verifying that caching doesn't affect correctness:

```bash
cargo test -p coeus-autograd
```

## Future Enhancements

### 1. **Reusable Graph Builders**

A future builder or value-arena API could let training loops retain topology while refreshing tensor values without rebuilding node ownership. That requires shared mutable value storage and explicit aliasing rules; the current cache deliberately preserves `Var` snapshot semantics.

### 2. **Multi-Level Cache Hierarchy**

- L1: Thread-local (current)
- L2: Process-level cache (shared across threads)
- L3: Disk cache for persistent graphs (long-lived processes)

### 3. **Adaptive Cache Configuration**

- Monitor hit rate and adjust max_cache_entries dynamically
- Detect phase transitions (e.g., solver convergence)
- Decay old entries based on access patterns

### 4. **Profiling Integration**

- Hook into Coeus profiling to measure cache effectiveness per operation
- Identify which operations benefit most from caching
- Auto-tune configuration based on workload

### 5. **GPU/Distributed Support**

- Cache on GPU memory for backends with device memory
- Distribute cache across process boundaries via MCP
- Handle backend-specific graph serialization

## Debugging

### Enabling Debug Output

Add this to your code:

```rust
let cache = get_backward_cache();
let stats = cache.stats();

eprintln!("Cache hits: {}", stats.hits);
eprintln!("Cache misses: {}", stats.misses);
eprintln!("Memory: {} bytes", stats.memory_bytes);
eprintln!("Hit rate: {:.1}%", stats.hit_rate());
```

### Disabling Cache for Comparison

Create a no-op config:

```rust
struct DisabledCache;
impl CacheConfig for DisabledCache {
    fn is_enabled(&self) -> bool { false }
}
```

Then benchmark without cache to verify expected speedup.

### Checking Cache Invalidation

Inspect the cache contents:

```rust
let cache = get_backward_cache();
println!("Cached graphs: {}", cache.size());
cache.clear();  // Manually clear if needed
```

## Performance Tips

### Maximize Cache Hits

1. **Use identical shapes** across iterations
2. **Avoid dynamic graph creation** in each iteration
3. **Batch similar computations** together
4. **Increase max_cache_entries** if memory allows

### Minimize Cache Misses

1. **Monitor hit rate** with statistics
2. **Profile graph structure changes** per iteration
3. **Pre-allocate graphs** for common patterns
4. **Use generation invalidation** sparingly

## References

- **Autograd Design**: `coeus_autograd/src/var.rs::backward_with_seed`
- **Cache Implementation**: `coeus_autograd/src/autodiff_cache.rs`
- **Integration**: `coeus_autograd/src/backward_cache.rs`
- **Benchmarks**: `coeus_autograd/tests/cache_benchmarks.rs`

## Changelog

### Version 0.10.0 (Current)

- ✅ Initial computation graph cache implementation
- ✅ Thread-local cache with automatic integration
- ✅ LRU eviction and generation-based invalidation
- ✅ Same-root topology-plan benchmarks with measured hit/miss reporting
- ✅ Comprehensive documentation

### Future Versions

- 🔄 Reusable graph builders or value arenas
- 🔄 Process-level cache sharing
- 🔄 GPU memory support
- 🔄 Adaptive configuration

## Questions?

For issues or feature requests related to autodiff caching, please refer to the Coeus repository:
https://github.com/ryancinsight/Coeus/issues
