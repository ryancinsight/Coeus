use coeus_autograd::backward_cache::compute_graph_structure_fingerprint;
use coeus_autograd::{
    add, get_backward_cache, mul, reset_backward_cache_stats, sum, topological_sort_with_cache,
    BackwardNode, Var, PLAN_PURGE_MIN_TABLE_SIZE,
};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use std::sync::Arc;
use std::time::Instant;

/// Timing + report body shared by the same-graph benchmarks.
///
/// `expected_plans` is the resident plan count the report should show
/// (1 for the single-plan benchmarks, the pre-filled table size for the
/// large-table variant). `prewarmed` selects the miss/hit accounting: when
/// the hot plan is already resident (pre-filled table), every timed
/// backward is a plan hit; otherwise the first is a miss and the rest hit.
/// Assertions are deltas against the pre-loop counters so either mode is
/// exact.
fn measure_same_graph_plan(
    label: &str,
    loss: &Var<f32, MoiraiBackend>,
    leaf: &Var<f32, MoiraiBackend>,
    iterations: usize,
    expected_plans: usize,
    prewarmed: bool,
) {
    let cache = get_backward_cache();
    let creator = loss
        .creator
        .as_ref()
        .expect("benchmark loss must have a creator");
    let (_, graph_info) = compute_graph_structure_fingerprint(creator);
    let before = cache.stats();
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        loss.backward().expect("benchmark backward must succeed");
        times.push(start.elapsed());
        leaf.zero_grad()
            .expect("invariant: test backend operation succeeds");
    }

    let stats = cache.stats();
    let first_ms = times[0].as_secs_f64() * 1000.0;
    let later_ms = times[1..]
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / (iterations - 1) as f64;
    let reduction = ((first_ms - later_ms) / first_ms) * 100.0;

    println!("\\n=== {label} Same-Graph Topology Plan ===");
    println!(
        "Graph nodes: {}, leaves: {}, max depth: {}",
        graph_info.node_count, graph_info.leaf_count, graph_info.max_depth
    );
    println!("First backward: {:.3} ms", first_ms);
    println!("Later average: {:.3} ms", later_ms);
    println!("Observed cold-to-warm reduction: {:.1}%", reduction);
    println!(
        "Plan hits: {}, misses: {}",
        stats.plan_hits, stats.plan_misses
    );
    println!(
        "Plan hit rate: {:.1}% across {} lookups",
        stats.plan_hit_rate(),
        stats.total_plan_ops()
    );
    println!(
        "Plan lifecycle: {} evictions, {} expirations",
        stats.plan_evictions, stats.plan_expirations
    );
    println!("Metadata hits: {}, misses: {}", stats.hits, stats.misses);
    let snapshot = cache.snapshot();
    println!(
        "Plan residency: {} entries, {} bytes (live snapshot: {} plans)",
        stats.plan_entries,
        stats.plan_memory_bytes,
        snapshot.plans.len()
    );
    // Plan table size vs the deferral threshold: below it the expired-plan
    // scan runs exactly on every operation; at/above it the scan is
    // deferred to every Nth op for amortized O(1). The threshold is the
    // cache's own constant, and the interval is the one the cache captured
    // from its config at construction — both sourced from the cache so the
    // report always agrees with the implementation.
    let deferral_threshold = PLAN_PURGE_MIN_TABLE_SIZE;
    let active_interval = cache.plan_purge_interval();
    let engaged = stats.plan_entries >= deferral_threshold;
    println!(
            "Plan table: {} entries, peak {} | {} the {}-entry deferral threshold -> {} | interval: {} ops{}",
            stats.plan_entries,
            stats.peak_plan_entries,
            if engaged { "at/above" } else { "below" },
            deferral_threshold,
            if engaged {
                "deferred (amortized) purge"
            } else {
                "exact per-op purge"
            },
            active_interval,
            if engaged { "" } else { " (not engaged below threshold)" }
        );
    println!(
        "Metadata residency: {} entries, {} bytes, {} evictions",
        snapshot.metadata_entries, snapshot.memory.metadata_bytes, stats.metadata_evictions
    );
    println!(
        "Memory breakdown: metadata {} B + plans {} B = total {} B",
        snapshot.memory.metadata_bytes, snapshot.memory.plan_bytes, snapshot.memory.total_bytes
    );
    for plan in &snapshot.plans {
        println!(
            "  Plan root={} nodes={} bytes={} age={} accesses={}",
            plan.root_id, plan.node_count, plan.memory_bytes, plan.residency_age, plan.access_count,
        );
    }

    if prewarmed {
        assert_eq!(stats.plan_misses, before.plan_misses);
        assert_eq!(stats.plan_hits, before.plan_hits + iterations as u64);
    } else {
        assert_eq!(stats.plan_misses, before.plan_misses + 1);
        assert_eq!(stats.plan_hits, before.plan_hits + (iterations - 1) as u64);
    }
    assert_eq!(snapshot.plans.len(), expected_plans);
    // Snapshot count must agree with the residency counter (snapshot purges
    // exactly before reporting).
    assert_eq!(snapshot.plans.len(), stats.plan_entries);
    // Find the hot plan by allocation identity rather than position: with a
    // large table the snapshot is sorted by (fingerprint, root_id), so
    // `plans[0]` is not necessarily the hot root.
    let hot_root_id = Arc::as_ptr(creator) as *const () as usize;
    let hot_plan = snapshot
        .plans
        .iter()
        .find(|plan| plan.root_id == hot_root_id)
        .unwrap_or_else(|| panic!("{label}: hot plan must be resident"));
    assert_eq!(hot_plan.node_count, graph_info.node_count);
}

/// Benchmark: Reuse a topology plan for repeated backward passes on one graph.
///
/// Unlike the dynamic-graph workloads above, this keeps the root graph alive
/// so the graph-local plan can be reused without retaining dropped graphs.
#[test]
#[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_topology_plan -- --ignored --nocapture
fn same_graph_topology_plan() {
    const ITERATIONS: usize = 100;

    reset_backward_cache_stats();
    let cache = get_backward_cache();
    cache.clear();

    let x = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([3], &[1.0, 2.0, 3.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let y = mul(&x, &x).expect("invariant: test operation succeeds");
    let loss = sum(&y).expect("invariant: test operation succeeds");
    measure_same_graph_plan("Small", &loss, &x, ITERATIONS, 1, false);
}

/// Benchmark: Topology-plan reuse for a deep sequential graph.
#[test]
#[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_deep_topology_plan -- --ignored --nocapture
fn same_graph_deep_topology_plan() {
    const ITERATIONS: usize = 100;
    const DEPTH: usize = 128;

    reset_backward_cache_stats();
    let cache = get_backward_cache();
    cache.clear();

    let x = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([4], &[0.1; 4]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let one = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([4], &[1.0; 4]).expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut y = x.clone();
    for _ in 0..DEPTH {
        y = add(&y, &x).expect("invariant: test operation succeeds");
        y = mul(&y, &one).expect("invariant: test operation succeeds");
    }
    let loss = sum(&y).expect("invariant: test operation succeeds");
    measure_same_graph_plan("Deep", &loss, &x, ITERATIONS, 1, false);
}

/// Benchmark: Topology-plan reuse for a large fan-out/fan-in graph.
#[test]
#[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_large_topology_plan -- --ignored --nocapture
fn same_graph_large_topology_plan() {
    const ITERATIONS: usize = 100;
    const LAYERS: usize = 8;
    const BRANCHES: usize = 16;

    reset_backward_cache_stats();
    let cache = get_backward_cache();
    cache.clear();

    let x = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([4], &[0.1; 4]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let one = Var::<f32, MoiraiBackend>::new(
        Tensor::from_slice([4], &[1.0; 4]).expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mut state = x.clone();
    for _ in 0..LAYERS {
        let mut merged = mul(&state, &one).expect("invariant: test operation succeeds");
        for _ in 1..BRANCHES {
            let branch = mul(&state, &one).expect("invariant: test operation succeeds");
            merged = add(&merged, &branch).expect("invariant: test operation succeeds");
        }
        state = merged;
    }
    let loss = sum(&state).expect("invariant: test operation succeeds");
    measure_same_graph_plan("Large", &loss, &x, ITERATIONS, 1, false);
}

/// Benchmark: Topology-plan reuse on one hot graph while the plan table
/// stays large, so the deferred-purge report line shows the engaged
/// (amortized) state instead of the exact per-op mode of the single-plan
/// benchmarks.
///
/// Pre-fills the table with `TABLE_SIZE` live plans (>= the
/// [`PLAN_PURGE_MIN_TABLE_SIZE`] deferral threshold; all roots retained so
/// nothing expires), then times repeated backward passes on one hot graph
/// whose plan is already resident. Every timed backward is a plan hit;
/// `plan_entries` stays at `TABLE_SIZE`, so the report reads
/// "at/above the {threshold}-entry deferral threshold -> deferred
/// (amortized) purge".
#[test]
#[ignore] // Run with: cargo test --test cache_benchmarks benches::same_graph_topology_plan_large_table -- --ignored --nocapture
fn same_graph_topology_plan_large_table() {
    const ITERATIONS: usize = 100;
    // Keep the table above the cache's deferral threshold so the
    // deferred-purge path stays engaged even if the threshold changes.
    const TABLE_SIZE: usize = PLAN_PURGE_MIN_TABLE_SIZE + 16;

    reset_backward_cache_stats();
    let cache = get_backward_cache();
    cache.clear();

    // Build TABLE_SIZE live graphs and insert a plan for each. All creator
    // roots are retained so every plan stays resident and the table sits
    // at/above the deferral threshold. Graph i gets (i % 4) + 1 layers so
    // the table holds several distinct fingerprints.
    let mut roots: Vec<Arc<dyn BackwardNode<f32, MoiraiBackend>>> = Vec::with_capacity(TABLE_SIZE);
    let mut hot: Option<(Var<f32, MoiraiBackend>, Var<f32, MoiraiBackend>)> = None;
    for i in 0..TABLE_SIZE {
        let x = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([4], &[0.1; 4]).expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");
        let one = Var::<f32, MoiraiBackend>::new(
            Tensor::from_slice([4], &[1.0; 4]).expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        let mut y = x.clone();
        for _ in 0..((i % 4) + 1) {
            y = add(&y, &x).expect("invariant: test operation succeeds");
            y = mul(&y, &one).expect("invariant: test operation succeeds");
        }
        let loss = sum(&y).expect("invariant: test operation succeeds");
        let creator = loss
            .creator
            .as_ref()
            .expect("benchmark loss must have a creator")
            .clone();
        let _ = topological_sort_with_cache(Some(&creator), &cache);
        roots.push(creator);
        if i == 0 {
            hot = Some((loss, x));
        }
    }

    let (loss, leaf) = hot.expect("hot graph must exist");
    // The hot plan is already resident: the timed loop is all hits.
    measure_same_graph_plan("LargeTable", &loss, &leaf, ITERATIONS, TABLE_SIZE, true);
}
