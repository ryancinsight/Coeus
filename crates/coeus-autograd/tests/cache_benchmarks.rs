//! Benchmarks measuring computation graph cache behavior.
//!
//! This module reports:
//! - First-pass versus subsequent-pass timing
//! - Cache hit rate and memory usage
//! - Cache effectiveness for repeated graph patterns
//!
//! The current cache stores metadata and weak-root-scoped topology plans. It
//! never reuses node pointers across separately constructed graphs. Timing
//! output is therefore observational rather than a fixed speedup gate for the
//! dynamic-graph workloads; `same_graph_topology_plan` measures plan reuse.

#![expect(
    clippy::print_stdout,
    reason = "test/bench/example code asserts and reports by design"
)]

#[path = "cache_benchmarks/benches/mod.rs"]
mod benches;
