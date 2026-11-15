//! Cache-behavior benchmarks, grouped by the behavior they measure.

mod graphs;
mod lookup;
mod plans;

use coeus_autograd::CacheConfig;

/// Cache configuration with a custom amortized plan-purge interval.
struct PurgeInterval(u64);

impl CacheConfig for PurgeInterval {
    fn plan_purge_interval(&self) -> u64 {
        self.0
    }
}
