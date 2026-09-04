//! Cache configuration for [`super::ComputeGraphCache`].

use super::PLAN_PURGE_INTERVAL;

/// Configuration trait for cache behavior.
///
/// Allows solver-specific customization of cache parameters.
pub trait CacheConfig: Send + Sync {
    /// Maximum number of cached graphs.
    fn max_cache_entries(&self) -> usize {
        1024
    }

    /// Maximum memory per cached entry (bytes).
    fn max_entry_memory(&self) -> usize {
        1024 * 1024 // 1MB default
    }

    /// Maximum aggregate memory for cached metadata entries (bytes).
    ///
    /// The default is unlimited so existing configurations retain the previous
    /// per-entry-only behavior. Override this to bound metadata residency.
    fn max_metadata_memory(&self) -> usize {
        usize::MAX
    }

    /// Maximum aggregate memory for resident topology plans (bytes).
    ///
    /// The default is unlimited so existing configurations retain the previous
    /// per-entry-only behavior. Override this to bound topology-plan residency.
    fn max_plan_memory(&self) -> usize {
        usize::MAX
    }

    /// Amortized expired-plan purge period (plan operations).
    ///
    /// Small plan tables (below the fixed 64-entry minimum) are purged exactly
    /// on every operation. Once the table is at or above that size, the full
    /// expired-plan scan is deferred to every Nth operation, keeping repeated
    /// lookups amortized O(1). A value of `0` disables deferral entirely (purge
    /// on every operation, so residency counters are always exact). Tune lower
    /// for fresher residency accounting, higher for cheaper lookups on large
    /// plan tables.
    fn plan_purge_interval(&self) -> u64 {
        PLAN_PURGE_INTERVAL
    }

    /// Whether to enable caching.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Cache generation ID (invalidates all entries when incremented).
    fn generation(&self) -> u32 {
        0
    }
}

/// Default cache configuration.
#[derive(Clone, Debug, Default)]
pub struct DefaultCacheConfig;

impl CacheConfig for DefaultCacheConfig {
    fn max_cache_entries(&self) -> usize {
        1024
    }

    fn max_entry_memory(&self) -> usize {
        1024 * 1024
    }

    fn is_enabled(&self) -> bool {
        true
    }

    fn generation(&self) -> u32 {
        0
    }
}
