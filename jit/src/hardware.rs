//! Hardware detection and capability assessment for JIT compilation

use crate::error::{JitError, Result};
use raw_cpuid::CpuId;
use std::sync::OnceLock;

/// SIMD instruction set support levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD support
    None,
    /// SSE (Streaming SIMD Extensions)
    Sse,
    /// SSE2 (SSE extensions)
    Sse2,
    /// SSE3 (SSE3 extensions)
    Sse3,
    /// SSSE3 (Supplemental SSE3)
    Ssse3,
    /// SSE4.1 support
    Sse41,
    /// SSE4.2 support
    Sse42,
    /// AVX (Advanced Vector Extensions)
    Avx,
    /// AVX2 (256-bit AVX)
    Avx2,
    /// AVX-512 Foundation
    Avx512f,
    /// AVX-512 with additional features
    Avx512Full,
}

/// CPU architecture and capabilities
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// CPU architecture
    pub architecture: Architecture,
    /// SIMD support level
    pub simd_level: SimdLevel,
    /// Cache line size in bytes
    pub cache_line_size: usize,
    /// L1 data cache size in bytes
    pub l1_cache_size: usize,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    /// L3 cache size in bytes (if available)
    pub l3_cache_size: Option<usize>,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Total logical cores (including hyper-threading)
    pub logical_cores: usize,
    /// Maximum SIMD register width in bits
    pub max_simd_width: usize,
    /// Has FMA3 support (Fused Multiply-Add)
    pub has_fma3: bool,
    /// Has FMA4 support
    pub has_fma4: bool,
    /// Prefetch instruction support
    pub has_prefetch: bool,
    /// Hardware transactional memory support
    pub has_tsx: bool,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self::detect().unwrap_or_else(|_| Self::fallback())
    }
}

impl HardwareCapabilities {
    /// Detect hardware capabilities at runtime
    pub fn detect() -> Result<Self> {
        let cpuid = CpuId::new();

        // Get basic CPU information
        let _vendor_info = cpuid
            .get_vendor_info()
            .ok_or_else(|| JitError::CompilationFailed {
                message: "Failed to get CPU vendor info".to_string(),
            })?;

        let feature_info = cpuid
            .get_feature_info()
            .ok_or_else(|| JitError::CompilationFailed {
                message: "Failed to get CPU feature info".to_string(),
            })?;

        let extended_feature_info = cpuid.get_extended_feature_info();
        let extended_processor_and_feature_identifiers =
            cpuid.get_extended_processor_and_feature_identifiers();

        // Determine architecture
        let architecture = if std::env::consts::ARCH == "x86_64" {
            Architecture::X86_64
        } else if std::env::consts::ARCH == "aarch64" {
            Architecture::AArch64
        } else {
            Architecture::Unknown
        };

        // Detect SIMD level based on features
        let mut simd_level = SimdLevel::None;

        // Simplified SIMD detection due to API changes
        if feature_info.has_avx() {
            // Basic AVX support
            if feature_info.has_avx() {
                // TODO: Re-enable full AVX-512 detection when API stabilizes
                // For now, assume AVX2 if AVX is available
                simd_level = SimdLevel::Avx2;
            }
        } else if feature_info.has_sse42() {
            simd_level = SimdLevel::Sse42;
        } else if feature_info.has_sse41() {
            simd_level = SimdLevel::Sse41;
        } else if feature_info.has_ssse3() {
            simd_level = SimdLevel::Ssse3;
        } else if feature_info.has_sse3() {
            simd_level = SimdLevel::Sse3;
        } else if feature_info.has_sse2() {
            simd_level = SimdLevel::Sse2;
        } else if feature_info.has_sse() {
            simd_level = SimdLevel::Sse;
        }

        // Get cache information
        let _cache_info = cpuid.get_cache_info();
        // Simplified cache detection due to API changes
        let cache_line_size = 64; // Default to 64 bytes
        let l1_cache_size = 32 * 1024; // Default to 32KB
        let l2_cache_size = 256 * 1024; // Default to 256KB
        let l3_cache_size = Some(8 * 1024 * 1024); // Default to 8MB

        // Get core information
        let logical_cores = num_cpus::get();
        let physical_cores = logical_cores; // Simplified

        // Determine maximum SIMD width
        let max_simd_width = match simd_level {
            SimdLevel::Avx512Full | SimdLevel::Avx512f => 512,
            SimdLevel::Avx2 | SimdLevel::Avx => 256,
            SimdLevel::Sse
            | SimdLevel::Sse2
            | SimdLevel::Sse3
            | SimdLevel::Ssse3
            | SimdLevel::Sse41
            | SimdLevel::Sse42 => 128,
            SimdLevel::None => 64, // Scalar operations
        };

        // Check for additional features
        let has_fma3 = feature_info.has_fma();
        let has_fma4 = extended_processor_and_feature_identifiers.is_some_and(
            |e: raw_cpuid::ExtendedProcessorFeatureIdentifiers| e.has_fma4(),
        );

        let has_prefetch = feature_info.has_sse(); // Simplified prefetch detection

        let has_tsx =
            extended_feature_info.is_some_and(|e: raw_cpuid::ExtendedFeatures| e.has_rtm());

        Ok(HardwareCapabilities {
            architecture,
            simd_level,
            cache_line_size,
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            physical_cores,
            logical_cores,
            max_simd_width,
            has_fma3,
            has_fma4,
            has_prefetch,
            has_tsx,
        })
    }

    /// Fallback capabilities for when CPUID is unavailable
    fn fallback() -> Self {
        let logical_cores = num_cpus::get();

        HardwareCapabilities {
            architecture: if std::env::consts::ARCH == "x86_64" {
                Architecture::X86_64
            } else if std::env::consts::ARCH == "aarch64" {
                Architecture::AArch64
            } else {
                Architecture::Unknown
            },
            simd_level: SimdLevel::None,
            cache_line_size: 64,
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: None,
            physical_cores: logical_cores,
            logical_cores,
            max_simd_width: 64,
            has_fma3: false,
            has_fma4: false,
            has_prefetch: false,
            has_tsx: false,
        }
    }

    /// Check if SIMD is supported at the specified level or higher
    pub fn supports_simd(&self, level: SimdLevel) -> bool {
        self.simd_level >= level
    }

    /// Get optimal vector width in bytes for the current SIMD level
    pub fn optimal_vector_bytes(&self, element_bytes: usize) -> usize {
        // Maximum elements that can fit in a SIMD register
        let max_elements_in_register = self.max_simd_width / (element_bytes * 8);

        // For best performance, prefer powers of 2
        match max_elements_in_register {
            n if n >= 16 => 16, // Full register utilization
            n if n >= 8 => 8,
            n if n >= 4 => 4,
            _ => 1, // Scalar fallback
        }
    }

    /// Estimate memory bandwidth based on cache hierarchy
    pub fn estimated_memory_bandwidth_gb_per_sec(&self) -> f64 {
        // Rough estimation based on cache sizes and typical bandwidths
        // This is a simplified model; real bandwidth depends on many factors

        let base_bandwidth = match self.simd_level {
            SimdLevel::Avx512Full => 100.0, // AVX-512 can achieve high bandwidth
            SimdLevel::Avx512f | SimdLevel::Avx2 => 80.0,
            SimdLevel::Avx => 60.0,
            SimdLevel::Sse42 | SimdLevel::Sse41 => 40.0,
            _ => 20.0,
        };

        // Adjust based on cache sizes (more cache = higher effective bandwidth)
        let cache_multiplier =
            (self.l1_cache_size + self.l2_cache_size) as f64 / (64.0 * 1024.0 + 512.0 * 1024.0);
        let cores_multiplier = self.logical_cores as f64 / 4.0; // Normalize to 4 cores

        base_bandwidth * cache_multiplier * cores_multiplier
    }
}

/// CPU architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// x86-64 architecture
    X86_64,
    /// ARM64/AArch64 architecture
    AArch64,
    /// Unknown/unsupported architecture
    Unknown,
}

static HARDWARE_CAPABILITIES: OnceLock<HardwareCapabilities> = OnceLock::new();

/// Get globally cached hardware capabilities
pub fn get_hardware_capabilities() -> &'static HardwareCapabilities {
    HARDWARE_CAPABILITIES.get_or_init(|| {
        HardwareCapabilities::detect().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to detect hardware capabilities: {}", e);
            HardwareCapabilities::fallback()
        })
    })
}

/// Hardware-specific cost model for operations
#[derive(Debug, Clone)]
pub struct HardwareCostModel {
    pub capabilities: &'static HardwareCapabilities,
}

impl HardwareCostModel {
    /// Create a new cost model for the current hardware
    pub fn new() -> Self {
        Self {
            capabilities: get_hardware_capabilities(),
        }
    }

    /// Estimate cycles for a SIMD vector operation
    pub fn estimate_simd_cycles(&self, operation: &str, vector_width: usize) -> u64 {
        let base_cycles = match operation {
            "add" | "sub" => 1,
            "mul" => 2,
            "fma" => 3,
            "div" => 10,
            "sqrt" => 8,
            "exp" | "log" => 15,
            _ => 5,
        };

        // SIMD operations are effectively parallel
        let parallelism_factor = (vector_width as f64 / 64.0).max(1.0);
        let adjusted_cycles = (base_cycles as f64 / parallelism_factor) as u64;

        // AVX-512 has additional efficiency
        if self.capabilities.supports_simd(SimdLevel::Avx512f) && vector_width >= 16 {
            adjusted_cycles.saturating_sub(1)
        } else {
            adjusted_cycles
        }
    }

    /// Estimate cycles for memory access with cache consideration
    pub fn estimate_memory_cycles(&self, access_size: usize, is_sequential: bool) -> u64 {
        let cache_line_size = self.capabilities.cache_line_size;

        if access_size <= cache_line_size {
            // L1 cache hit
            if is_sequential {
                2
            } else {
                4
            }
        } else if access_size <= self.capabilities.l2_cache_size {
            // L2 cache hit
            if is_sequential {
                8
            } else {
                12
            }
        } else {
            // Main memory access (much slower)
            if is_sequential {
                50
            } else {
                100
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let caps =
            HardwareCapabilities::detect().unwrap_or_else(|_| HardwareCapabilities::fallback());

        // Basic sanity checks
        assert!(caps.logical_cores > 0);
        assert!(caps.cache_line_size >= 16); // Minimum cache line size
        assert!(caps.max_simd_width >= 64); // At least scalar operations
    }

    #[test]
    fn test_simd_support_levels() {
        let caps =
            HardwareCapabilities::detect().unwrap_or_else(|_| HardwareCapabilities::fallback());

        // SIMD support should be monotonically non-decreasing
        let levels = vec![
            SimdLevel::None,
            SimdLevel::Sse,
            SimdLevel::Sse2,
            SimdLevel::Sse3,
            SimdLevel::Ssse3,
            SimdLevel::Sse41,
            SimdLevel::Sse42,
            SimdLevel::Avx,
            SimdLevel::Avx2,
            SimdLevel::Avx512f,
            SimdLevel::Avx512Full,
        ];

        let current_level = caps.simd_level as usize;
        for (i, level) in levels.into_iter().enumerate() {
            if i <= current_level {
                assert!(
                    caps.supports_simd(level),
                    "Should support SIMD level {:?}",
                    level
                );
            }
        }
    }

    #[test]
    fn test_cost_model() {
        let model = HardwareCostModel::new();

        // Vector operations should be faster than scalar
        let scalar_cycles = model.estimate_simd_cycles("add", 1);
        let vector_cycles = model.estimate_simd_cycles("add", 8);

        assert!(
            vector_cycles <= scalar_cycles,
            "Vector operations should be faster or equal"
        );

        // Memory access costs should be reasonable
        let l1_cycles = model.estimate_memory_cycles(64, true);
        let mem_cycles = model.estimate_memory_cycles(1024 * 1024, true);

        assert!(
            l1_cycles < mem_cycles,
            "Memory access should be faster for smaller data"
        );
    }
}
