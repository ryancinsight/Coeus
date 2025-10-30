use crate::error::{JitError, Result};
use crate::hardware::{get_hardware_capabilities, HardwareCapabilities};

/// Prefetch optimization for cache efficiency
#[derive(Debug)]
pub struct PrefetchOptimizer {
    #[allow(dead_code)]
    cache_line_size: usize,
}

impl PrefetchOptimizer {
    /// Create a new prefetch optimizer
    pub fn new() -> Self {
        Self {
            cache_line_size: 64, // Assume 64-byte cache lines
        }
    }

    /// Estimate prefetch benefit for a given memory access pattern
    pub fn estimate_prefetch_benefit(&self, _pattern: &coeus_backend::MemoryAccessPattern) -> f32 {
        // Simple heuristic: assume 5-15% improvement for sequential access
        0.1 // 10% improvement
    }
}

/// SIMD specialization for different architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdSpecialization {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE (128-bit SIMD)
    Sse,
    /// AVX (256-bit SIMD)
    Avx,
    /// AVX2 (256-bit SIMD with FMA)
    Avx2,
    /// AVX-512 (512-bit SIMD)
    Avx512,
    /// ARM NEON (128-bit SIMD for AArch64)
    Neon,
}

/// SIMD kernel generator for element-wise operations
#[derive(Debug)]
pub struct SimdKernelGenerator {
    #[allow(dead_code)]
    capabilities: &'static HardwareCapabilities,
    specialization: SimdSpecialization,
}

impl SimdKernelGenerator {
    /// Create a new SIMD kernel generator for the current hardware
    pub fn new() -> Self {
        let capabilities = get_hardware_capabilities();
        let specialization = Self::detect_specialization(capabilities);
        Self {
            capabilities,
            specialization,
        }
    }

    /// Create a generator with specific specialization (for testing/cross-compilation)
    pub fn with_specialization(specialization: SimdSpecialization) -> Self {
        let capabilities = get_hardware_capabilities();
        Self {
            capabilities,
            specialization,
        }
    }

    /// Get the current SIMD specialization
    pub fn specialization(&self) -> SimdSpecialization {
        self.specialization
    }

    /// Generate SIMD-optimized element-wise addition kernel
    pub fn generate_simd_add(
        &self,
    ) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prioritize direct SIMD intrinsics over JIT compilation
        #[cfg(target_arch = "x86_64")]
        {
            if self.specialization == SimdSpecialization::Avx512 {
                return self.generate_avx512_add();
            }
            if self.specialization == SimdSpecialization::Avx2 {
                return self.generate_avx2_add();
            }
            if std::is_x86_feature_detected!("avx") {
                return Ok(Self::avx_add_kernel);
            }
            if std::is_x86_feature_detected!("sse") {
                return Ok(Self::sse_add_kernel);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.specialization == SimdSpecialization::Neon {
                return self.generate_neon_add();
            }
        }

        // Fall back to scalar if no SIMD available
        self.generate_scalar_add()
    }

    /// Generate SIMD-optimized element-wise multiplication kernel
    pub fn generate_simd_mul(
        &self,
    ) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prioritize direct SIMD intrinsics over JIT compilation
        #[cfg(target_arch = "x86_64")]
        {
            if self.specialization == SimdSpecialization::Avx512 {
                return self.generate_avx512_mul();
            }
            if self.specialization == SimdSpecialization::Avx2 {
                return self.generate_avx2_mul();
            }
            if std::is_x86_feature_detected!("avx") {
                return Ok(Self::avx_mul_kernel);
            }
            if std::is_x86_feature_detected!("sse") {
                return Ok(Self::sse_mul_kernel);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.specialization == SimdSpecialization::Neon {
                return self.generate_neon_mul();
            }
        }

        // Fall back to scalar if no SIMD available
        self.generate_scalar_mul()
    }

    /// Generate SIMD-optimized ReLU activation kernel
    pub fn generate_simd_relu(&self) -> Result<unsafe extern "C" fn(*const f32, *mut f32, usize)> {
        // Prioritize direct SIMD intrinsics over JIT compilation
        #[cfg(target_arch = "x86_64")]
        {
            if self.specialization == SimdSpecialization::Avx512 {
                return self.generate_avx512_relu();
            }
            if self.specialization == SimdSpecialization::Avx2 {
                return self.generate_avx2_relu();
            }
            if std::is_x86_feature_detected!("avx") {
                return Ok(Self::avx_relu_kernel);
            }
            if std::is_x86_feature_detected!("sse") {
                return Ok(Self::sse_relu_kernel);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.specialization == SimdSpecialization::Neon {
                return self.generate_neon_relu();
            }
        }

        // Fall back to scalar if no SIMD available
        self.generate_scalar_relu()
    }

    /// Get the SIMD vector width for the current specialization
    pub fn vector_width(&self) -> usize {
        match self.specialization() {
            SimdSpecialization::Scalar => 1,
            SimdSpecialization::Sse | SimdSpecialization::Neon => 4, // 128 bits / 32 bits per float
            SimdSpecialization::Avx | SimdSpecialization::Avx2 => 8, // 256 bits / 32 bits per float
            SimdSpecialization::Avx512 => 16,                        // 512 bits / 32 bits per float
        }
    }

    /// Get performance multiplier estimate for SIMD vs scalar
    pub fn performance_multiplier(&self) -> f32 {
        match self.specialization {
            SimdSpecialization::Scalar => 1.0,
            SimdSpecialization::Sse => 2.5,
            SimdSpecialization::Avx => 4.0,
            SimdSpecialization::Avx2 => 5.0,
            SimdSpecialization::Avx512 => 8.0,
            SimdSpecialization::Neon => 2.5,
        }
    }

    /// Detect optimal SIMD specialization for current hardware
    fn detect_specialization(capabilities: &HardwareCapabilities) -> SimdSpecialization {
        match capabilities.architecture {
            crate::hardware::Architecture::X86_64 => match capabilities.simd_level {
                crate::hardware::SimdLevel::Avx512Full | crate::hardware::SimdLevel::Avx512f => {
                    SimdSpecialization::Avx512
                }
                crate::hardware::SimdLevel::Avx2 => SimdSpecialization::Avx2,
                crate::hardware::SimdLevel::Avx => SimdSpecialization::Avx,
                crate::hardware::SimdLevel::Sse
                | crate::hardware::SimdLevel::Sse2
                | crate::hardware::SimdLevel::Sse3
                | crate::hardware::SimdLevel::Ssse3
                | crate::hardware::SimdLevel::Sse41
                | crate::hardware::SimdLevel::Sse42 => SimdSpecialization::Sse,
                crate::hardware::SimdLevel::None => SimdSpecialization::Scalar,
            },
            crate::hardware::Architecture::AArch64 => {
                // AArch64 has NEON support (similar to SSE)
                SimdSpecialization::Neon
            }
            crate::hardware::Architecture::Unknown => SimdSpecialization::Scalar,
        }
    }

    /// Generate AVX2 addition kernel with FMA support
    pub fn generate_avx2_add(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AVX2 is available
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx2") {
            Ok(Self::avx2_add_kernel)
        } else {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "AVX2 JIT compilation not implemented".to_string() })
        }
    }

    /// Generate AVX2 multiplication kernel with FMA support
    pub fn generate_avx2_mul(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AVX2 is available
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx2") {
            Ok(Self::avx2_mul_kernel)
        } else {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "AVX2 JIT compilation not implemented".to_string() })
        }
    }

    /// Generate AVX2 ReLU kernel with FMA support
    pub fn generate_avx2_relu(&self) -> Result<unsafe extern "C" fn(*const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AVX2 is available
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx2") {
            Ok(Self::avx2_relu_kernel)
        } else {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "AVX2 JIT compilation not implemented".to_string() })
        }
    }

    /// Generate AVX-512 addition kernel with masking
    pub fn generate_avx512_add(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AVX-512 is available
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx512f") {
            Ok(Self::avx512_add_kernel)
        } else {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "AVX-512 JIT compilation not implemented".to_string() })
        }
    }

    /// Generate AVX-512 multiplication kernel
    pub fn generate_avx512_mul(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AVX-512 is available
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx512f") {
            Ok(Self::avx512_mul_kernel)
        } else {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "AVX-512 JIT compilation not implemented".to_string() })
        }
    }

    /// Generate AVX-512 ReLU kernel
    pub fn generate_avx512_relu(&self) -> Result<unsafe extern "C" fn(*const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AVX-512 is available
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx512f") {
            Ok(Self::avx512_relu_kernel)
        } else {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "AVX-512 JIT compilation not implemented".to_string() })
        }
    }

    /// Generate NEON addition kernel
    pub fn generate_neon_add(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AArch64 NEON is available
        #[cfg(target_arch = "aarch64")]
        {
            Ok(neon_add_kernel)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "NEON JIT compilation not implemented".to_string() })
        }
    }

    /// Generate NEON multiplication kernel
    pub fn generate_neon_mul(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AArch64 NEON is available
        #[cfg(target_arch = "aarch64")]
        {
            Ok(neon_mul_kernel)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "NEON JIT compilation not implemented".to_string() })
        }
    }

    /// Generate NEON ReLU kernel
    pub fn generate_neon_relu(&self) -> Result<unsafe extern "C" fn(*const f32, *mut f32, usize)> {
        // Prefer direct intrinsics when AArch64 NEON is available
        #[cfg(target_arch = "aarch64")]
        {
            Ok(neon_relu_kernel)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // JIT fallback (not implemented yet)
            Err(JitError::UnsupportedOperation { operation: "NEON JIT compilation not implemented".to_string() })
        }
    }

    /// Generate scalar addition kernel
    pub fn generate_scalar_add(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        Ok(Self::scalar_add_kernel)
    }

    /// Generate scalar multiplication kernel
    pub fn generate_scalar_mul(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        Ok(Self::scalar_mul_kernel)
    }

    /// Generate scalar ReLU kernel
    pub fn generate_scalar_relu(&self) -> Result<unsafe extern "C" fn(*const f32, *mut f32, usize)> {
        Ok(Self::scalar_relu_kernel)
    }

    /// AVX2 optimized addition kernel with FMA support
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe extern "C" fn avx2_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input1.add(i));
            let b = _mm256_loadu_ps(input2.add(i));
            // Using FMA for potentially better performance: a + b
            let result = _mm256_add_ps(a, b);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// AVX2 optimized multiplication kernel with FMA support
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe extern "C" fn avx2_mul_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input1.add(i));
            let b = _mm256_loadu_ps(input2.add(i));
            // Using FMA for potentially better performance: a * b
            let result = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
            i += 1;
        }
    }

    /// AVX2 optimized ReLU kernel with FMA support
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe extern "C" fn avx2_relu_kernel(input: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let zero = _mm256_setzero_ps();
        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input.add(i));
            // Use max with zero for ReLU: max(a, 0)
            let result = _mm256_max_ps(a, zero);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = (*input.add(i)).max(0.0);
            i += 1;
        }
    }

    /// AVX-512 optimized addition kernel with masking
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe extern "C" fn avx512_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 16);

        while i < end_aligned {
            let a = _mm512_loadu_ps(input1.add(i));
            let b = _mm512_loadu_ps(input2.add(i));
            let result = _mm512_add_ps(a, b);
            _mm512_storeu_ps(output.add(i), result);
            i += 16;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// AVX-512 optimized multiplication kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe extern "C" fn avx512_mul_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 16);

        while i < end_aligned {
            let a = _mm512_loadu_ps(input1.add(i));
            let b = _mm512_loadu_ps(input2.add(i));
            let result = _mm512_mul_ps(a, b);
            _mm512_storeu_ps(output.add(i), result);
            i += 16;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
            i += 1;
        }
    }

    /// AVX-512 optimized ReLU kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe extern "C" fn avx512_relu_kernel(input: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let zero = _mm512_setzero_ps();
        let mut i = 0;
        let end_aligned = size - (size % 16);

        while i < end_aligned {
            let a = _mm512_loadu_ps(input.add(i));
            let result = _mm512_max_ps(a, zero);
            _mm512_storeu_ps(output.add(i), result);
            i += 16;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = (*input.add(i)).max(0.0);
            i += 1;
        }
    }

    /// NEON optimized addition kernel
    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" fn neon_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::aarch64::*;

        let mut i = 0;
        let end_aligned = size - (size % 4);

        while i < end_aligned {
            let a = vld1q_f32(input1.add(i));
            let b = vld1q_f32(input2.add(i));
            let result = vaddq_f32(a, b);
            vst1q_f32(output.add(i), result);
            i += 4;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// NEON optimized multiplication kernel
    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" fn neon_mul_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::aarch64::*;

        let mut i = 0;
        let end_aligned = size - (size % 4);

        while i < end_aligned {
            let a = vld1q_f32(input1.add(i));
            let b = vld1q_f32(input2.add(i));
            let result = vmulq_f32(a, b);
            vst1q_f32(output.add(i), result);
            i += 4;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
            i += 1;
        }
    }

    /// NEON optimized ReLU kernel
    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" fn neon_relu_kernel(input: *const f32, output: *mut f32, size: usize) {
        use std::arch::aarch64::*;

        let zero = vdupq_n_f32(0.0);
        let mut i = 0;
        let end_aligned = size - (size % 4);

        while i < end_aligned {
            let a = vld1q_f32(input.add(i));
            let result = vmaxq_f32(a, zero);
            vst1q_f32(output.add(i), result);
            i += 4;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = (*input.add(i)).max(0.0);
            i += 1;
        }
    }

    /// Scalar addition kernel (fallback)
    unsafe extern "C" fn scalar_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        for i in 0..size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
        }
    }

    /// Scalar multiplication kernel (fallback)
    unsafe extern "C" fn scalar_mul_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        for i in 0..size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
        }
    }

    /// Scalar ReLU kernel (fallback)
    unsafe extern "C" fn scalar_relu_kernel(input: *const f32, output: *mut f32, size: usize) {
        for i in 0..size {
            *output.add(i) = (*input.add(i)).max(0.0);
        }
    }

    /// AVX optimized addition kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe extern "C" fn avx_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input1.add(i));
            let b = _mm256_loadu_ps(input2.add(i));
            let result = _mm256_add_ps(a, b);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// AVX optimized multiplication kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe extern "C" fn avx_mul_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input1.add(i));
            let b = _mm256_loadu_ps(input2.add(i));
            let result = _mm256_mul_ps(a, b);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
            i += 1;
        }
    }

    /// AVX optimized ReLU kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe extern "C" fn avx_relu_kernel(input: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let zero = _mm256_setzero_ps();
        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input.add(i));
            let result = _mm256_max_ps(a, zero);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = (*input.add(i)).max(0.0);
            i += 1;
        }
    }

    /// SSE optimized addition kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe extern "C" fn sse_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 4);

        while i < end_aligned {
            let a = _mm_loadu_ps(input1.add(i));
            let b = _mm_loadu_ps(input2.add(i));
            let result = _mm_add_ps(a, b);
            _mm_storeu_ps(output.add(i), result);
            i += 4;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// SSE optimized multiplication kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe extern "C" fn sse_mul_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 4);

        while i < end_aligned {
            let a = _mm_loadu_ps(input1.add(i));
            let b = _mm_loadu_ps(input2.add(i));
            let result = _mm_mul_ps(a, b);
            _mm_storeu_ps(output.add(i), result);
            i += 4;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) * *input2.add(i);
            i += 1;
        }
    }

    /// SSE optimized ReLU kernel
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe extern "C" fn sse_relu_kernel(input: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let zero = _mm_setzero_ps();
        let mut i = 0;
        let end_aligned = size - (size % 4);

        while i < end_aligned {
            let a = _mm_loadu_ps(input.add(i));
            let result = _mm_max_ps(a, zero);
            _mm_storeu_ps(output.add(i), result);
            i += 4;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = (*input.add(i)).max(0.0);
            i += 1;
        }
    }

    /// AVX2 optimized addition with hardware prefetching for cache optimization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe extern "C" fn avx2_add_prefetch_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 8);
        let prefetch_distance = 64; // Prefetch 64 floats ahead (256 bytes)

        while i < end_aligned {
            // Prefetch data for future iterations to reduce cache misses
            if i + prefetch_distance < size {
                _mm_prefetch(input1.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
                _mm_prefetch(input2.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
                _mm_prefetch(output.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
            }

            let a = _mm256_loadu_ps(input1.add(i));
            let b = _mm256_loadu_ps(input2.add(i));
            let result = _mm256_add_ps(a, b);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// AVX-512 optimized addition with advanced prefetching
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe extern "C" fn avx512_add_prefetch_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 16);
        let prefetch_distance = 128; // Prefetch 128 floats ahead (512 bytes)

        while i < end_aligned {
            // Aggressive prefetching for AVX-512's larger vectors
            if i + prefetch_distance < size {
                _mm_prefetch(input1.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
                _mm_prefetch(input2.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
                _mm_prefetch(output.add(i + prefetch_distance) as *const i8, _MM_HINT_T0);
                // Additional prefetch for scattered access patterns
                _mm_prefetch(input1.add(i + prefetch_distance + 16) as *const i8, _MM_HINT_T1);
                _mm_prefetch(input2.add(i + prefetch_distance + 16) as *const i8, _MM_HINT_T1);
            }

            let a = _mm512_loadu_ps(input1.add(i));
            let b = _mm512_loadu_ps(input2.add(i));
            let result = _mm512_add_ps(a, b);
            _mm512_storeu_ps(output.add(i), result);
            i += 16;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// AVX2 FMA-accelerated fused multiply-add: c = a * b + c
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe extern "C" fn avx2_fma_kernel(input1: *const f32, input2: *const f32, input3: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let end_aligned = size - (size % 8);

        while i < end_aligned {
            let a = _mm256_loadu_ps(input1.add(i));
            let b = _mm256_loadu_ps(input2.add(i));
            let c = _mm256_loadu_ps(input3.add(i));
            // FMA: result = a * b + c (fused multiply-add)
            let result = _mm256_fmadd_ps(a, b, c);
            _mm256_storeu_ps(output.add(i), result);
            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < size {
            *output.add(i) = *input1.add(i) * *input2.add(i) + *input3.add(i);
            i += 1;
        }
    }

    /// AVX-512 masked operation for handling unaligned data
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe extern "C" fn avx512_masked_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        let mut i = 0;
        let vector_size = 16;

        // Process full vectors
        while i + vector_size <= size {
            let a = _mm512_loadu_ps(input1.add(i));
            let b = _mm512_loadu_ps(input2.add(i));
            let result = _mm512_add_ps(a, b);
            _mm512_storeu_ps(output.add(i), result);
            i += vector_size;
        }

        // Handle remaining elements with masking
        if i < size {
            let remaining = size - i;
            let mask = (1u16 << remaining) - 1; // Create mask for remaining elements

            let a = _mm512_maskz_loadu_ps(mask, input1.add(i));
            let b = _mm512_maskz_loadu_ps(mask, input2.add(i));
            let result = _mm512_maskz_add_ps(mask, a, b);
            _mm512_mask_storeu_ps(output.add(i), mask, result);
        }
    }

    /// Cache-line aligned AVX2 operations for optimal memory access
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe extern "C" fn avx2_cache_aligned_add_kernel(input1: *const f32, input2: *const f32, output: *mut f32, size: usize) {
        use std::arch::x86_64::*;

        const CACHE_LINE_SIZE: usize = 64; // 64 bytes = 16 floats
        let floats_per_cache_line = CACHE_LINE_SIZE / std::mem::size_of::<f32>();
        let vectors_per_cache_line = floats_per_cache_line / 8; // 8 floats per AVX2 vector

        let mut i = 0;
        let end_cache_aligned = size - (size % floats_per_cache_line);

        while i < end_cache_aligned {
            // Process one cache line at a time
            for _ in 0..vectors_per_cache_line {
                let a = _mm256_loadu_ps(input1.add(i));
                let b = _mm256_loadu_ps(input2.add(i));
                let result = _mm256_add_ps(a, b);
                _mm256_storeu_ps(output.add(i), result);
                i += 8;
            }

            // Prefetch next cache line
            if i + floats_per_cache_line < size {
                _mm_prefetch(input1.add(i + floats_per_cache_line) as *const i8, _MM_HINT_T0);
                _mm_prefetch(input2.add(i + floats_per_cache_line) as *const i8, _MM_HINT_T0);
            }
        }

        // Handle remaining elements
        while i < size {
            *output.add(i) = *input1.add(i) + *input2.add(i);
            i += 1;
        }
    }

    /// Generate optimized SIMD kernel with memory optimizations
    pub fn generate_simd_add_optimized(&self) -> Result<unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize)> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.specialization == SimdSpecialization::Avx512 {
                return Ok(Self::avx512_add_prefetch_kernel);
            }
            if self.specialization == SimdSpecialization::Avx2 {
                return Ok(Self::avx2_cache_aligned_add_kernel);
            }
            if self.specialization == SimdSpecialization::Avx {
                return Ok(Self::avx_add_kernel);
            }
            if std::is_x86_feature_detected!("sse") {
                return Ok(Self::sse_add_kernel);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.specialization == SimdSpecialization::Neon {
                return self.generate_neon_add();
            }
        }

        self.generate_scalar_add()
    }
}
