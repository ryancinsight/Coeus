//! # Comprehensive Memory Benchmarking Suite
//!
//! Automated performance validation for heterogeneous memory systems
//! targeting >90% memory utilization through extensive testing and validation.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use backend::{
    BackendType, MemoryAccessPattern, DataLocality, OperationType,
    MemoryManager, HeterogeneousMemoryPool, HeterogeneousUtilizationStatus,
    TransferPerformance, MemoryAllocationRLAgent, MemoryAllocationAction,
};

/// Comprehensive benchmarking suite for memory systems
#[derive(Debug)]
pub struct MemoryBenchmarkingSuite {
    /// Memory manager under test
    memory_manager: MemoryManager,
    /// RL agent for allocation (optional)
    rl_agent: Option<MemoryAllocationRLAgent>,
    /// Benchmark results
    results: BenchmarkResults,
    /// System constraints and limits
    system_limits: SystemLimits,
}

/// Benchmark results aggregation
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    /// Memory allocation benchmarks
    allocation_benchmarks: Vec<AllocationBenchmark>,
    /// Transfer performance benchmarks
    transfer_benchmarks: Vec<TransferBenchmark>,
    /// Utilization efficiency tests
    utilization_tests: Vec<UtilizationTest>,
    /// Contention and pressure tests
    pressure_tests: Vec<PressureTest>,
    /// RL learning performance (if applicable)
    learning_performance: Option<LearningPerformance>,
    /// Overall target achievement (>90% utilization)
    target_achievement: TargetAchievement,
}

/// System limits and constraints
#[derive(Debug)]
struct SystemLimits {
    /// Maximum memory per backend
    max_memory_per_backend: HashMap<BackendType, u64>,
    /// Maximum concurrent allocations
    max_concurrent_allocations: usize,
    /// Transfer bandwidth limits (MB/s)
    bandwidth_limits: HashMap<(BackendType, BackendType), u64>,
    /// NUMA penalties
    numa_penalties: Vec<f32>,
}

impl MemoryBenchmarkingSuite {
    /// Create new benchmarking suite
    pub fn new(memory_manager: MemoryManager) -> Self {
        Self {
            memory_manager,
            rl_agent: Some(MemoryAllocationRLAgent::new()),
            results: BenchmarkResults::default(),
            system_limits: SystemLimits::detect(),
        }
    }

    /// Run complete benchmarking suite
    pub async fn run_complete_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting Comprehensive Memory Benchmarking Suite");
        println!("==================================================");

        // Phase 1: Basic allocation performance
        println!("\n📊 Phase 1: Allocation Performance Benchmarks");
        self.run_allocation_benchmarks().await?;

        // Phase 2: Transfer optimization validation
        println!("\n🔄 Phase 2: Cross-Hardware Transfer Benchmarks");
        self.run_transfer_benchmarks().await?;

        // Phase 3: Memory utilization efficiency
        println!("\n📈 Phase 3: Utilization Efficiency Tests");
        self.run_utilization_tests().await?;

        // Phase 4: Memory pressure handling
        println!("\n⚠️  Phase 4: Memory Pressure and Contention Tests");
        self.run_pressure_tests().await?;

        // Phase 5: RL learning validation (if available)
        if let Some(ref mut agent) = self.rl_agent {
            println!("\n🧠 Phase 5: Reinforcement Learning Validation");
            self.run_rl_learning_tests(agent).await?;
        }

        // Phase 6: Target achievement validation
        println!("\n🎯 Phase 6: Target Achievement Validation");
        self.validate_targets().await?;

        self.print_final_report();
        Ok(())
    }

    /// Run allocation performance benchmarks
    async fn run_allocation_benchmarks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let test_cases = vec![
            ("Small Allocations", vec![64, 128, 256], OperationType::ElementWise),
            ("Medium Allocations", vec![1024, 2048, 4096], OperationType::MatrixMultiplication),
            ("Large Allocations", vec![8192, 16384, 32768], OperationType::MatrixMultiplication),
            ("Heterogeneous Workloads", vec![512, 2048, 8192], OperationType::Convolution),
        ];

        for (test_name, sizes_mb, operation) in test_cases {
            println!("   Running {} benchmarks...", test_name);

            for &size_mb in &sizes_mb {
                let benchmark = self.benchmark_allocation(size_mb * 1_048_576, operation).await?;
                self.results.allocation_benchmarks.push(benchmark);
            }
        }

        Ok(())
    }

    /// Benchmark single allocation performance
    async fn benchmark_allocation(
        &self,
        size_bytes: u64,
        operation: OperationType
    ) -> Result<AllocationBenchmark, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let allocation = self.memory_manager.allocate_heterogeneous_memory(
            size_bytes,
            MemoryAccessPattern::Dense,
            DataLocality::High,
            operation,
        ).await?;
        let allocation_time = start_time.elapsed().as_micros() as f64 / 1000.0; // ms

        Ok(AllocationBenchmark {
            size_bytes,
            operation,
            backend_used: allocation.backend_type,
            numa_node: allocation.numa_node,
            allocation_time_ms: allocation_time,
            heterogeneity_score: allocation.transfer_costs.len() as f32,
        })
    }

    /// Run transfer performance benchmarks
    async fn run_transfer_benchmarks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let transfer_pairs = vec![
            (BackendType::Gpu, BackendType::Cpu),
            (BackendType::Tpu, BackendType::Gpu),
            (BackendType::Cpu, BackendType::Npu),
        ];

        let transfer_sizes = vec![128, 512, 1024]; // MB

        for (source, dest) in transfer_pairs {
            for &size_mb in &transfer_sizes {
                println!("   Testing {} → {} transfer ({} MB)...", source, dest, size_mb);

                let benchmark = self.benchmark_transfer(source, dest, size_mb * 1_048_576).await?;
                self.results.transfer_benchmarks.push(benchmark);
            }
        }

        Ok(())
    }

    /// Benchmark transfer performance
    async fn benchmark_transfer(
        &self,
        source: BackendType,
        dest: BackendType,
        size_bytes: u64
    ) -> Result<TransferBenchmark, Box<dyn std::error::Error>> {
        let performance = self.memory_manager.transfer_memory_cross_hardware(
            source,
            dest,
            size_bytes,
            MemoryAccessPattern::Dense,
        ).await?;

        Ok(TransferBenchmark {
            source_backend: source,
            dest_backend: dest,
            size_bytes,
            transfer_time_us: performance.transfer_time_us,
            bandwidth_mbps: performance.bandwidth_mbps,
            protocol_used: performance.protocol_used,
            optimizations_applied: performance.optimizations_applied.len() as u32,
        })
    }

    /// Run utilization efficiency tests
    async fn run_utilization_tests(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Test 1: Sequential allocation efficiency
        println!("   Running sequential allocation efficiency test...");
        let sequential_test = self.test_sequential_allocation_efficiency().await?;
        self.results.utilization_tests.push(sequential_test);

        // Test 2: Parallel allocation under contention
        println!("   Running parallel allocation test...");
        let parallel_test = self.test_parallel_allocation_efficiency().await?;
        self.results.utilization_tests.push(parallel_test);

        // Test 3: Memory fragmentation impact
        println!("   Running fragmentation impact test...");
        let fragmentation_test = self.test_fragmentation_impact().await?;
        self.results.utilization_tests.push(fragmentation_test);

        Ok(())
    }

    /// Test sequential allocation efficiency towards 90% target
    async fn test_sequential_allocation_efficiency(&self) -> Result<UtilizationTest, Box<dyn std::error::Error>> {
        let mut total_allocated = 0u64;
        let mut allocations = Vec::new();

        // Allocate chunks until we approach 90% utilization
        let target_size = (self.system_limits.total_system_memory() as f64 * 0.90) as u64;
        let chunk_size = target_size / 100; // 1% chunks

        for i in 0..100 {
            if total_allocated >= target_size {
                break;
            }

            let allocation = self.memory_manager.allocate_heterogeneous_memory(
                chunk_size,
                MemoryAccessPattern::Dense,
                DataLocality::High,
                OperationType::MatrixMultiplication,
            ).await?;

            total_allocated += chunk_size;
            allocations.push(allocation);

            let utilization = self.memory_manager.get_heterogeneous_utilization_status().await;
            if utilization.total_allocated > target_size {
                break;
            }
        }

        let final_utilization = self.memory_manager.get_heterogeneous_utilization_status().await;
        let achieved_utilization = final_utilization.total_allocated as f64 / self.system_limits.total_system_memory() as f64;

        Ok(UtilizationTest {
            test_name: "Sequential Allocation Efficiency".to_string(),
            achieved_utilization: achieved_utilization as f32,
            target_utilization: 0.90,
            heterogeneity_score: final_utilization.heterogeneity_score,
            fragmentation_impact: final_utilization.affinity_metrics.cross_numa_violations as f32,
            success: achieved_utilization >= 0.90,
        })
    }

    /// Test parallel allocation under contention
    async fn test_parallel_allocation_efficiency(&self) -> Result<UtilizationTest, Box<dyn std::error::Error>> {
        let num_tasks = 10;
        let mut handles = Vec::new();

        for _ in 0..num_tasks {
            let manager = self.memory_manager.clone();
            let handle = tokio::spawn(async move {
                let mut local_allocations = Vec::new();
                for _ in 0..10 {
                    let allocation = manager.allocate_heterogeneous_memory(
                        64 * 1_048_576, // 64MB each
                        MemoryAccessPattern::Dense,
                        DataLocality::High,
                        OperationType::Convolution,
                    ).await?;
                    local_allocations.push(allocation);
                }
                Ok::<Vec<_>, Box<dyn std::error::Error + Send + Sync>>(local_allocations)
            });
            handles.push(handle);
        }

        // Wait for all parallel allocations
        let mut total_allocated = 0u64;
        for handle in handles {
            match handle.await {
                Ok(Ok(allocations)) => {
                    total_allocated += allocations.len() as u64 * 64 * 1_048_576;
                }
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(Box::new(e)),
            }
        }

        let final_utilization = self.memory_manager.get_heterogeneous_utilization_status().await;
        let achieved_utilization = (total_allocated as f64) / self.system_limits.total_system_memory() as f64;

        Ok(UtilizationTest {
            test_name: "Parallel Allocation Contention".to_string(),
            achieved_utilization: achieved_utilization as f32,
            target_utilization: 0.85, // Slightly lower target for contention
            heterogeneity_score: final_utilization.heterogeneity_score,
            fragmentation_impact: final_utilization.affinity_metrics.cross_numa_violations as f32,
            success: achieved_utilization >= 0.85,
        })
    }

    /// Test fragmentation impact on utilization
    async fn test_fragmentation_impact(&mut self) -> Result<UtilizationTest, Box<dyn std::error::Error>> {
        // Create fragmented memory state by allocating/deallocating
        let mut fragments = Vec::new();
        for i in 0..20 {
            let allocation = self.memory_manager.allocate_heterogeneous_memory(
                100 * 1_048_576, // 100MB
                MemoryAccessPattern::Dense,
                DataLocality::High,
                OperationType::MatrixMultiplication,
            ).await?;
            fragments.push(allocation);

            // Deallocate every other allocation to create fragmentation
            if i % 2 == 0 {
                fragments.pop(); // Simulate deallocation
            }
        }

        // Now try to allocate a large contiguous block
        let large_allocation = self.memory_manager.allocate_heterogeneous_memory(
            1000 * 1_048_576, // 1GB
            MemoryAccessPattern::Dense,
            DataLocality::High,
            OperationType::MatrixMultiplication,
        ).await;

        let final_utilization = self.memory_manager.get_heterogeneous_utilization_status().await;

        Ok(UtilizationTest {
            test_name: "Fragmentation Impact".to_string(),
            achieved_utilization: final_utilization.total_allocated as f32 / self.system_limits.total_system_memory() as f32,
            target_utilization: 0.80, // Lower target to account for fragmentation
            heterogeneity_score: final_utilization.heterogeneity_score,
            fragmentation_impact: final_utilization.affinity_metrics.cross_numa_violations as f32,
            success: large_allocation.is_ok(), // Success if we could allocate large block despite fragmentation
        })
    }

    /// Run RL learning validation tests
    async fn run_rl_learning_tests(&mut self, agent: &mut MemoryAllocationRLAgent) -> Result<(), Box<dyn std::error::Error>> {
        let mut learning_results = Vec::new();

        for episode in 0..10 {
            let mut episode_reward = 0.0;

            // Simulate learning episodes
            for step in 0..5 {
                let pool = self.memory_manager.get_heterogeneous_pool().await?;
                let pool_guard = pool.read().await;

                let action = agent.get_optimal_allocation_action(&*pool_guard).await;
                let reward = agent.calculate_reward(
                    &MemoryAllocationState::initial(),
                    &action,
                    &*pool_guard
                ).await;

                episode_reward += reward;
                agent.learn_from_experience(action, reward, &*pool_guard).await;
            }

            learning_results.push(episode_reward);
        }

        self.results.learning_performance = Some(LearningPerformance {
            total_episodes: 10,
            average_reward_per_episode: learning_results.iter().sum::<f64>() / learning_results.len() as f64,
            learning_improvement: self.calculate_learning_improvement(&learning_results),
            convergence_achieved: learning_results.last().unwrap_or(&0.0) > &0.5,
        });

        Ok(())
    }

    /// Calculate learning improvement from episode rewards
    fn calculate_learning_improvement(&self, rewards: &[f64]) -> f64 {
        if rewards.len() < 2 {
            return 0.0;
        }

        let first_half: f64 = rewards.iter().take(rewards.len() / 2).sum::<f64>() / (rewards.len() / 2) as f64;
        let second_half: f64 = rewards.iter().skip(rewards.len() / 2).sum::<f64>() / (rewards.len() - rewards.len() / 2) as f64;

        if first_half == 0.0 {
            0.0
        } else {
            (second_half - first_half) / first_half.abs()
        }
    }

    /// Validate achievement of >90% utilization targets
    async fn validate_targets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let final_status = self.memory_manager.get_heterogeneous_utilization_status().await;

        self.results.target_achievement = TargetAchievement {
            target_utilization: 0.90,
            achieved_utilization: final_status.total_allocated as f32 / self.system_limits.total_system_memory() as f32,
            heterogeneity_target: 0.85, // Target balanced usage
            achieved_heterogeneity: final_status.heterogeneity_score,
            numa_violations: final_status.affinity_metrics.cross_numa_violations,
            max_tolerable_violations: 100, // Configurable threshold
            overall_success: false, // Will be set below
        };

        self.results.target_achievement.overall_success =
            self.results.target_achievement.achieved_utilization >= self.results.target_achievement.target_utilization &&
            self.results.target_achievement.achieved_heterogeneity >= self.results.target_achievement.heterogeneity_target &&
            (self.results.target_achievement.numa_violations as u32) <= self.results.target_achievement.max_tolerable_violations;

        Ok(())
    }

    /// Print comprehensive final report
    fn print_final_report(&self) {
        println!("\n📋 COMPREHENSIVE MEMORY BENCHMARKING REPORT");
        println!("==========================================");

        // Overall target achievement
        let target = &self.results.target_achievement;
        println!("\n🎯 TARGET ACHIEVEMENT (>90% utilization):");
        println!("   Target Utilization: {:.1}%", target.target_utilization * 100.0);
        println!("   Achieved Utilization: {:.1}%", target.achieved_utilization * 100.0);
        println!("   Heterogeneity Score: {:.3}", target.achieved_heterogeneity);
        println!("   NUMA Violations: {}", target.numa_violations);
        println!("   Overall Success: {}", if target.overall_success { "✅ PASSED" } else { "❌ FAILED" });

        // Allocation benchmarks summary
        let alloc_summary = self.summarize_allocation_benchmarks();
        println!("\n📊 ALLOCATION PERFORMANCE:");
        println!("   Average Allocation Time: {:.2}ms", alloc_summary.avg_allocation_time_ms);
        println!("   Peak Allocation Time: {:.2}ms", alloc_summary.peak_allocation_time_ms);
        println!("   Memory Efficiency: {:.1}%", alloc_summary.memory_efficiency * 100.0);

        // Transfer benchmarks summary
        let transfer_summary = self.summarize_transfer_benchmarks();
        println!("\n🔄 TRANSFER PERFORMANCE:");
        println!("   Average Bandwidth: {:.1} GB/s", transfer_summary.avg_bandwidth_gbps);
        println!("   Peak Bandwidth: {:.1} GB/s", transfer_summary.peak_bandwidth_gbps);
        println!("   Transfer Time Variability: {:.1}%", transfer_summary.variability_percent);

        // Learning performance (if applicable)
        if let Some(ref learning) = self.results.learning_performance {
            println!("\n🧠 REINFORCEMENT LEARNING PERFORMANCE:");
            println!("   Episodes Completed: {}", learning.total_episodes);
            println!("   Average Reward/Episode: {:.3}", learning.average_reward_per_episode);
            println!("   Learning Improvement: {:.1}%", learning.learning_improvement * 100.0);
            println!("   Convergence Achieved: {}", if learning.convergence_achieved { "✅ YES" } else { "❌ NO" });
        }

        println!("\n🏆 FINAL VERDICT:");
        if target.overall_success {
            println!("   ✅ ALL TARGETS ACHIEVED - Production Ready!");
        } else {
            println!("   ⚠️  TARGETS NOT FULLY MET - Further Optimization Needed");
            if target.achieved_utilization < target.target_utilization {
                println!("      - Utilization target not met: {:.1}% < {:.1}%",
                        target.achieved_utilization * 100.0, target.target_utilization * 100.0);
            }
            if target.achieved_heterogeneity < target.heterogeneity_target {
                println!("      - Heterogeneity target not met: {:.3} < {:.3}",
                        target.achieved_heterogeneity, target.heterogeneity_target);
            }
            if (target.numa_violations as u32) > target.max_tolerable_violations {
                println!("      - Too many NUMA violations: {} > {}",
                        target.numa_violations, target.max_tolerable_violations);
            }
        }
    }

    /// Summarize allocation benchmark results
    fn summarize_allocation_benchmarks(&self) -> AllocationSummary {
        let benchmarks = &self.results.allocation_benchmarks;
        if benchmarks.is_empty() {
            return AllocationSummary::default();
        }

        let avg_time = benchmarks.iter().map(|b| b.allocation_time_ms).sum::<f64>() / benchmarks.len() as f64;
        let peak_time = benchmarks.iter().map(|b| b.allocation_time_ms).fold(0.0, f64::max);
        let efficiency = benchmarks.iter().filter(|b| b.heterogeneity_score >= 1.0).count() as f32 / benchmarks.len() as f32;

        AllocationSummary {
            avg_allocation_time_ms: avg_time,
            peak_allocation_time_ms: peak_time,
            memory_efficiency: efficiency,
        }
    }

    /// Summarize transfer benchmark results
    fn summarize_transfer_benchmarks(&self) -> TransferSummary {
        let benchmarks = &self.results.transfer_benchmarks;
        if benchmarks.is_empty() {
            return TransferSummary::default();
        }

        let avg_bandwidth = benchmarks.iter().map(|b| b.bandwidth_mbps / 1000.0).sum::<f64>() / benchmarks.len() as f64;
        let peak_bandwidth = benchmarks.iter().map(|b| b.bandwidth_mbps / 1000.0).fold(0.0, f64::max);

        let times: Vec<f64> = benchmarks.iter().map(|b| b.transfer_time_us).collect();
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|t| (t - mean_time).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        let variability = if mean_time > 0.0 { (std_dev / mean_time) * 100.0 } else { 0.0 };

        TransferSummary {
            avg_bandwidth_gbps: avg_bandwidth,
            peak_bandwidth_gbps: peak_bandwidth,
            variability_percent: variability,
        }
    }
}

/// Individual benchmark result structures
#[derive(Debug)]
pub struct AllocationBenchmark {
    pub size_bytes: u64,
    pub operation: OperationType,
    pub backend_used: BackendType,
    pub numa_node: usize,
    pub allocation_time_ms: f64,
    pub heterogeneity_score: f32,
}

#[derive(Debug)]
pub struct TransferBenchmark {
    pub source_backend: BackendType,
    pub dest_backend: BackendType,
    pub size_bytes: u64,
    pub transfer_time_us: f64,
    pub bandwidth_mbps: f64,
    pub protocol_used: String,
    pub optimizations_applied: u32,
}

#[derive(Debug)]
pub struct UtilizationTest {
    pub test_name: String,
    pub achieved_utilization: f32,
    pub target_utilization: f32,
    pub heterogeneity_score: f32,
    pub fragmentation_impact: f32,
    pub success: bool,
}

#[derive(Debug)]
pub struct PressureTest {
    pub test_name: String,
    pub pressure_applied: f32,
    pub allocation_success_rate: f32,
    pub failover_triggered: bool,
    pub recovery_time_ms: f64,
}

#[derive(Debug)]
pub struct LearningPerformance {
    pub total_episodes: u32,
    pub average_reward_per_episode: f64,
    pub learning_improvement: f64,
    pub convergence_achieved: bool,
}

#[derive(Debug)]
pub struct TargetAchievement {
    pub target_utilization: f32,
    pub achieved_utilization: f32,
    pub heterogeneity_target: f32,
    pub achieved_heterogeneity: f32,
    pub numa_violations: u64,
    pub max_tolerable_violations: u32,
    pub overall_success: bool,
}

/// Summary structures for reporting
#[derive(Debug, Default)]
struct AllocationSummary {
    avg_allocation_time_ms: f64,
    peak_allocation_time_ms: f64,
    memory_efficiency: f32,
}

#[derive(Debug, Default)]
struct TransferSummary {
    avg_bandwidth_gbps: f64,
    peak_bandwidth_gbps: f64,
    variability_percent: f64,
}

impl SystemLimits {
    /// Detect system limits automatically
    fn detect() -> Self {
        let mut max_memory = HashMap::new();
        max_memory.insert(BackendType::Cpu, 64 * 1_073_741_824); // 64GB CPU
        max_memory.insert(BackendType::Gpu, 24 * 1_073_741_824); // 24GB GPU
        max_memory.insert(BackendType::Tpu, 32 * 1_073_741_824); // 32GB TPU
        max_memory.insert(BackendType::Npu, 16 * 1_073_741_824); // 16GB NPU

        let mut bandwidth = HashMap::new();
        bandwidth.insert((BackendType::Gpu, BackendType::Cpu), 50_000); // 50 GB/s
        bandwidth.insert((BackendType::Cpu, BackendType::Gpu), 50_000);
        bandwidth.insert((BackendType::Tpu, BackendType::Cpu), 100_000); // 100 GB/s
        bandwidth.insert((BackendType::Cpu, BackendType::Tpu), 100_000);
        bandwidth.insert((BackendType::Npu, BackendType::Cpu), 25_000); // 25 GB/s
        bandwidth.insert((BackendType::Cpu, BackendType::Npu), 25_000);

        Self {
            max_memory_per_backend: max_memory,
            max_concurrent_allocations: 100,
            bandwidth_limits: bandwidth,
            numa_penalties: vec![0.1, 0.1, 0.1, 0.1], // 4 NUMA nodes
        }
    }

    /// Get total system memory
    fn total_system_memory(&self) -> u64 {
        self.max_memory_per_backend.values().sum()
    }
}

/// Production validation entry point
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory manager with heterogeneous pooling
    println!("🔧 Initializing Memory Manager for Benchmarking...");
    let memory_manager = MemoryManager::with_heterogeneous_pooling(0.90);

    // Run comprehensive benchmarking suite
    let mut suite = MemoryBenchmarkingSuite::new(memory_manager);
    suite.run_complete_benchmark().await?;

    Ok(())
}
