//! # Reinforcement Learning Memory Allocation Agent
//!
//! Implements RL-driven memory allocation policies to achieve >90% memory utilization
//! through learned optimal allocation strategies across heterogeneous hardware.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use rand::Rng;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Import types from parent module
use crate::{BackendType, MemoryAccessPattern};

// Memory allocation and management types
/// Transfer performance metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransferPerformanceRecord {
    pub transfer_size: u64,
    pub duration_micros: u64,
    pub bandwidth_mbps: f64,
    pub success: bool,
}

/// Memory allocator for NUMA-aware allocations
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NUMAAllocator {
    pub numa_nodes: usize,
}

impl NUMAAllocator {
    /// Get NUMA affinity metrics for memory allocations
    pub fn get_affinity_metrics(&self) -> AffinityMetrics {
        // TODO: Implement actual NUMA affinity tracking
        // This would query the system for actual NUMA node allocations
        // and calculate cross-NUMA memory access violations
        AffinityMetrics {
            cross_numa_violations: 0, // Placeholder until NUMA support is implemented
        }
    }
}

/// Affinity metrics for NUMA allocations
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffinityMetrics {
    pub cross_numa_violations: u64,
}

/// Individual memory pool for a specific backend
#[derive(Debug)]
pub struct MemoryPool {
    pub total_memory_bytes: u64,
    pub allocated_memory: u64,
    pub fragmentation_ratio: f32,
}

/// Global memory tracker implementation
#[derive(Debug)]
pub struct GlobalMemoryTracker {
    pub backend_usage: std::collections::HashMap<BackendType, u64>,
    pub numa_pressure: Vec<f32>,
}

impl GlobalMemoryTracker {
    /// Calculate heterogeneity score based on backend utilization variance
    pub fn get_heterogeneity_score(&self) -> f32 {
        if self.backend_usage.is_empty() {
            return 0.0;
        }

        // Calculate mean utilization
        let total_memory: u64 = self.backend_usage.values().sum();
        let total_capacity: u64 = self
            .backend_usage
            .keys()
            .map(|backend| match backend {
                BackendType::Cpu => 4_000_000_000,  // 4GB
                BackendType::Gpu => 8_000_000_000,  // 8GB
                BackendType::Tpu => 16_000_000_000, // 16GB
                BackendType::Npu => 8_000_000_000,  // 8GB
            })
            .sum();

        if total_capacity == 0 {
            return 0.0;
        }

        let mean_utilization = total_memory as f32 / total_capacity as f32;

        // Calculate variance from mean (heterogeneity measure)
        let variance: f32 = self
            .backend_usage
            .values()
            .map(|&used| {
                let utilization = used as f32 / total_capacity as f32;
                (utilization - mean_utilization).powi(2)
            })
            .sum::<f32>()
            / self.backend_usage.len() as f32;

        // Return normalized heterogeneity score (0.0 = perfectly balanced, 1.0 = maximum imbalance)
        variance.sqrt().min(1.0)
    }

    /// Get average utilization across all backends
    pub fn get_average_utilization(&self) -> f32 {
        if self.backend_usage.is_empty() {
            return 0.0;
        }

        let total_used: u64 = self.backend_usage.values().sum();
        let total_capacity: u64 = self
            .backend_usage
            .keys()
            .map(|backend| {
                // TODO: Query actual hardware capacities instead of using defaults
                // This should be configurable and detected at runtime
                match backend {
                    BackendType::Cpu => 4_000_000_000,  // 4GB default
                    BackendType::Gpu => 8_000_000_000,  // 8GB default
                    BackendType::Tpu => 16_000_000_000, // 16GB default
                    BackendType::Npu => 8_000_000_000,  // 8GB default
                }
            })
            .sum();

        if total_capacity == 0 {
            0.0
        } else {
            total_used as f32 / total_capacity as f32
        }
    }
}

/// Heterogeneous memory pool coordinator
#[derive(Debug)]
pub struct HeterogeneousMemoryPool {
    pub memory_pools: std::collections::HashMap<BackendType, MemoryPool>,
    pub global_memory_tracker: GlobalMemoryTracker,
    pub numa_allocator: NUMAAllocator,
    pub transfer_performance_history:
        std::collections::HashMap<String, Vec<TransferPerformanceRecord>>,
}

/// Memory manager for heterogeneous memory allocation and optimization
#[derive(Debug)]
pub struct MemoryManager {
    pub rl_agent: Option<MemoryAllocationRLAgent>,
}

impl MemoryManager {
    /// Get current heterogeneous memory utilization status
    pub async fn get_heterogeneous_utilization_status(&self) -> HeterogeneousUtilizationStatus {
        // TODO: Implement actual heterogeneous memory tracking
        // This should aggregate data from all backend memory pools
        HeterogeneousUtilizationStatus {
            total_allocated: 0,          // TODO: Sum allocations across all backends
            numa_pressure: vec![0.0; 4], // TODO: Query actual NUMA node pressures
            heterogeneity_score: 0.5,    // TODO: Calculate from actual backend utilizations
            transfer_protocol_stats: std::collections::HashMap::new(), // TODO: Track actual transfer stats
            affinity_metrics: AffinityMetrics {
                cross_numa_violations: 0, // TODO: Implement NUMA violation tracking
            },
        }
    }

    /// Get access to the heterogeneous memory pool
    pub async fn get_heterogeneous_pool(&self) -> tokio::sync::RwLock<HeterogeneousMemoryPool> {
        // TODO: Implement actual heterogeneous memory pool management
        // This should maintain a global pool across all backends
        tokio::sync::RwLock::new(HeterogeneousMemoryPool {
            memory_pools: std::collections::HashMap::new(), // TODO: Initialize with actual backend pools
            global_memory_tracker: GlobalMemoryTracker {
                backend_usage: std::collections::HashMap::new(), // TODO: Track actual usage
                numa_pressure: vec![0.0; 4], // TODO: Query actual NUMA pressures
            },
            numa_allocator: NUMAAllocator { numa_nodes: 4 }, // TODO: Detect actual NUMA topology
            transfer_performance_history: std::collections::HashMap::new(), // TODO: Track actual transfers
        })
    }
}

/// Utilization status across heterogeneous memory
#[derive(Debug)]
pub struct HeterogeneousUtilizationStatus {
    pub total_allocated: u64,
    pub numa_pressure: Vec<f32>,
    pub heterogeneity_score: f32,
    pub transfer_protocol_stats: std::collections::HashMap<String, TransferStats>,
    pub affinity_metrics: AffinityMetrics,
}

/// Transfer protocol statistics
#[derive(Debug)]
pub struct TransferStats {
    pub total_transfers: u64,
    pub successful_transfers: u64,
}

/// State representation for memory allocation RL agent
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryAllocationState {
    /// Current memory utilization by backend (0.0-1.0)
    pub backend_utilization: HashMap<BackendType, f32>,
    /// Memory access patterns observed
    pub access_patterns: Vec<MemoryAccessPattern>,
    /// NUMA node pressures (0.0-1.0)
    pub numa_pressures: Vec<f32>,
    /// Recent transfer performance metrics
    pub transfer_history: Vec<TransferPerformanceRecord>,
    /// Heterogeneity score
    pub heterogeneity_score: f32,
    /// Memory fragmentation levels
    pub fragmentation_levels: HashMap<BackendType, f32>,
}

/// Action space for memory allocation decisions
#[derive(Debug, Clone)]
pub enum MemoryAllocationAction {
    /// Allocate to specific backend with NUMA affinity
    AllocateToBackend {
        backend_type: BackendType,
        numa_node: usize,
        allocation_size_mb: u64,
    },
    /// Transfer memory between backends
    TransferBetweenBackends {
        source_backend: BackendType,
        dest_backend: BackendType,
        transfer_size_mb: u64,
    },
    /// Defragment specific backend
    DefragmentBackend { backend_type: BackendType },
    /// Rebalance memory across NUMA nodes
    RebalanceNUMA,
    /// No action (observe only)
    NoAction,
}

/// Reinforcement learning agent for memory allocation optimization
#[derive(Debug)]
pub struct MemoryAllocationRLAgent {
    /// Current state of memory system
    current_state: MemoryAllocationState,
    /// Q-learning value function (state-action values)
    q_table: HashMap<String, HashMap<String, f64>>,
    /// Learning rate (alpha)
    learning_rate: f64,
    /// Discount factor (gamma)
    discount_factor: f64,
    /// Exploration rate (epsilon)
    exploration_rate: f64,
    /// Exploration decay rate
    exploration_decay: f64,
    /// Minimum exploration rate
    min_exploration_rate: f64,
    /// Reward function parameters
    reward_weights: RewardWeights,
    /// Experience replay buffer
    experience_buffer: Vec<Experience>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Training iterations
    training_steps: usize,
}

/// Experience tuple for replay learning
#[derive(Debug, Clone)]
struct Experience {
    state: MemoryAllocationState,
    action: MemoryAllocationAction,
    reward: f64,
    next_state: MemoryAllocationState,
    done: bool,
}

/// Reward function parameters
#[derive(Debug)]
struct RewardWeights {
    /// Weight for utilization efficiency (0.0-1.0 target)
    utilization_weight: f64,
    /// Weight for heterogeneity score (-1.0 to 1.0)
    heterogeneity_weight: f64,
    /// Weight for transfer efficiency
    transfer_efficiency_weight: f64,
    /// Weight for fragmentation reduction
    fragmentation_weight: f64,
    /// Penalty for cross-NUMA violations
    numa_penalty_weight: f64,
    /// Weight for memory pressure reduction
    pressure_reduction_weight: f64,
}

impl MemoryAllocationRLAgent {
    /// Create new RL agent for memory allocation
    pub fn new() -> Self {
        Self {
            current_state: MemoryAllocationState::initial(),
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.95,
            exploration_rate: 1.0,
            exploration_decay: 0.995,
            min_exploration_rate: 0.01,
            reward_weights: RewardWeights {
                utilization_weight: 1.0,
                heterogeneity_weight: 0.8,
                transfer_efficiency_weight: 0.6,
                fragmentation_weight: 0.4,
                numa_penalty_weight: -0.3,
                pressure_reduction_weight: 0.5,
            },
            experience_buffer: Vec::new(),
            max_buffer_size: 1000,
            training_steps: 0,
        }
    }
}

impl Default for MemoryAllocationRLAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAllocationRLAgent {
    /// Get optimal memory allocation action for current state
    pub async fn get_optimal_allocation_action(
        &mut self,
        pool: &HeterogeneousMemoryPool,
    ) -> MemoryAllocationAction {
        // Update current state from pool
        self.update_state_from_pool(pool).await;

        // Exploration vs exploitation
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.exploration_rate {
            // Explore: random action
            self.explore_random_action()
        } else {
            // Exploit: best action from Q-table
            self.exploit_best_action()
        }
    }

    /// Learn from experience (Q-learning update)
    pub async fn learn_from_experience(
        &mut self,
        action: MemoryAllocationAction,
        reward: f64,
        next_pool_state: &HeterogeneousMemoryPool,
    ) {
        let previous_state = self.current_state.clone();
        self.update_state_from_pool(next_pool_state).await;
        let next_state_ref = &self.current_state;

        let state_key = self.state_to_key(&previous_state);
        let action_key = self.action_to_key(&action);

        // Get next best Q-value before modifying q_table
        let next_best_q = self.get_best_q_value(next_state_ref);

        // Initialize Q-value if not exists and update it
        let q_value = self
            .q_table
            .entry(state_key.clone())
            .or_default()
            .entry(action_key)
            .or_insert(0.0);

        // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
        let current_q = *q_value;
        let new_q_value = current_q
            + self.learning_rate * (reward + self.discount_factor * next_best_q - current_q);

        *q_value = new_q_value;

        // Check if episode should terminate (critical thresholds exceeded)
        let done = self.should_terminate_episode(next_state_ref);

        // Store experience for replay
        let experience = Experience {
            state: previous_state,
            action,
            reward,
            next_state: next_state_ref.clone(),
            done,
        };
        self.experience_buffer.push(experience);

        // Keep buffer size limited
        if self.experience_buffer.len() > self.max_buffer_size {
            self.experience_buffer.remove(0);
        }

        // Decay exploration rate
        self.exploration_rate =
            (self.exploration_rate * self.exploration_decay).max(self.min_exploration_rate);

        self.training_steps += 1;

        // Periodic replay learning
        if self.training_steps % 10 == 0 {
            self.replay_experiences();
        }
    }

    /// Check if episode should terminate based on critical thresholds
    fn should_terminate_episode(&self, state: &MemoryAllocationState) -> bool {
        // Episode terminates if critical system conditions are met
        let avg_pressure =
            state.numa_pressures.iter().sum::<f32>() / state.numa_pressures.len().max(1) as f32;

        // Calculate average utilization across backends
        let avg_utilization = if state.backend_utilization.is_empty() {
            0.0
        } else {
            state.backend_utilization.values().sum::<f32>() / state.backend_utilization.len() as f32
        };

        // Calculate average fragmentation
        let avg_fragmentation = if state.fragmentation_levels.is_empty() {
            0.0
        } else {
            state.fragmentation_levels.values().sum::<f32>()
                / state.fragmentation_levels.len() as f32
        };

        // Critical episode termination conditions (using reasonable hardcoded thresholds)
        // These could be made configurable in the future
        avg_pressure > 0.95 ||  // Critical NUMA pressure
        avg_utilization > 0.95 ||  // Critical utilization
        avg_fragmentation > 0.9 // Critical fragmentation
    }

    /// Calculate reward for state transition
    pub async fn calculate_reward(
        &self,
        previous_state: &MemoryAllocationState,
        action: &MemoryAllocationAction,
        new_pool_state: &HeterogeneousMemoryPool,
    ) -> f64 {
        let mut reward = 0.0;

        // Utilization efficiency reward (closer to 90% target = better)
        let avg_utilization = new_pool_state
            .global_memory_tracker
            .get_average_utilization() as f64;
        let utilization_distance = (avg_utilization - 0.90).abs();
        reward += self.reward_weights.utilization_weight * (1.0 - utilization_distance);

        // Heterogeneity reward (higher score = better)
        let heterogeneity_improvement = new_pool_state
            .global_memory_tracker
            .get_heterogeneity_score()
            - previous_state.heterogeneity_score;
        reward += self.reward_weights.heterogeneity_weight * heterogeneity_improvement as f64;

        // Transfer efficiency reward (for transfer actions)
        if let MemoryAllocationAction::TransferBetweenBackends { .. } = action {
            // TODO: Calculate actual transfer efficiency based on:
            // - Transfer bandwidth achieved vs theoretical maximum
            // - Transfer latency vs expected latency
            // - Memory access pattern compatibility
            // For now, provide a small positive reward for attempting transfers
            reward += self.reward_weights.transfer_efficiency_weight * 0.1;
        }

        // Fragmentation reduction reward
        let avg_fragmentation_before = previous_state.fragmentation_levels.values().sum::<f32>()
            / previous_state.fragmentation_levels.len().max(1) as f32;
        let avg_fragmentation_after = new_pool_state
            .memory_pools
            .values()
            .map(|pool| pool.fragmentation_ratio)
            .sum::<f32>()
            / new_pool_state.memory_pools.len().max(1) as f32;
        let fragmentation_improvement = avg_fragmentation_before - avg_fragmentation_after;
        reward += self.reward_weights.fragmentation_weight * fragmentation_improvement as f64;

        // NUMA penalty (cross-NUMA violations = bad)
        let numa_violations = new_pool_state
            .numa_allocator
            .get_affinity_metrics()
            .cross_numa_violations;
        reward += self.reward_weights.numa_penalty_weight * (numa_violations as f64 * -0.01);

        // Memory pressure reduction reward
        let avg_pressure_before = previous_state.numa_pressures.iter().sum::<f32>()
            / previous_state.numa_pressures.len().max(1) as f32;
        let avg_pressure_after = new_pool_state
            .global_memory_tracker
            .numa_pressure
            .iter()
            .sum::<f32>()
            / new_pool_state
                .global_memory_tracker
                .numa_pressure
                .len()
                .max(1) as f32;
        let pressure_improvement = avg_pressure_before - avg_pressure_after;
        reward += self.reward_weights.pressure_reduction_weight * pressure_improvement as f64;

        reward
    }

    /// Get action space for current state
    fn get_action_space(&self) -> Vec<MemoryAllocationAction> {
        let mut actions = Vec::new();

        // Allocation actions to different backends/NUMA combinations
        for &backend_type in &[
            BackendType::Cpu,
            BackendType::Gpu,
            BackendType::Cpu,
            BackendType::Npu,
        ] {
            for numa_node in 0..4 {
                // TODO: Query actual NUMA topology instead of assuming 4 nodes
                for &size_mb in &[64, 128, 256, 512, 1024] {
                    // Common allocation sizes
                    actions.push(MemoryAllocationAction::AllocateToBackend {
                        backend_type,
                        numa_node,
                        allocation_size_mb: size_mb,
                    });
                }
            }
        }

        // Transfer actions between backends
        for &source in &[
            BackendType::Cpu,
            BackendType::Gpu,
            BackendType::Cpu,
            BackendType::Npu,
        ] {
            for &dest in &[
                BackendType::Cpu,
                BackendType::Gpu,
                BackendType::Cpu,
                BackendType::Npu,
            ] {
                if source != dest {
                    for &size_mb in &[128, 256, 512] {
                        actions.push(MemoryAllocationAction::TransferBetweenBackends {
                            source_backend: source,
                            dest_backend: dest,
                            transfer_size_mb: size_mb,
                        });
                    }
                }
            }
        }

        // Defragmentation actions
        for &backend_type in &[BackendType::Gpu, BackendType::Cpu, BackendType::Npu] {
            actions.push(MemoryAllocationAction::DefragmentBackend { backend_type });
        }

        // Rebalancing action
        actions.push(MemoryAllocationAction::RebalanceNUMA);

        // No action
        actions.push(MemoryAllocationAction::NoAction);

        actions
    }

    /// Choose random action for exploration
    fn explore_random_action(&self) -> MemoryAllocationAction {
        let actions = self.get_action_space();
        let mut rng = rand::thread_rng();
        let random_idx = rng.gen_range(0..actions.len());
        actions[random_idx].clone()
    }

    /// Choose best action from Q-table for exploitation
    fn exploit_best_action(&self) -> MemoryAllocationAction {
        let actions = self.get_action_space();
        let state_key = self.state_to_key(&self.current_state);

        let mut best_action = MemoryAllocationAction::NoAction;
        let mut best_q_value = f64::NEG_INFINITY;

        for action in actions {
            let action_key = self.action_to_key(&action);
            let q_value = self
                .q_table
                .get(&state_key)
                .and_then(|state_actions| state_actions.get(&action_key))
                .copied()
                .unwrap_or(0.0);

            if q_value > best_q_value {
                best_q_value = q_value;
                best_action = action;
            }
        }

        best_action
    }

    /// Get best Q-value for a state
    fn get_best_q_value(&self, state: &MemoryAllocationState) -> f64 {
        let state_key = self.state_to_key(state);
        let actions = self.get_action_space();

        let mut best_q: f64 = 0.0;
        for action in actions {
            let action_key = self.action_to_key(&action);
            if let Some(q_value) = self
                .q_table
                .get(&state_key)
                .and_then(|state_actions| state_actions.get(&action_key))
            {
                best_q = best_q.max(*q_value);
            }
        }
        best_q
    }

    /// Convert state to string key for Q-table
    fn state_to_key(&self, state: &MemoryAllocationState) -> String {
        // Create a discretized state representation
        let mut key_parts = Vec::new();

        // Discretize backend utilization (0-10 levels)
        for (backend, utilization) in &state.backend_utilization {
            let discretized = ((utilization * 10.0) as i32).clamp(0, 10);
            key_parts.push(format!("{}_{}", backend_to_str(*backend), discretized));
        }

        // Discretize heterogeneity score
        let hetero_discrete = ((state.heterogeneity_score * 10.0) as i32).clamp(0, 10);
        key_parts.push(format!("hetero_{}", hetero_discrete));

        // Discretize average NUMA pressure
        let avg_pressure =
            state.numa_pressures.iter().sum::<f32>() / state.numa_pressures.len().max(1) as f32;
        let pressure_discrete = ((avg_pressure * 10.0) as i32).clamp(0, 10);
        key_parts.push(format!("pressure_{}", pressure_discrete));

        key_parts.join("|")
    }

    /// Convert action to string key
    fn action_to_key(&self, action: &MemoryAllocationAction) -> String {
        match action {
            MemoryAllocationAction::AllocateToBackend {
                backend_type,
                numa_node,
                allocation_size_mb,
            } => {
                format!(
                    "alloc_{}_{}_{}",
                    backend_to_str(*backend_type),
                    numa_node,
                    allocation_size_mb
                )
            }
            MemoryAllocationAction::TransferBetweenBackends {
                source_backend,
                dest_backend,
                transfer_size_mb,
            } => {
                format!(
                    "transfer_{}_{}_{}",
                    backend_to_str(*source_backend),
                    backend_to_str(*dest_backend),
                    transfer_size_mb
                )
            }
            MemoryAllocationAction::DefragmentBackend { backend_type } => {
                format!("defrag_{}", backend_to_str(*backend_type))
            }
            MemoryAllocationAction::RebalanceNUMA => "rebalance_numa".to_string(),
            MemoryAllocationAction::NoAction => "no_action".to_string(),
        }
    }

    /// Update agent state from current pool status
    async fn update_state_from_pool(&mut self, pool: &HeterogeneousMemoryPool) {
        let mut backend_utilization = HashMap::new();
        for (backend_type, memory_pool) in &pool.memory_pools {
            let utilization =
                memory_pool.allocated_memory as f32 / memory_pool.total_memory_bytes.max(1) as f32;
            backend_utilization.insert(*backend_type, utilization);
        }

        let fragmentation_levels = pool
            .memory_pools
            .iter()
            .map(|(backend, pool)| (*backend, pool.fragmentation_ratio))
            .collect();

        self.current_state = MemoryAllocationState {
            backend_utilization,
            access_patterns: vec![MemoryAccessPattern::Dense], // TODO: Analyze actual access patterns from workload
            numa_pressures: pool.global_memory_tracker.numa_pressure.clone(),
            transfer_history: pool
                .transfer_performance_history
                .values()
                .flatten()
                .take(10)
                .cloned()
                .collect(),
            heterogeneity_score: pool.global_memory_tracker.get_heterogeneity_score(),
            fragmentation_levels,
        };
    }

    /// Experience replay learning
    fn replay_experiences(&mut self) {
        if self.experience_buffer.len() < 32 {
            return; // Not enough experiences
        }

        // Sample random experiences for replay
        let mut rng = rand::thread_rng();
        for _ in 0..32 {
            let idx = rng.gen_range(0..self.experience_buffer.len());
            let experience = &self.experience_buffer[idx];

            let state_key = self.state_to_key(&experience.state);
            let action_key = self.action_to_key(&experience.action);

            // Get next best Q-value first to avoid borrow checker issues
            let next_best_q = if experience.done {
                0.0 // Terminal state has no future reward
            } else {
                self.get_best_q_value(&experience.next_state)
            };

            // Now get and update Q-value
            if let Some(q_value) = self
                .q_table
                .entry(state_key)
                .or_default()
                .get_mut(&action_key)
            {
                let target = experience.reward + self.discount_factor * next_best_q;
                let error = target - *q_value;
                *q_value += self.learning_rate * error;
            }
        }
    }
}

impl MemoryAllocationState {
    /// Create initial state
    fn initial() -> Self {
        Self {
            backend_utilization: HashMap::new(),
            access_patterns: Vec::new(),
            numa_pressures: vec![0.0; 4],
            transfer_history: Vec::new(),
            heterogeneity_score: 0.0,
            fragmentation_levels: HashMap::new(),
        }
    }
}

/// Convert BackendType to string for serialization
fn backend_to_str(backend: BackendType) -> &'static str {
    match backend {
        BackendType::Cpu => "cpu",
        BackendType::Gpu => "gpu",
        BackendType::Tpu => "tpu",
        BackendType::Npu => "npu",
    }
}

// ============================================================================
// PRODUCTION MONITORING AND ALERTING
// ============================================================================

/// Production monitoring system for >90% utilization tracking and alerting
pub struct ProductionMemoryMonitor {
    /// Memory manager being monitored
    memory_manager: Arc<MemoryManager>,
    /// Alert thresholds and rules
    alert_rules: AlertRules,
    /// Current alerts active
    active_alerts: Vec<MemoryAlert>,
    /// Historical metrics for trending
    metrics_history: VecDeque<MetricsSnapshot>,
    /// Utilization dashboard
    dashboard: UtilizationDashboard,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Alert callbacks for external systems
    alert_callbacks: Vec<Box<dyn AlertCallback + Send + Sync>>,
}

/// Comprehensive alerting rules for memory system
#[derive(Debug)]
pub struct AlertRules {
    /// Utilization alerts (>90% target tracking)
    utilization_alerts: UtilizationAlerts,
    /// Memory pressure alerts
    pressure_alerts: PressureAlerts,
    /// Fragmentation alerts
    fragmentation_alerts: FragmentationAlerts,
    /// Transfer performance alerts
    transfer_alerts: TransferAlerts,
    /// RL learning alerts (for automated systems)
    learning_alerts: LearningAlerts,
}

/// Utilization-based alerts (>90% target focused)
#[derive(Debug)]
pub struct UtilizationAlerts {
    /// Target utilization threshold (90%)
    pub target_utilization: f32,
    /// Warning threshold (85%)
    pub warning_threshold: f32,
    /// Critical threshold (95%)
    pub critical_threshold: f32,
    /// Under-utilization warning (50%)
    pub under_utilization_threshold: f32,
    /// Heterogeneity imbalance threshold
    pub heterogeneity_imbalance_threshold: f32,
}

/// Memory pressure alerting rules
#[derive(Debug)]
pub struct PressureAlerts {
    /// High pressure threshold (80%)
    pub high_pressure_threshold: f32,
    /// Critical pressure threshold (95%)
    pub critical_pressure_threshold: f32,
    /// NUMA violation rate threshold
    pub numa_violation_rate_threshold: u32,
    /// Sustained pressure duration (seconds)
    pub sustained_pressure_duration_secs: u64,
}

/// Fragmentation monitoring alerts
#[derive(Debug)]
pub struct FragmentationAlerts {
    /// High fragmentation threshold (0.5)
    pub high_fragmentation_threshold: f32,
    /// Critical fragmentation threshold (0.7)
    pub critical_fragmentation_threshold: f32,
    /// Defragmentation failure threshold
    pub defragmentation_failure_threshold: u32,
}

/// Transfer performance alerts
#[derive(Debug)]
pub struct TransferAlerts {
    /// Bandwidth degradation threshold (% drop from baseline)
    pub bandwidth_degradation_threshold: f32,
    /// Transfer failure rate threshold (%)
    pub transfer_failure_rate_threshold: f32,
    /// Protocol optimization failure threshold
    pub optimization_failure_threshold: u32,
}

/// RL learning system alerts
#[derive(Debug)]
pub struct LearningAlerts {
    /// Learning convergence failure threshold
    pub convergence_failure_threshold: f64,
    /// Reward degradation threshold
    pub reward_degradation_threshold: f64,
    /// Decision quality drop threshold
    pub decision_quality_threshold: f32,
}

/// Alert instance with severity and metadata
#[derive(Debug, Clone)]
pub struct MemoryAlert {
    /// Alert ID for tracking
    pub id: String,
    /// Alert level
    pub level: AlertLevel,
    /// Alert type categorization
    pub alert_type: AlertType,
    /// Human-readable message
    pub message: String,
    /// Detailed description
    pub description: String,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Timestamp when alert was triggered
    pub timestamp: u64,
    /// Metrics snapshot when alert triggered
    pub metrics_snapshot: MetricsSnapshot,
    /// Alert-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AlertLevel {
    /// Info-level alerts (no action required)
    Info,
    /// Warning-level alerts (monitor closely)
    Warning,
    /// High-severity alerts (action recommended)
    High,
    /// Critical alerts (immediate action required)
    Critical,
    /// Emergency alerts (system impact)
    Emergency,
}

/// Categorization of alert types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    /// Utilization-related alerts
    Utilization,
    /// Memory pressure alerts
    Pressure,
    /// Fragmentation alerts
    Fragmentation,
    /// Transfer performance alerts
    Transfer,
    /// Learning system alerts
    Learning,
    /// System health alerts
    SystemHealth,
}

/// Snapshot of all memory metrics at a point in time
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetricsSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Overall system utilization (0.0-1.0)
    pub system_utilization: f32,
    /// Backend-specific utilization
    pub backend_utilization: HashMap<BackendType, f32>,
    /// NUMA node pressures
    pub numa_pressures: Vec<f32>,
    /// Average fragmentation
    pub avg_fragmentation: f32,
    /// Heterogeneity score
    pub heterogeneity_score: f32,
    /// Active transfer count
    pub active_transfers: usize,
    /// NUMA violations (cumulative)
    pub numa_violations: u64,
    /// Transfer failures (cumulative)
    pub transfer_failures: u64,
}

/// Real-time utilization dashboard
#[derive(Debug)]
pub struct UtilizationDashboard {
    /// Current metrics display
    current_metrics: MetricsSnapshot,
    /// Trend analysis over time
    trends: TrendAnalysis,
    /// Efficiency ratings per backend
    backend_efficiency: HashMap<BackendType, EfficiencyRating>,
    /// Threshold breaches tracking
    threshold_breaches: Vec<ThresholdBreach>,
    /// Performance predictions
    predictions: PerformancePredictions,
}

/// Trend analysis for key metrics
#[derive(Debug)]
pub struct TrendAnalysis {
    /// Utilization trend over last hour
    utilization_trend: MetricTrend,
    /// Pressure trend over last hour
    pressure_trend: MetricTrend,
    /// Heterogeneity trend over last hour
    heterogeneity_trend: MetricTrend,
    /// Transfer performance trend
    transfer_trend: MetricTrend,
}

impl TrendAnalysis {
    /// Get utilization trend
    pub fn utilization_trend(&self) -> &MetricTrend {
        &self.utilization_trend
    }

    /// Get pressure trend
    pub fn pressure_trend(&self) -> &MetricTrend {
        &self.pressure_trend
    }

    /// Get heterogeneity trend
    pub fn heterogeneity_trend(&self) -> &MetricTrend {
        &self.heterogeneity_trend
    }

    /// Get transfer trend
    pub fn transfer_trend(&self) -> &MetricTrend {
        &self.transfer_trend
    }

    /// Update utilization trend
    pub fn set_utilization_trend(&mut self, trend: MetricTrend) {
        self.utilization_trend = trend;
    }

    /// Update pressure trend
    pub fn set_pressure_trend(&mut self, trend: MetricTrend) {
        self.pressure_trend = trend;
    }

    /// Update heterogeneity trend
    pub fn set_heterogeneity_trend(&mut self, trend: MetricTrend) {
        self.heterogeneity_trend = trend;
    }

    /// Update transfer trend
    pub fn set_transfer_trend(&mut self, trend: MetricTrend) {
        self.transfer_trend = trend;
    }
}

/// Metric trend with direction and rate
#[derive(Debug)]
pub struct MetricTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend rate (change per minute)
    pub rate: f64,
    /// Confidence in trend (0.0-1.0)
    pub confidence: f32,
    /// Prediction for next hour
    pub prediction_1h: f64,
}

/// Trend direction indicators
#[derive(Debug)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Degrading trend
    Degrading,
    /// Volatile (unpredictable changes)
    Volatile,
}

/// Backend efficiency ratings
#[derive(Debug)]
pub struct EfficiencyRating {
    /// Utilization efficiency (0.0-1.0)
    pub utilization_efficiency: f32,
    /// Allocation speed rating
    pub allocation_speed: AllocationSpeedRating,
    /// Transfer efficiency rating
    pub transfer_efficiency: TransferEfficiencyRating,
    /// Overall health score
    pub health_score: f32,
}

/// Allocation speed performance ratings
#[derive(Debug)]
pub enum AllocationSpeedRating {
    Excellent,
    Good,
    Average,
    Poor,
    Critical,
}

/// Transfer efficiency performance ratings
#[derive(Debug)]
pub enum TransferEfficiencyRating {
    Excellent,
    Good,
    Average,
    Poor,
    Critical,
}

/// Threshold breach tracking
#[derive(Debug)]
pub struct ThresholdBreach {
    /// Metric that breached threshold
    pub metric: String,
    /// Breached threshold value
    pub threshold_value: f64,
    /// Actual value at breach
    pub actual_value: f64,
    /// Breach duration (seconds)
    pub duration_secs: u64,
    /// Timestamp of breach start
    pub start_timestamp: u64,
    /// Timestamp of breach end (if resolved)
    pub end_timestamp: Option<u64>,
    /// Breach severity
    pub severity: AlertLevel,
}

/// Performance predictions for proactive alerting
#[derive(Debug)]
pub struct PerformancePredictions {
    /// Predicted utilization in 1 hour
    pub utilization_1h: f64,
    /// Risk of hitting critical thresholds within 1 hour
    pub critical_risk_1h: f32,
    /// Predicted heterogeneity score degradation
    pub heterogeneity_degradation: f64,
    /// Memory exhaustion risk (0.0-1.0)
    pub exhaustion_risk: f32,
}

/// Monitoring configuration
#[derive(Debug)]
pub struct MonitoringConfig {
    /// Monitoring interval (seconds)
    pub monitoring_interval_secs: u64,
    /// Metrics retention period (hours)
    pub metrics_retention_hours: u64,
    /// Alert persistence period (seconds)
    pub alert_persistence_secs: u64,
    /// Dashboard update frequency (seconds)
    pub dashboard_update_freq_secs: u64,
    /// Performance prediction lookahead (hours)
    pub prediction_lookahead_hours: u64,
}

/// Alert callback trait for external integrations
pub trait AlertCallback: Send + Sync {
    /// Called when alert is triggered
    fn on_alert(&self, alert: &MemoryAlert);

    /// Called when alert is resolved
    fn on_alert_resolved(&self, alert_id: &str);

    /// Called for periodic health checks
    fn on_health_check(&self, status: &HealthStatus);
}

/// Overall system health status
#[derive(Debug)]
pub struct HealthStatus {
    /// Overall health score (0.0-1.0)
    pub overall_score: f32,
    /// Active alerts count by level
    pub active_alerts_by_level: HashMap<AlertLevel, usize>,
    /// Components health status
    pub component_health: HashMap<String, ComponentHealth>,
    /// Last health check timestamp
    pub last_check_timestamp: u64,
}

/// Individual component health
#[derive(Debug)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Health score (0.0-1.0)
    pub health_score: f32,
    /// Status indicator
    pub status: HealthStatusIndicator,
    /// Last update timestamp
    pub last_update: u64,
    /// Issue description (if any)
    pub issues: Vec<String>,
}

/// Component status indicators
#[derive(Debug)]
pub enum HealthStatusIndicator {
    /// Fully healthy
    Healthy,
    /// Minor issues, monitoring needed
    Warning,
    /// Significant issues, action needed
    Degraded,
    /// Critical issues, immediate action required
    Critical,
    /// Component failure
    Failed,
}

impl ProductionMemoryMonitor {
    /// Create new production monitor
    pub fn new(memory_manager: Arc<MemoryManager>) -> Self {
        Self {
            memory_manager,
            alert_rules: AlertRules::default(),
            active_alerts: Vec::new(),
            metrics_history: VecDeque::with_capacity(3600), // 1 hour at 1Hz
            dashboard: UtilizationDashboard::new(),
            config: MonitoringConfig::default(),
            alert_callbacks: Vec::new(),
        }
    }

    /// Add alert callback for external systems
    pub fn add_alert_callback(&mut self, callback: Box<dyn AlertCallback + Send + Sync>) {
        self.alert_callbacks.push(callback);
    }

    /// Collect current metrics snapshot
    pub async fn collect_metrics(&mut self) -> Result<MetricsSnapshot, crate::BackendError> {
        let status = self
            .memory_manager
            .get_heterogeneous_utilization_status()
            .await;
        let pool = self.memory_manager.get_heterogeneous_pool().await;
        let pool_guard = pool.read().await;

        let backend_utilization = pool_guard
            .memory_pools
            .iter()
            .map(|(backend, memory_pool)| {
                let utilization = memory_pool.allocated_memory as f32
                    / memory_pool.total_memory_bytes.max(1) as f32;
                (*backend, utilization)
            })
            .collect();

        let avg_fragmentation = pool_guard
            .memory_pools
            .values()
            .map(|pool| pool.fragmentation_ratio)
            .sum::<f32>()
            / pool_guard.memory_pools.len().max(1) as f32;

        let snapshot = MetricsSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            system_utilization: status.total_allocated as f32
                / pool_guard
                    .global_memory_tracker
                    .get_average_utilization()
                    .max(0.001),
            backend_utilization,
            numa_pressures: status.numa_pressure.clone(),
            avg_fragmentation,
            heterogeneity_score: status.heterogeneity_score,
            active_transfers: status.transfer_protocol_stats.len(),
            numa_violations: status.affinity_metrics.cross_numa_violations,
            transfer_failures: status
                .transfer_protocol_stats
                .values()
                .map(|stats| {
                    stats
                        .total_transfers
                        .saturating_sub(stats.successful_transfers)
                })
                .sum(),
        };

        // Store in history
        self.metrics_history.push_back(snapshot.clone());
        if self.metrics_history.len() > (self.config.metrics_retention_hours * 3600) as usize {
            self.metrics_history.pop_front();
        }

        // Update dashboard
        self.dashboard.current_metrics = snapshot.clone();
        self.update_trend_analysis().await;

        Ok(snapshot)
    }

    /// Update trend analysis in dashboard
    async fn update_trend_analysis(&mut self) {
        if self.metrics_history.len() < 2 {
            return;
        }

        // Analyze last hour of data (assuming 1Hz collection)
        let window_size = std::cmp::min(3600, self.metrics_history.len());
        let recent_history: Vec<&MetricsSnapshot> = self
            .metrics_history
            .iter()
            .rev()
            .take(window_size)
            .collect();

        self.dashboard.trends = TrendAnalysis {
            utilization_trend: Self::analyze_trend(
                recent_history
                    .iter()
                    .rev()
                    .map(|m| m.system_utilization as f64)
                    .collect(),
            ),
            pressure_trend: Self::analyze_trend(
                recent_history
                    .iter()
                    .rev()
                    .map(|m| {
                        m.numa_pressures.iter().sum::<f32>() as f64 / m.numa_pressures.len() as f64
                    })
                    .collect(),
            ),
            heterogeneity_trend: Self::analyze_trend(
                recent_history
                    .iter()
                    .rev()
                    .map(|m| m.heterogeneity_score as f64)
                    .collect(),
            ),
            transfer_trend: Self::analyze_trend(
                recent_history
                    .iter()
                    .rev()
                    .map(|m| m.active_transfers as f64)
                    .collect(),
            ),
        };
    }

    /// Analyze trend for a series of values
    fn analyze_trend(values: Vec<f64>) -> MetricTrend {
        if values.len() < 2 {
            return MetricTrend {
                direction: TrendDirection::Stable,
                rate: 0.0,
                confidence: 0.0,
                prediction_1h: 0.0,
            };
        }

        // Simple linear regression for trend
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let xx_sum: f64 = (0..values.len()).map(|i| (i * i) as f64).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum);
        let intercept = (y_sum - slope * x_sum) / n;

        // Determine direction
        let direction = if slope > 0.001 {
            TrendDirection::Improving
        } else if slope < -0.001 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        // Calculate prediction for next hour (3600 seconds at 1Hz)
        let prediction_1h = intercept + slope * (values.len() + 3600) as f64;

        // Simple confidence based on variance
        let y_mean = y_sum / n;
        let variance = values.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n;
        let confidence = 1.0 / (1.0 + variance).sqrt();

        MetricTrend {
            direction,
            rate: slope * 60.0, // Convert to per-minute rate
            confidence: confidence as f32,
            prediction_1h,
        }
    }

    /// Check for alert conditions and trigger alerts
    pub async fn check_alerts(&mut self) -> Result<(), crate::BackendError> {
        let current_metrics = self.collect_metrics().await?;

        // Check each alert rule category
        self.check_utilization_alerts(&current_metrics).await?;
        self.check_pressure_alerts(&current_metrics).await?;
        self.check_fragmentation_alerts(&current_metrics).await?;
        self.check_transfer_alerts(&current_metrics).await?;
        self.check_learning_alerts(&current_metrics).await?;

        // Clean up expired alerts
        self.cleanup_expired_alerts();

        Ok(())
    }

    /// Check utilization-related alerts
    async fn check_utilization_alerts(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<(), crate::BackendError> {
        // Check if utilization is below target
        let utilization_distance =
            self.alert_rules.utilization_alerts.target_utilization - metrics.system_utilization;
        if utilization_distance > 0.1 {
            // More than 10% below target
            let alert = MemoryAlert {
                        id: format!("low_utilization_{}", metrics.timestamp),
                        level: AlertLevel::High,
                        alert_type: AlertType::Utilization,
                        message: format!("System utilization {:.1}% is below target of {:.1}%", metrics.system_utilization * 100.0, self.alert_rules.utilization_alerts.target_utilization * 100.0),
                        description: format!("Current utilization is {:.1}% below the >90% target. This may indicate inefficient memory usage or allocation policies.", utilization_distance * 100.0),
                        recommended_actions: vec![
                            "Review memory allocation policies".to_string(),
                            "Consider adjusting RL agent reward weights".to_string(),
                            "Monitor backend workload distribution".to_string(),
                        ],
                        timestamp: metrics.timestamp,
                        metrics_snapshot: metrics.clone(),
                        metadata: HashMap::from([
                            ("current_utilization".to_string(), format!("{:.3}", metrics.system_utilization)),
                            ("target_utilization".to_string(), format!("{:.3}", self.alert_rules.utilization_alerts.target_utilization)),
                            ("deficit".to_string(), format!("{:.3}", utilization_distance)),
                        ]),
                    };
            self.trigger_alert(alert).await?;
        }

        // Check if utilization exceeds critical threshold
        if metrics.system_utilization > self.alert_rules.utilization_alerts.critical_threshold {
            let alert = MemoryAlert {
                        id: format!("high_utilization_{}", metrics.timestamp),
                        level: AlertLevel::Critical,
                        alert_type: AlertType::Utilization,
                        message: format!("Critical utilization: {:.1}% exceeds {:.1}% threshold", metrics.system_utilization * 100.0, self.alert_rules.utilization_alerts.critical_threshold * 100.0),
                        description: "System memory utilization has reached critical levels. Immediate action required to prevent memory exhaustion.".to_string(),
                        recommended_actions: vec![
                            "Stop accepting new workloads".to_string(),
                            "Trigger memory defragmentation".to_string(),
                            "Scale up memory resources if available".to_string(),
                            "Alert system administrators".to_string(),
                        ],
                        timestamp: metrics.timestamp,
                        metrics_snapshot: metrics.clone(),
                        metadata: HashMap::from([
                            ("current_utilization".to_string(), format!("{:.3}", metrics.system_utilization)),
                            ("critical_threshold".to_string(), format!("{:.3}", self.alert_rules.utilization_alerts.critical_threshold)),
                            ("excess".to_string(), format!("{:.3}", metrics.system_utilization - self.alert_rules.utilization_alerts.critical_threshold)),
                        ]),
                    };
            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Check pressure-related alerts
    async fn check_pressure_alerts(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<(), crate::BackendError> {
        // Check NUMA violation rate
        if metrics.numa_violations
            > self
                .alert_rules
                .pressure_alerts
                .numa_violation_rate_threshold as u64
        {
            let alert = MemoryAlert {
                        id: format!("numa_violations_{}", metrics.timestamp),
                        level: AlertLevel::Warning,
                        alert_type: AlertType::Pressure,
                        message: format!("High NUMA violation rate: {} violations", metrics.numa_violations),
                        description: "Cross-NUMA memory access violations are occurring at high frequency, impacting performance.".to_string(),
                        recommended_actions: vec![
                            "Review NUMA affinity allocation policies".to_string(),
                            "Optimize workload placement".to_string(),
                            "Consider memory migration for better affinity".to_string(),
                        ],
                        timestamp: metrics.timestamp,
                        metrics_snapshot: metrics.clone(),
                        metadata: HashMap::from([
                            ("numa_violations".to_string(), format!("{}", metrics.numa_violations)),
                            ("threshold".to_string(), format!("{}", self.alert_rules.pressure_alerts.numa_violation_rate_threshold)),
                        ]),
                    };
            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Check fragmentation-related alerts
    async fn check_fragmentation_alerts(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<(), crate::BackendError> {
        if metrics.avg_fragmentation
            > self
                .alert_rules
                .fragmentation_alerts
                .critical_fragmentation_threshold
        {
            let alert = MemoryAlert {
                        id: format!("critical_fragmentation_{}", metrics.timestamp),
                        level: AlertLevel::Critical,
                        alert_type: AlertType::Fragmentation,
                        message: format!("Critical fragmentation: {:.1}% average fragmentation", metrics.avg_fragmentation * 100.0),
                        description: "Memory fragmentation has reached critical levels, severely impacting allocation efficiency.".to_string(),
                        recommended_actions: vec![
                            "Trigger immediate defragmentation operations".to_string(),
                            "Adjust allocation strategies to reduce fragmentation".to_string(),
                            "Consider memory pool restructuring".to_string(),
                        ],
                        timestamp: metrics.timestamp,
                        metrics_snapshot: metrics.clone(),
                        metadata: HashMap::from([
                            ("avg_fragmentation".to_string(), format!("{:.3}", metrics.avg_fragmentation)),
                            ("critical_threshold".to_string(), format!("{:.3}", self.alert_rules.fragmentation_alerts.critical_fragmentation_threshold)),
                        ]),
                    };
            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Check transfer-related alerts
    async fn check_transfer_alerts(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<(), crate::BackendError> {
        let failure_rate = if metrics.active_transfers > 0 {
            metrics.transfer_failures as f32 / metrics.active_transfers as f32
        } else {
            0.0
        };

        if failure_rate
            > self
                .alert_rules
                .transfer_alerts
                .transfer_failure_rate_threshold
        {
            let alert = MemoryAlert {
                id: format!("transfer_failures_{}", metrics.timestamp),
                level: AlertLevel::High,
                alert_type: AlertType::Transfer,
                message: format!("High transfer failure rate: {:.1}%", failure_rate * 100.0),
                description: "Cross-hardware memory transfers are failing at unacceptable rates."
                    .to_string(),
                recommended_actions: vec![
                    "Investigate transfer protocol failures".to_string(),
                    "Check hardware interconnect status".to_string(),
                    "Adjust transfer optimization policies".to_string(),
                ],
                timestamp: metrics.timestamp,
                metrics_snapshot: metrics.clone(),
                metadata: HashMap::from([
                    ("failure_rate".to_string(), format!("{:.3}", failure_rate)),
                    (
                        "total_failures".to_string(),
                        format!("{}", metrics.transfer_failures),
                    ),
                    (
                        "threshold".to_string(),
                        format!(
                            "{:.3}",
                            self.alert_rules
                                .transfer_alerts
                                .transfer_failure_rate_threshold
                        ),
                    ),
                ]),
            };
            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Check learning system alerts
    async fn check_learning_alerts(
        &mut self,
        metrics: &MetricsSnapshot,
    ) -> Result<(), crate::BackendError> {
        // Check for learning system anomalies using available metrics
        // For now, use transfer failures as a proxy for learning system issues
        let failure_rate = if metrics.active_transfers > 0 {
            metrics.transfer_failures as f32
                / (metrics.active_transfers as f32 + metrics.transfer_failures as f32)
        } else {
            0.0
        };

        if failure_rate > self.alert_rules.learning_alerts.decision_quality_threshold {
            let alert = MemoryAlert {
                id: format!("learning_transfer_failures_{}", metrics.timestamp),
                level: AlertLevel::Warning,
                alert_type: AlertType::Learning,
                message: format!(
                    "High transfer failure rate {:.3} indicates potential learning system issues",
                    failure_rate
                ),
                description: "The learning system may be making poor allocation decisions leading to transfer failures."
                    .to_string(),
                recommended_actions: vec![
                    "Review recent allocation decisions".to_string(),
                    "Check learning algorithm convergence".to_string(),
                    "Consider resetting exploration rate".to_string(),
                ],
                timestamp: metrics.timestamp,
                metrics_snapshot: metrics.clone(),
                metadata: HashMap::from([
                    ("failure_rate".to_string(), format!("{:.3}", failure_rate)),
                    ("transfer_failures".to_string(), metrics.transfer_failures.to_string()),
                    ("active_transfers".to_string(), metrics.active_transfers.to_string()),
                ]),
            };
            self.trigger_alert(alert).await?;
        }

        // Check for excessive NUMA violations as learning system indicator
        if metrics.numa_violations
            > self
                .alert_rules
                .pressure_alerts
                .numa_violation_rate_threshold as u64
        {
            let alert = MemoryAlert {
                id: format!("learning_numa_violations_{}", metrics.timestamp),
                level: AlertLevel::Warning,
                alert_type: AlertType::Learning,
                message: format!(
                    "High NUMA violation count {} indicates learning system inefficiency",
                    metrics.numa_violations
                ),
                description: "The learning system is not effectively optimizing memory placement across NUMA nodes."
                    .to_string(),
                recommended_actions: vec![
                    "Audit NUMA-aware allocation logic".to_string(),
                    "Increase weight on NUMA pressure in reward function".to_string(),
                    "Review learning exploration parameters".to_string(),
                ],
                timestamp: metrics.timestamp,
                metrics_snapshot: metrics.clone(),
                metadata: HashMap::from([
                    ("numa_violations".to_string(), metrics.numa_violations.to_string()),
                    ("threshold".to_string(), self.alert_rules.pressure_alerts.numa_violation_rate_threshold.to_string()),
                ]),
            };
            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Trigger an alert and notify all callbacks
    async fn trigger_alert(&mut self, alert: MemoryAlert) -> Result<(), crate::BackendError> {
        // Check if similar alert already exists
        let existing_alert = self
            .active_alerts
            .iter()
            .find(|a| a.alert_type == alert.alert_type && a.level == alert.level);

        if let Some(_existing) = existing_alert {
            // Update existing alert timestamp
            // In real implementation, would update the alert in place
            return Ok(());
        }

        // Add to active alerts
        self.active_alerts.push(alert.clone());

        // Notify all callbacks
        for callback in &self.alert_callbacks {
            callback.on_alert(&alert);
        }

        Ok(())
    }

    /// Clean up expired alerts
    fn cleanup_expired_alerts(&mut self) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut resolved_alerts = Vec::new();

        self.active_alerts.retain(|alert| {
            let age = current_time - alert.timestamp;
            if age > self.config.alert_persistence_secs {
                resolved_alerts.push(alert.id.clone());
                false
            } else {
                true
            }
        });

        // Notify callbacks about resolved alerts
        for alert_id in resolved_alerts {
            for callback in &self.alert_callbacks {
                callback.on_alert_resolved(&alert_id);
            }
        }
    }

    /// Get current utilization dashboard
    pub fn get_dashboard(&self) -> &UtilizationDashboard {
        &self.dashboard
    }

    /// Generate health report for external monitoring
    pub async fn get_health_status(&self) -> HealthStatus {
        let active_alerts_count: HashMap<AlertLevel, usize> =
            self.active_alerts
                .iter()
                .fold(HashMap::new(), |mut acc, alert| {
                    *acc.entry(alert.level).or_insert(0) += 1;
                    acc
                });

        let overall_score = self.calculate_overall_health_score();

        HealthStatus {
            overall_score,
            active_alerts_by_level: active_alerts_count,
            component_health: self.get_component_health(),
            last_check_timestamp: self.dashboard.current_metrics.timestamp,
        }
    }

    /// Calculate overall health score
    fn calculate_overall_health_score(&self) -> f32 {
        // Start with perfect score
        let mut score = 1.0;

        // Penalize for active alerts
        let alert_penalty: f32 = self
            .active_alerts
            .iter()
            .map(|alert| match alert.level {
                AlertLevel::Info => 0.01,
                AlertLevel::Warning => 0.05,
                AlertLevel::High => 0.15,
                AlertLevel::Critical => 0.30,
                AlertLevel::Emergency => 0.50,
            })
            .sum();

        score -= alert_penalty.min(0.8);

        // Penalize for low utilization
        if self.dashboard.current_metrics.system_utilization
            < self.alert_rules.utilization_alerts.target_utilization
        {
            let deficit = self.alert_rules.utilization_alerts.target_utilization
                - self.dashboard.current_metrics.system_utilization;
            score -= (deficit * 0.5).min(0.3);
        }

        // Bonus for high heterogeneity score (good balanced usage)
        if self.dashboard.current_metrics.heterogeneity_score > 0.8 {
            score += 0.1;
        }

        score.max(0.0)
    }

    /// Get component-level health status
    fn get_component_health(&self) -> HashMap<String, ComponentHealth> {
        let mut component_health = HashMap::new();

        // RL Agent health
        let rl_health = ComponentHealth {
            name: "RL Memory Agent".to_string(),
            health_score: if self.memory_manager.rl_agent.is_some() {
                0.95
            } else {
                0.0
            },
            status: HealthStatusIndicator::Healthy,
            last_update: self.dashboard.current_metrics.timestamp,
            issues: Vec::new(),
        };
        component_health.insert("rl_agent".to_string(), rl_health);

        // Memory pools health
        for (backend, utilization) in &self.dashboard.current_metrics.backend_utilization {
            let health_score = if *utilization > 0.9 {
                0.9 // Over-utilized but still functional
            } else if *utilization > 0.7 {
                1.0 // Good utilization
            } else {
                0.7 // Under-utilized
            };

            let status = if health_score > 0.9 {
                HealthStatusIndicator::Warning
            } else if health_score > 0.7 {
                HealthStatusIndicator::Healthy
            } else {
                HealthStatusIndicator::Warning
            };

            let issues = if *utilization < 0.5 {
                vec![format!("Low utilization: {:.1}%", utilization * 100.0)]
            } else {
                Vec::new()
            };

            let component = ComponentHealth {
                name: format!("{:?} Memory Pool", backend),
                health_score,
                status,
                last_update: self.dashboard.current_metrics.timestamp,
                issues,
            };
            component_health.insert(format!("pool_{:?}", backend), component);
        }

        component_health
    }
}

impl Default for AlertRules {
    fn default() -> Self {
        Self {
            utilization_alerts: UtilizationAlerts {
                target_utilization: 0.90,
                warning_threshold: 0.85,
                critical_threshold: 0.95,
                under_utilization_threshold: 0.50,
                heterogeneity_imbalance_threshold: 0.2,
            },
            pressure_alerts: PressureAlerts {
                high_pressure_threshold: 0.80,
                critical_pressure_threshold: 0.95,
                numa_violation_rate_threshold: 100,
                sustained_pressure_duration_secs: 300,
            },
            fragmentation_alerts: FragmentationAlerts {
                high_fragmentation_threshold: 0.5,
                critical_fragmentation_threshold: 0.7,
                defragmentation_failure_threshold: 5,
            },
            transfer_alerts: TransferAlerts {
                bandwidth_degradation_threshold: 0.2,
                transfer_failure_rate_threshold: 0.05,
                optimization_failure_threshold: 10,
            },
            learning_alerts: LearningAlerts {
                convergence_failure_threshold: 0.1,
                reward_degradation_threshold: 0.2,
                decision_quality_threshold: 0.7,
            },
        }
    }
}

impl UtilizationDashboard {
    /// Create new empty dashboard
    fn new() -> Self {
        Self {
            current_metrics: MetricsSnapshot {
                timestamp: 0,
                system_utilization: 0.0,
                backend_utilization: HashMap::new(),
                numa_pressures: Vec::new(),
                avg_fragmentation: 0.0,
                heterogeneity_score: 0.0,
                active_transfers: 0,
                numa_violations: 0,
                transfer_failures: 0,
            },
            trends: TrendAnalysis {
                utilization_trend: MetricTrend {
                    direction: TrendDirection::Stable,
                    rate: 0.0,
                    confidence: 0.0,
                    prediction_1h: 0.0,
                },
                pressure_trend: MetricTrend {
                    direction: TrendDirection::Stable,
                    rate: 0.0,
                    confidence: 0.0,
                    prediction_1h: 0.0,
                },
                heterogeneity_trend: MetricTrend {
                    direction: TrendDirection::Stable,
                    rate: 0.0,
                    confidence: 0.0,
                    prediction_1h: 0.0,
                },
                transfer_trend: MetricTrend {
                    direction: TrendDirection::Stable,
                    rate: 0.0,
                    confidence: 0.0,
                    prediction_1h: 0.0,
                },
            },
            backend_efficiency: HashMap::new(),
            threshold_breaches: Vec::new(),
            predictions: PerformancePredictions {
                utilization_1h: 0.0,
                critical_risk_1h: 0.0,
                heterogeneity_degradation: 0.0,
                exhaustion_risk: 0.0,
            },
        }
    }

    /// Get backend efficiency ratings
    pub fn backend_efficiency(&self) -> &HashMap<BackendType, EfficiencyRating> {
        &self.backend_efficiency
    }

    /// Get threshold breaches
    pub fn threshold_breaches(&self) -> &[ThresholdBreach] {
        &self.threshold_breaches
    }

    /// Get performance predictions
    pub fn predictions(&self) -> &PerformancePredictions {
        &self.predictions
    }

    /// Update backend efficiency rating
    pub fn update_backend_efficiency(
        &mut self,
        backend: BackendType,
        efficiency: EfficiencyRating,
    ) {
        self.backend_efficiency.insert(backend, efficiency);
    }

    /// Add threshold breach
    pub fn add_threshold_breach(&mut self, breach: ThresholdBreach) {
        self.threshold_breaches.push(breach);
    }

    /// Update predictions
    pub fn update_predictions(&mut self, predictions: PerformancePredictions) {
        self.predictions = predictions;
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_secs: 60,
            metrics_retention_hours: 24,
            alert_persistence_secs: 3600,
            dashboard_update_freq_secs: 30,
            prediction_lookahead_hours: 1,
        }
    }
}

// ============================================================================
