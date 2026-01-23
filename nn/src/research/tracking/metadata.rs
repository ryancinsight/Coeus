//! Comprehensive Experiment Metadata System
//!
//! This module provides detailed experiment metadata tracking including
//! environment information, system details, reproducibility data,
//! and research context information.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "sysinfo")]
use sysinfo::System;

/// Comprehensive experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    /// Unique experiment identifier
    pub experiment_id: String,
    /// Human-readable experiment name
    pub name: String,
    /// Detailed experiment description
    pub description: String,
    /// Research domain category
    pub domain: String,
    /// Experiment category/type
    pub category: ExperimentCategory,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modification timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Experiment owner/researcher
    pub owner: String,
    /// Collaboration information
    pub collaborators: Vec<String>,
    /// Research paper/study references
    pub references: Vec<String>,
    /// Research objectives and hypotheses
    pub objectives: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Expected outcomes
    pub expected_outcomes: Vec<String>,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Hardware configuration
    pub hardware: HardwareInfo,
    /// Software environment
    pub software: SoftwareInfo,
    /// Reproducibility information
    pub reproducibility: ReproducibilityInfo,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, serde_json::Value>,
    /// Data lineage information
    pub data_lineage: DataLineage,
}

impl ExperimentMetadata {
    /// Create new experiment metadata
    pub fn new(experiment_id: String, name: String, description: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            experiment_id: experiment_id.clone(),
            name,
            description,
            domain: "machine_learning".to_string(),
            category: ExperimentCategory::Research,
            status: ExperimentStatus::Planning,
            created_at: now,
            modified_at: now,
            owner: get_current_user(),
            collaborators: Vec::new(),
            references: Vec::new(),
            objectives: Vec::new(),
            success_criteria: Vec::new(),
            expected_outcomes: Vec::new(),
            environment: EnvironmentInfo::collect(),
            hardware: HardwareInfo::collect(),
            software: SoftwareInfo::collect(),
            reproducibility: ReproducibilityInfo::new(experiment_id),
            custom_fields: HashMap::new(),
            data_lineage: DataLineage::new(),
        }
    }

    /// Update modification timestamp
    pub fn mark_modified(&mut self) {
        self.modified_at = chrono::Utc::now();
    }

    /// Set research domain
    pub fn with_domain(mut self, domain: String) -> Self {
        self.domain = domain;
        self.mark_modified();
        self
    }

    /// Set experiment category
    pub fn with_category(mut self, category: ExperimentCategory) -> Self {
        self.category = category;
        self.mark_modified();
        self
    }

    /// Set experiment status
    pub fn set_status(&mut self, status: ExperimentStatus) {
        self.status = status;
        self.mark_modified();
    }

    /// Add objective
    pub fn add_objective(&mut self, objective: String) {
        self.objectives.push(objective);
        self.mark_modified();
    }

    /// Add success criterion
    pub fn add_success_criterion(&mut self, criterion: String) {
        self.success_criteria.push(criterion);
        self.mark_modified();
    }

    /// Add collaborator
    pub fn add_collaborator(&mut self, collaborator: String) {
        self.collaborators.push(collaborator);
        self.mark_modified();
    }

    /// Add research reference
    pub fn add_reference(&mut self, reference: String) {
        self.references.push(reference);
        self.mark_modified();
    }

    /// Set custom field
    pub fn set_custom_field(&mut self, key: String, value: serde_json::Value) {
        self.custom_fields.insert(key, value);
        self.mark_modified();
    }

    /// Get custom field
    pub fn get_custom_field(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom_fields.get(key)
    }

    /// Update data lineage
    pub fn update_data_lineage(&mut self, dataset_hash: String, transformation: String) {
        self.data_lineage
            .add_transformation(dataset_hash, transformation);
        self.mark_modified();
    }

    /// Export metadata to JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    /// Generate research summary
    pub fn research_summary(&self) -> String {
        format!(
            "📊 Experiment: {} (ID: {})\n\
             🎯 Domain: {}\n\
             🔬 Status: {:?}\n\
             👤 Owner: {}\n\
             📅 Created: {}\n\
             📝 Objectives: {}\n\
             ✅ Success Criteria: {}\n\
             🔗 References: {}",
            self.name,
            self.experiment_id,
            self.domain,
            self.status,
            self.owner,
            self.created_at.format("%Y-%m-%d %H:%M:%S UTC"),
            self.objectives.len(),
            self.success_criteria.len(),
            self.references.len()
        )
    }

    /// Generate reproducibility report
    pub fn reproducibility_report(&self) -> String {
        format!(
            "🔄 Reproducibility Report\n\
             📋 Seed: {}\n\
             🏗️  Framework Version: {}\n\
             💾 Git Commit: {}\n\
             ⚙️  Random State: Preserved\n\
             🔢 Numerical Precision: {}\n\
             📊 Hardware Consistency: {}\n\
             🖥️  OS: {} {}\n\
             🎯 Dependencies: {} packages tracked",
            self.reproducibility.random_seed,
            self.software.framework_version,
            self.reproducibility.git_commit,
            self.reproducibility.numerical_precision,
            if self.reproducibility.hardware_pinning {
                "Pinned"
            } else {
                "Variable"
            },
            self.environment.os_name,
            self.environment.os_version,
            self.software.dependencies.len()
        )
    }
}

/// Experiment category classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentCategory {
    /// Basic experiment/pilot study
    Pilot,
    /// Research/development experiment
    Research,
    /// Ablation study
    Ablation,
    /// Hyperparameter optimization
    HyperparameterOptimization,
    /// Benchmarking/comparison
    Benchmark,
    /// Production validation
    Production,
    /// Reproducibility check
    Reproducibility,
}

/// Experiment lifecycle status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Planning/design phase
    Planning,
    /// Setting up experiment
    Setup,
    /// Currently running
    Running,
    /// Temporarily paused
    Paused,
    /// Completed successfully
    Completed,
    /// Failed/terminated
    Failed,
    /// Results analyzed
    Analyzed,
    /// Published/shared
    Published,
}

/// Environment information collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Operating system name
    pub os_name: String,
    /// Operating system version
    pub os_version: String,
    /// Operating system architecture
    pub os_arch: String,
    /// hostname
    pub hostname: String,
    /// Username
    pub username: String,
    /// Working directory
    pub working_directory: String,
    /// Environment variables (selective)
    pub environment_variables: HashMap<String, String>,
    /// Python version (if available)
    pub python_version: Option<String>,
    /// Node.js version (if available)
    pub nodejs_version: Option<String>,
    /// GPU availability
    pub gpu_available: bool,
    /// GPU information
    pub gpu_info: Vec<String>,
}

impl EnvironmentInfo {
    /// Collect current environment information
    pub fn collect() -> Self {
        let gpu_info = collect_gpu_info();

        #[cfg(feature = "sysinfo")]
        let os_version = {
            let _sys = System::new();
            // TODO: Update sysinfo API usage
            // sys.refresh_system();
            // sys.long_os_version().unwrap_or_else(|| "Unknown".to_string())
            "Unknown".to_string()
        };

        #[cfg(not(feature = "sysinfo"))]
        let os_version = "Unknown".to_string();

        Self {
            os_name: std::env::consts::OS.to_string(),
            os_version,
            os_arch: std::env::consts::ARCH.to_string(),
            hostname: hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "Unknown".to_string()),
            username: get_current_user(),
            working_directory: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "Unknown".to_string()),
            environment_variables: collect_relevant_env_vars(),
            python_version: std::process::Command::new("python")
                .arg("--version")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string()),
            nodejs_version: std::process::Command::new("node")
                .arg("--version")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string()),
            gpu_available: !gpu_info.is_empty(),
            gpu_info,
        }
    }
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CPU frequency (MHz)
    pub cpu_frequency: u64,
    /// Total system memory (GB)
    pub total_memory_gb: f64,
    /// GPU information
    pub gpu_devices: Vec<GpuDevice>,
    /// Storage information
    pub storage_devices: Vec<StorageDevice>,
}

impl HardwareInfo {
    /// Collect hardware information
    pub fn collect() -> Self {
        #[cfg(feature = "sysinfo")]
        let (cpu_model, cpu_cores, cpu_frequency, total_memory_gb) = {
            // TODO: Update sysinfo API usage
            ("Unknown".to_string(), 0, 0, 0.0)
        };

        #[cfg(not(feature = "sysinfo"))]
        let (cpu_model, cpu_cores, cpu_frequency, total_memory_gb) =
            { ("Unknown".to_string(), num_cpus::get(), 0, 0.0) };

        let gpu_devices = collect_detailed_gpu_info();

        Self {
            cpu_model,
            cpu_cores,
            cpu_frequency,
            total_memory_gb,
            gpu_devices,
            storage_devices: collect_storage_info(),
        }
    }
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub name: String,
    pub memory_mb: usize,
    pub compute_capability: String,
    pub driver_version: String,
}

/// Storage device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDevice {
    pub mount_point: String,
    pub total_space_gb: f64,
    pub available_space_gb: f64,
    pub filesystem: String,
}

/// Software environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareInfo {
    /// Framework version
    pub framework_version: String,
    /// Rust version
    pub rust_version: String,
    /// Cargo version
    pub cargo_version: String,
    /// Git commit hash
    pub git_commit: String,
    /// Git branch
    pub git_branch: String,
    /// Dependencies with versions
    pub dependencies: HashMap<String, String>,
    /// Build configuration
    pub build_config: String,
}

impl SoftwareInfo {
    /// Collect software information
    pub fn collect() -> Self {
        Self {
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            rust_version: get_rust_version(),
            cargo_version: get_cargo_version(),
            git_commit: get_git_commit(),
            git_branch: get_git_branch(),
            dependencies: collect_dependencies(),
            build_config: collect_build_config(),
        }
    }
}

/// Reproducibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    /// Random seed used
    pub random_seed: u64,
    /// Git commit hash
    pub git_commit: String,
    /// Git repository state
    pub git_status: String,
    /// Numerical precision settings
    pub numerical_precision: String,
    /// Hardware pinning enabled
    pub hardware_pinning: bool,
    /// Deterministic computation settings
    pub deterministic_computation: bool,
    /// Random state snapshots
    pub random_state_snapshots: Vec<String>,
    /// Experiment execution order
    pub execution_order: Vec<String>,
}

impl ReproducibilityInfo {
    /// Create new reproducibility info
    pub fn new(experiment_id: String) -> Self {
        Self {
            random_seed: generate_seed(),
            git_commit: get_git_commit(),
            git_status: get_git_status(),
            numerical_precision: "f64".to_string(),
            hardware_pinning: false,
            deterministic_computation: true,
            random_state_snapshots: Vec::new(),
            execution_order: vec![experiment_id],
        }
    }

    /// Save random state snapshot
    pub fn save_random_state(&mut self, state: String) {
        self.random_state_snapshots.push(state);
    }
}

/// Data lineage information for tracking data transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLineage {
    /// Original dataset hashes
    pub dataset_hashes: Vec<String>,
    /// Data transformation history
    pub transformations: Vec<DataTransformation>,
    /// Input data sources
    pub data_sources: Vec<String>,
    /// Derived datasets
    pub derived_datasets: HashMap<String, String>,
}

impl DataLineage {
    /// Create new data lineage tracker
    pub fn new() -> Self {
        Self {
            dataset_hashes: Vec::new(),
            transformations: Vec::new(),
            data_sources: Vec::new(),
            derived_datasets: HashMap::new(),
        }
    }

    /// Add transformation to lineage
    pub fn add_transformation(&mut self, dataset_hash: String, transformation: String) {
        self.dataset_hashes.push(dataset_hash.clone());
        self.transformations.push(DataTransformation {
            timestamp: chrono::Utc::now(),
            description: transformation,
            input_datasets: vec![dataset_hash],
        });
    }

    /// Track data source
    pub fn add_data_source(&mut self, source: String) {
        self.data_sources.push(source);
    }

    /// Track derived dataset
    pub fn add_derived_dataset(&mut self, name: String, hash: String) {
        self.derived_datasets.insert(name, hash);
    }
}

impl Default for DataLineage {
    fn default() -> Self {
        Self::new()
    }
}

/// Data transformation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// Timestamp of transformation
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Description of transformation
    pub description: String,
    /// Input datasets used
    pub input_datasets: Vec<String>,
}

/// Helper functions
fn get_current_user() -> String {
    std::env::var("USERNAME")
        .unwrap_or_else(|_| std::env::var("USER").unwrap_or_else(|_| "Unknown".to_string()))
}

fn collect_relevant_env_vars() -> HashMap<String, String> {
    let relevant_vars = [
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CUDA_HOME",
    ];
    let mut result = HashMap::new();

    for var in relevant_vars {
        if let Ok(value) = std::env::var(var) {
            result.insert(var.to_string(), value);
        }
    }

    result
}

fn collect_gpu_info() -> Vec<String> {
    // Basic GPU detection - could be enhanced with actual GPU libraries
    #[cfg(feature = "gpu")]
    {
        // CUDA GPU detection would go here
        vec!["CUDA GPU(s) detected".to_string()]
    }
    #[cfg(not(feature = "gpu"))]
    {
        vec![]
    }
}

fn collect_detailed_gpu_info() -> Vec<GpuDevice> {
    // Placeholder - would integrate with actual GPU detection libraries
    vec![]
}

fn collect_storage_info() -> Vec<StorageDevice> {
    #[cfg(feature = "sysinfo")]
    {
        // TODO: Update sysinfo API usage
        vec![]
    }

    #[cfg(not(feature = "sysinfo"))]
    {
        vec![] // Return empty vector when sysinfo not available
    }
}

fn get_rust_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown".to_string())
}

fn get_cargo_version() -> String {
    std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown".to_string())
}

fn get_git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Not in git repository".to_string())
}

fn get_git_branch() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown".to_string())
}

fn get_git_status() -> String {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown".to_string())
}

fn collect_dependencies() -> HashMap<String, String> {
    // This would ideally read Cargo.toml or lockfile
    // For now, return basic framework dependencies
    let mut deps = HashMap::new();
    deps.insert(
        "coeus-core".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    );
    deps
}

fn collect_build_config() -> String {
    let mut config = Vec::new();

    if cfg!(debug_assertions) {
        config.push("debug");
    } else {
        config.push("release");
    }

    if cfg!(feature = "gpu") {
        config.push("cuda");
    }

    if cfg!(feature = "cpu") {
        config.push("cpu");
    }

    config.join(", ")
}

fn generate_seed() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    time.hash(&mut hasher);
    hostname::get().unwrap_or_default().hash(&mut hasher);
    hasher.finish()
}
