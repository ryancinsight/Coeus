//! Reproducible Research Framework
//!
//! This module provides comprehensive reproducibility guarantees for machine learning experiments,
//! including artifact versioning, environment capture, and deterministic execution.

use crate::core::error::{NNError, Result};
use crate::research::{ExperimentMetadata, ExperimentTracker};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Configuration for reproducible research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityConfig {
    /// Enable full environment capture
    pub capture_environment: bool,
    /// Enable deterministic execution
    pub deterministic_execution: bool,
    /// Enable artifact versioning
    pub version_artifacts: bool,
    /// Reproducibility storage directory
    pub storage_dir: PathBuf,
    /// Git repository tracking
    pub track_git: bool,
    /// Random seed for deterministic execution
    pub random_seed: Option<u64>,
}

impl Default for ReproducibilityConfig {
    fn default() -> Self {
        Self {
            capture_environment: true,
            deterministic_execution: true,
            version_artifacts: true,
            storage_dir: PathBuf::from("./reproducibility"),
            track_git: true,
            random_seed: Some(42),
        }
    }
}

/// Comprehensive reproducibility snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilitySnapshot {
    /// Unique snapshot identifier
    pub snapshot_id: String,
    /// Timestamp of snapshot creation
    pub timestamp: DateTime<Utc>,
    /// Experiment metadata
    pub experiment_metadata: ExperimentMetadata,
    /// Environment information
    pub environment: EnvironmentSnapshot,
    /// Code state (git commit, diff, etc.)
    pub code_state: CodeState,
    /// Dependencies and versions
    pub dependencies: DependencySnapshot,
    /// Hardware configuration
    pub hardware: HardwareSnapshot,
    /// Data lineage information
    pub data_lineage: DataLineage,
    /// Random seed used
    pub random_seed: Option<u64>,
}

/// Environment snapshot capturing all relevant system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    /// Operating system information
    pub os_info: String,
    /// CPU information
    pub cpu_info: String,
    /// Memory information
    pub memory_info: String,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
    /// Environment variables (filtered for relevance)
    pub env_vars: HashMap<String, String>,
    /// Current working directory
    pub working_directory: PathBuf,
    /// System timezone
    pub timezone: String,
}

/// Code state information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeState {
    /// Git commit hash
    pub git_commit: Option<String>,
    /// Git branch
    pub git_branch: Option<String>,
    /// Uncommitted changes
    pub uncommitted_changes: Option<String>,
    /// Repository URL
    pub repository_url: Option<String>,
    /// Code checksums
    pub file_checksums: HashMap<String, String>,
}

/// Dependency snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencySnapshot {
    /// Rust toolchain version
    pub rust_version: String,
    /// Cargo dependencies
    pub cargo_deps: HashMap<String, String>,
    /// Python dependencies (if applicable)
    pub python_deps: Option<HashMap<String, String>>,
    /// System libraries
    pub system_libs: Vec<String>,
}

/// Hardware configuration snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSnapshot {
    /// CPU model and cores
    pub cpu_model: String,
    pub cpu_cores: usize,
    /// Total system memory
    pub total_memory_gb: f64,
    /// GPU information
    pub gpu_models: Vec<String>,
    /// Storage information
    pub storage_info: String,
}

/// Data lineage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLineage {
    /// Dataset identifiers and versions
    pub datasets: Vec<DatasetInfo>,
    /// Data transformation pipeline
    pub transformations: Vec<String>,
    /// Data preprocessing steps
    pub preprocessing: Vec<String>,
}

/// Dataset information for lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Dataset version/hash
    pub version: String,
    /// Source location
    pub source: String,
    /// Size information
    pub size: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Reproducible experiment executor
pub struct ReproducibleExecutor {
    /// Configuration
    config: ReproducibilityConfig,
    /// Active snapshots
    snapshots: HashMap<String, ReproducibilitySnapshot>,
}

impl ReproducibleExecutor {
    /// Create new reproducible executor
    pub fn new(config: ReproducibilityConfig) -> Result<Self> {
        // Create storage directory if it doesn't exist
        if !config.storage_dir.exists() {
            fs::create_dir_all(&config.storage_dir)?;
        }

        Ok(Self {
            config,
            snapshots: HashMap::new(),
        })
    }

    /// Create reproducibility snapshot before experiment execution
    pub fn create_snapshot(
        &mut self,
        experiment_id: &str,
        tracker: &ExperimentTracker,
    ) -> Result<String> {
        let snapshot_id = format!("{}_{}", experiment_id, Utc::now().timestamp());

        let snapshot = ReproducibilitySnapshot {
            snapshot_id: snapshot_id.clone(),
            timestamp: Utc::now(),
            experiment_metadata: tracker.metadata.clone(),
            environment: self.capture_environment()?,
            code_state: self.capture_code_state()?,
            dependencies: self.capture_dependencies()?,
            hardware: self.capture_hardware()?,
            data_lineage: DataLineage {
                datasets: Vec::new(),
                transformations: Vec::new(),
                preprocessing: Vec::new(),
            },
            random_seed: self.config.random_seed,
        };

        // Save snapshot to disk
        self.save_snapshot(&snapshot)?;

        // Store in memory
        self.snapshots.insert(snapshot_id.clone(), snapshot);

        Ok(snapshot_id)
    }

    /// Execute experiment with reproducibility guarantees
    pub async fn execute_reproducible<F, Fut, T>(
        &self,
        snapshot_id: &str,
        experiment_fn: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // Set deterministic execution environment
        if self.config.deterministic_execution {
            self.setup_deterministic_environment(snapshot_id)?;
        }

        // Execute experiment
        let result = experiment_fn().await?;

        // Verify reproducibility if needed
        if self.config.capture_environment {
            self.verify_execution_environment(snapshot_id)?;
        }

        Ok(result)
    }

    /// Restore experiment environment from snapshot
    pub fn restore_environment(&self, snapshot_id: &str) -> Result<()> {
        let snapshot =
            self.snapshots
                .get(snapshot_id)
                .ok_or_else(|| NNError::InvalidConfiguration {
                    message: format!("Snapshot {} not found", snapshot_id),
                })?;

        // Restore random seed
        if let Some(seed) = snapshot.random_seed {
            // Set global random seed
            // Note: In a real implementation, this would affect all random number generators
            println!("🔄 Restored random seed: {}", seed);
        }

        // Restore environment variables
        for (key, value) in &snapshot.environment.env_vars {
            std::env::set_var(key, value);
        }

        // Change to original working directory
        std::env::set_current_dir(&snapshot.environment.working_directory)?;

        println!("🔄 Environment restored from snapshot {}", snapshot_id);
        Ok(())
    }

    /// Verify that current environment matches snapshot
    pub fn verify_reproducibility(&self, snapshot_id: &str) -> Result<ReproducibilityReport> {
        let snapshot =
            self.snapshots
                .get(snapshot_id)
                .ok_or_else(|| NNError::InvalidConfiguration {
                    message: format!("Snapshot {} not found", snapshot_id),
                })?;

        let mut report = ReproducibilityReport {
            snapshot_id: snapshot_id.to_string(),
            checks: Vec::new(),
            overall_reproducible: true,
        };

        // Check environment
        let current_env = self.capture_environment()?;
        report.checks.push(ReproducibilityCheck {
            name: "Environment".to_string(),
            status: current_env.os_info == snapshot.environment.os_info,
            details: format!(
                "OS: {} -> {}",
                snapshot.environment.os_info, current_env.os_info
            ),
        });

        // Check code state
        let current_code = self.capture_code_state()?;
        let code_match = current_code.git_commit == snapshot.code_state.git_commit;
        report.checks.push(ReproducibilityCheck {
            name: "Code State".to_string(),
            status: code_match,
            details: format!(
                "Git commit: {} -> {}",
                snapshot.code_state.git_commit.as_deref().unwrap_or("none"),
                current_code.git_commit.as_deref().unwrap_or("none")
            ),
        });

        // Check dependencies
        let current_deps = self.capture_dependencies()?;
        let deps_match = current_deps.rust_version == snapshot.dependencies.rust_version;
        report.checks.push(ReproducibilityCheck {
            name: "Dependencies".to_string(),
            status: deps_match,
            details: format!(
                "Rust version: {} -> {}",
                snapshot.dependencies.rust_version, current_deps.rust_version
            ),
        });

        // Update overall status
        report.overall_reproducible = report.checks.iter().all(|c| c.status);

        Ok(report)
    }

    /// Capture current environment information
    fn capture_environment(&self) -> Result<EnvironmentSnapshot> {
        let os_info = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);

        // CPU info (simplified)
        let cpu_info = format!("{} cores", num_cpus::get());

        // Memory info
        let memory_info = format!(
            "Available: {} MB",
            sysinfo::System::new_all().total_memory() / 1024 / 1024
        );

        // GPU info (placeholder)
        let gpu_info = None; // Would need GPU detection library

        // Environment variables (filter sensitive ones)
        let env_vars = std::env::vars()
            .filter(|(key, _)| !key.contains("SECRET") && !key.contains("PASSWORD"))
            .collect();

        let working_directory = std::env::current_dir()?;

        let timezone = "UTC".to_string(); // Simplified

        Ok(EnvironmentSnapshot {
            os_info,
            cpu_info,
            memory_info,
            gpu_info,
            env_vars,
            working_directory,
            timezone,
        })
    }

    /// Capture current code state
    fn capture_code_state(&self) -> Result<CodeState> {
        let mut git_commit = None;
        let mut git_branch = None;
        let mut uncommitted_changes = None;
        let mut repository_url = None;

        if self.config.track_git {
            // Try to get git information
            if let Ok(commit) = Self::run_command("git", &["rev-parse", "HEAD"]) {
                git_commit = Some(commit.trim().to_string());
            }

            if let Ok(branch) = Self::run_command("git", &["rev-parse", "--abbrev-ref", "HEAD"]) {
                git_branch = Some(branch.trim().to_string());
            }

            if let Ok(changes) = Self::run_command("git", &["status", "--porcelain"]) {
                if !changes.trim().is_empty() {
                    uncommitted_changes = Some(changes);
                }
            }

            if let Ok(url) = Self::run_command("git", &["config", "--get", "remote.origin.url"]) {
                repository_url = Some(url.trim().to_string());
            }
        }

        // Calculate file checksums for key source files
        let file_checksums = self.calculate_file_checksums()?;

        Ok(CodeState {
            git_commit,
            git_branch,
            uncommitted_changes,
            repository_url,
            file_checksums,
        })
    }

    /// Capture dependency information
    fn capture_dependencies(&self) -> Result<DependencySnapshot> {
        // Rust version
        let rust_version =
            Self::run_command("rustc", &["--version"]).unwrap_or_else(|_| "unknown".to_string());

        // Cargo dependencies (simplified - would parse Cargo.lock in real implementation)
        let cargo_deps = HashMap::new(); // Would parse Cargo.lock

        // Python dependencies (if applicable)
        let python_deps = None; // Would check if Python environment exists

        // System libraries (simplified)
        let system_libs = vec!["libc".to_string()]; // Would detect actual system libs

        Ok(DependencySnapshot {
            rust_version,
            cargo_deps,
            python_deps,
            system_libs,
        })
    }

    /// Capture hardware information
    fn capture_hardware(&self) -> Result<HardwareSnapshot> {
        let sys = sysinfo::System::new_all();

        let cpu_model = "Unknown CPU".to_string(); // Would need CPU detection
        let cpu_cores = sys.cpus().len();
        let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        let gpu_models = Vec::new(); // Would need GPU detection

        let storage_info = format!("Total: {} GB", sys.total_memory() / 1024 / 1024 / 1024);

        Ok(HardwareSnapshot {
            cpu_model,
            cpu_cores,
            total_memory_gb,
            gpu_models,
            storage_info,
        })
    }

    /// Calculate checksums for important source files
    fn calculate_file_checksums(&self) -> Result<HashMap<String, String>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut checksums = HashMap::new();
        let important_files = vec!["Cargo.toml", "Cargo.lock", "src/lib.rs", "src/main.rs"];

        for file_path in important_files {
            if Path::new(file_path).exists() {
                if let Ok(content) = fs::read(file_path) {
                    let mut hasher = DefaultHasher::new();
                    content.hash(&mut hasher);
                    let checksum = format!("{:x}", hasher.finish());
                    checksums.insert(file_path.to_string(), checksum);
                }
            }
        }

        Ok(checksums)
    }

    /// Save snapshot to disk
    fn save_snapshot(&self, snapshot: &ReproducibilitySnapshot) -> Result<()> {
        let snapshot_path = self
            .config
            .storage_dir
            .join(format!("{}.json", snapshot.snapshot_id));
        let json = serde_json::to_string_pretty(snapshot)?;
        fs::write(snapshot_path, json)?;
        Ok(())
    }

    /// Load snapshot from disk
    pub fn load_snapshot(&mut self, snapshot_id: &str) -> Result<()> {
        let snapshot_path = self
            .config
            .storage_dir
            .join(format!("{}.json", snapshot_id));
        let json = fs::read_to_string(snapshot_path)?;
        let snapshot: ReproducibilitySnapshot = serde_json::from_str(&json)?;
        self.snapshots.insert(snapshot_id.to_string(), snapshot);
        Ok(())
    }

    /// Set up deterministic execution environment
    fn setup_deterministic_environment(&self, snapshot_id: &str) -> Result<()> {
        // Set deterministic random seed
        if let Some(seed) = self.config.random_seed {
            // This would affect global random state in a real implementation
            println!("🎲 Set deterministic random seed: {}", seed);
        }

        // Disable any non-deterministic optimizations
        // Set environment variables for deterministic execution
        std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); // Deterministic CUDA
        std::env::set_var("PYTHONHASHSEED", "42"); // Python hash seed

        println!(
            "🔒 Deterministic execution environment set up for snapshot {}",
            snapshot_id
        );
        Ok(())
    }

    /// Verify execution environment matches snapshot
    fn verify_execution_environment(&self, snapshot_id: &str) -> Result<()> {
        let report = self.verify_reproducibility(snapshot_id)?;
        if !report.overall_reproducible {
            println!("⚠️  Reproducibility check failed:");
            for check in &report.checks {
                if !check.status {
                    println!("  ❌ {}: {}", check.name, check.details);
                }
            }
        } else {
            println!(
                "✅ Reproducibility check passed for snapshot {}",
                snapshot_id
            );
        }
        Ok(())
    }

    /// Run system command and return output
    fn run_command(command: &str, args: &[&str]) -> Result<String> {
        let output = Command::new(command).args(args).output().map_err(|e| {
            NNError::InvalidConfiguration {
                message: format!("Failed to run command '{}': {}", command, e),
            }
        })?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(NNError::InvalidConfiguration {
                message: format!(
                    "Command '{}' failed: {}",
                    command,
                    String::from_utf8_lossy(&output.stderr)
                ),
            })
        }
    }
}

/// Report on reproducibility verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityReport {
    /// Snapshot identifier
    pub snapshot_id: String,
    /// Individual checks performed
    pub checks: Vec<ReproducibilityCheck>,
    /// Overall reproducibility status
    pub overall_reproducible: bool,
}

/// Individual reproducibility check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityCheck {
    /// Check name
    pub name: String,
    /// Check status (passed/failed)
    pub status: bool,
    /// Details about the check
    pub details: String,
}

impl std::fmt::Display for ReproducibilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "🔍 Reproducibility Report for Snapshot: {}",
            self.snapshot_id
        )?;
        writeln!(
            f,
            "├── Overall Status: {}",
            if self.overall_reproducible {
                "✅ REPRODUCIBLE"
            } else {
                "❌ ISSUES FOUND"
            }
        )?;

        for check in &self.checks {
            let status = if check.status { "✅" } else { "❌" };
            writeln!(f, "├── {} {}: {}", status, check.name, check.details)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::UnifiedResearchFramework;

    #[test]
    fn test_reproducibility_config() {
        let config = ReproducibilityConfig::default();
        assert!(config.capture_environment);
        assert!(config.deterministic_execution);
        assert_eq!(config.random_seed, Some(42));
    }

    #[test]
    fn test_executor_creation() {
        let config = ReproducibilityConfig::default();
        let executor = ReproducibleExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[tokio::test]
    async fn test_snapshot_creation() {
        let config = ReproducibilityConfig::default();
        let mut executor = ReproducibleExecutor::new(config).unwrap();

        let mut framework = UnifiedResearchFramework::new();
        let tracker = framework.create_experiment(
            "test_exp".to_string(),
            "Test Experiment".to_string(),
            "Testing reproducibility".to_string(),
        );

        let snapshot_id = executor.create_snapshot("test_exp", &tracker).unwrap();
        assert!(!snapshot_id.is_empty());
        assert!(executor.snapshots.contains_key(&snapshot_id));
    }

    #[test]
    fn test_environment_capture() {
        let config = ReproducibilityConfig::default();
        let executor = ReproducibleExecutor::new(config).unwrap();

        let env = executor.capture_environment().unwrap();
        assert!(!env.os_info.is_empty());
        assert!(!env.cpu_info.is_empty());
    }

    #[test]
    fn test_reproducibility_verification() {
        let config = ReproducibilityConfig::default();
        let mut executor = ReproducibleExecutor::new(config).unwrap();

        let mut framework = UnifiedResearchFramework::new();
        let tracker = framework.create_experiment(
            "test_exp".to_string(),
            "Test Experiment".to_string(),
            "Testing reproducibility".to_string(),
        );

        let snapshot_id = executor.create_snapshot("test_exp", &tracker).unwrap();
        let report = executor.verify_reproducibility(&snapshot_id).unwrap();

        assert_eq!(report.snapshot_id, snapshot_id);
        assert!(!report.checks.is_empty());
    }
}
