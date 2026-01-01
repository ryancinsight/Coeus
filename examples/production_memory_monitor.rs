//! # Production Memory Monitoring and Alerting System
//!
//! Real-time monitoring, alerting, and dashboard for >90% memory utilization targets
//! with comprehensive production-ready features including predictive analytics,
//! automated alerting, and health monitoring.

use backend::{
    AlertCallback, AlertLevel, BackendType, HealthStatus, MemoryAlert, MemoryManager,
    ProductionMemoryMonitor,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

/// Comprehensive production memory monitor with real-time dashboard
pub struct ProductionMemoryMonitorExample {
    /// The memory monitor instance
    monitor: Arc<RwLock<ProductionMemoryMonitor>>,
    /// Console alert callback
    console_callback: ConsoleAlertCallback,
    /// Monitoring active flag
    monitoring_active: Arc<RwLock<bool>>,
}

impl ProductionMemoryMonitorExample {
    /// Create new production monitor example
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Create memory manager with heterogeneous pooling
        let memory_manager = Arc::new(MemoryManager::with_heterogeneous_pooling(0.90));

        // Create production monitor
        let production_monitor = ProductionMemoryMonitor::new(memory_manager.clone());

        let monitor = Arc::new(RwLock::new(production_monitor));
        let console_callback = ConsoleAlertCallback::new();

        // Add console callback for alerts
        {
            let mut monitor_guard = monitor.write().await;
            monitor_guard.add_alert_callback(Box::new(console_callback.clone()));
        }

        Ok(Self {
            monitor,
            console_callback,
            monitoring_active: Arc::new(RwLock::new(false)),
        })
    }

    /// Run production monitoring with real-time dashboard
    pub async fn run_production_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting Production Memory Monitoring System");
        println!("===============================================");

        // Set monitoring as active
        {
            let mut active = self.monitoring_active.write().await;
            *active = true;
        }

        // Spawn monitoring task
        let monitor_clone = self.monitor.clone();
        let active_clone = self.monitoring_active.clone();

        let monitoring_handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Check every 30 seconds

            while *active_clone.read().await {
                interval.tick().await;

                let mut monitor = monitor_clone.write().await;

                // Collect metrics and check alerts
                if let Err(e) = monitor.collect_metrics().await {
                    eprintln!("Failed to collect metrics: {}", e);
                    continue;
                }

                if let Err(e) = monitor.check_alerts().await {
                    eprintln!("Failed to check alerts: {}", e);
                    continue;
                }

                // Get current dashboard
                let dashboard = monitor.get_dashboard();

                // Print dashboard update
                Self::print_dashboard_update(dashboard).await;

                // Get health status
                let health = monitor.get_health_status().await;

                // Print health summary
                Self::print_health_summary(&health).await;
            }
        });

        // Spawn dashboard display task
        let monitor_clone2 = self.monitor.clone();
        let active_clone2 = self.monitoring_active.clone();

        let dashboard_handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Update dashboard every minute

            while *active_clone2.read().await {
                interval.tick().await;

                Self::display_full_dashboard(&monitor_clone2).await;
            }
        });

        // Simulate memory workload for testing
        let workload_handle = self.simulate_memory_workload().await;

        // Wait for user input to stop
        println!("\n📊 Monitoring active. Press Enter to stop...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        // Stop monitoring
        {
            let mut active = self.monitoring_active.write().await;
            *active = false;
        }

        // Wait for tasks to complete
        let _ = tokio::try_join!(monitoring_handle, dashboard_handle, workload_handle);

        // Final comprehensive report
        self.print_final_report().await?;

        Ok(())
    }

    /// Simulate realistic memory workload patterns
    async fn simulate_memory_workload(&self) -> tokio::task::JoinHandle<()> {
        let monitor_clone = self.monitor.clone();

        tokio::spawn(async move {
            // Phase 1: Ramp up to high utilization (simulate training start)
            println!("\n🔧 Simulating memory workload patterns...");

            for i in 0..20 {
                {
                    let monitor = monitor_clone.read().await;
                    let memory_manager = &monitor.memory_manager;

                    // Allocate varying sized blocks to simulate different operations
                    let sizes = [
                        64 * 1_048_576,   // 64MB (small allocations)
                        256 * 1_048_576,  // 256MB (medium allocations)
                        1024 * 1_048_576, // 1GB (large allocations)
                        512 * 1_048_576,  // 512MB (mixed workload)
                    ];

                    let operations = [
                        backend::OperationType::ElementWise,
                        backend::OperationType::MatrixMultiplication,
                        backend::OperationType::Convolution,
                        backend::OperationType::Reduction,
                    ];

                    for (j, (&size, &operation)) in sizes.iter().zip(operations.iter()).enumerate()
                    {
                        if i + j < 25 {
                            // Vary allocation pattern
                            let _allocation = memory_manager
                                .allocate_heterogeneous_memory(
                                    size,
                                    backend::MemoryAccessPattern::Dense,
                                    backend::DataLocality::High,
                                    operation,
                                )
                                .await;

                            // Occasionally trigger transfers
                            if j % 3 == 0 && i > 10 {
                                let _transfer = memory_manager
                                    .transfer_memory_cross_hardware(
                                        backend::BackendType::Gpu,
                                        backend::BackendType::Cpu,
                                        size / 2,
                                        backend::MemoryAccessPattern::Dense,
                                    )
                                    .await;
                            }
                        }
                    }
                }

                tokio::time::sleep(Duration::from_secs(15)).await;
            }

            // Phase 2: Steady state with occasional spikes
            println!("📈 Transitioning to steady-state operation...");
            for _ in 0..10 {
                {
                    let monitor = monitor_clone.read().await;
                    let memory_manager = &monitor.memory_manager;

                    // Simulate steady workload with occasional large allocations
                    if rand::random::<f32>() < 0.3 {
                        // 30% chance
                        let _allocation = memory_manager
                            .allocate_heterogeneous_memory(
                                2048 * 1_048_576, // 2GB spike
                                backend::MemoryAccessPattern::Dense,
                                backend::DataLocality::High,
                                backend::OperationType::MatrixMultiplication,
                            )
                            .await;
                    }
                }

                tokio::time::sleep(Duration::from_secs(30)).await;
            }

            // Phase 3: Deallocation and cleanup (simulate training end)
            println!("🧹 Simulating cleanup and deallocation...");
            tokio::time::sleep(Duration::from_secs(10)).await;

            println!("✅ Memory workload simulation completed");
        })
    }

    /// Print real-time dashboard update
    async fn print_dashboard_update(dashboard: &backend::UtilizationDashboard) {
        let metrics = &dashboard.current_metrics;

        println!("\n📊 REAL-TIME DASHBOARD UPDATE");
        println!("============================");
        println!(
            "System Utilization: {:.1}%",
            metrics.system_utilization * 100.0
        );

        // Show backend breakdown
        println!("Backend Utilization:");
        for (backend, utilization) in &metrics.backend_utilization {
            println!("  {:?}: {:.1}%", backend, utilization * 100.0);
        }

        println!("Heterogeneity Score: {:.3}", metrics.heterogeneity_score);
        println!(
            "Average Fragmentation: {:.1}%",
            metrics.avg_fragmentation * 100.0
        );
        println!("Active Transfers: {}", metrics.active_transfers);

        // Show utilization trend
        println!("Trend: {:?}", dashboard.trends.utilization_trend.direction);
        println!(
            "1H Prediction: {:.1}%",
            dashboard.trends.utilization_trend.prediction_1h * 100.0
        );

        println!(
            "Last Update: {}",
            chrono::NaiveDateTime::from_timestamp_opt(metrics.timestamp as i64, 0)
                .map(|dt| dt.to_string())
                .unwrap_or_else(|| "Unknown".to_string())
        );
    }

    /// Display comprehensive full dashboard
    async fn display_full_dashboard(monitor: &Arc<RwLock<ProductionMemoryMonitor>>) {
        let monitor_guard = monitor.read().await;
        let dashboard = monitor_guard.get_dashboard();

        println!("\n📈 COMPREHENSIVE MEMORY DASHBOARD");
        println!("=================================");

        // Target achievement summary
        println!("\n🎯 TARGET ACHIEVEMENT (>90% utilization)");
        println!(
            "   Current Utilization: {:.1}%",
            dashboard.current_metrics.system_utilization * 100.0
        );
        println!("   Target: 90.0%");
        let diff = dashboard.current_metrics.system_utilization - 0.90;
        if diff >= 0.0 {
            println!("   Status: ✅ TARGET ACHIEVED (+{:.1}%)", diff * 100.0);
        } else {
            println!("   Status: ⚠️ BELOW TARGET ({:.1}%)", diff.abs() * 100.0);
        }

        // Trend analysis
        println!("\n📈 TREND ANALYSIS");
        println!(
            "   Utilization Trend: {:?} (Rate: {:.3}%/min)",
            dashboard.trends.utilization_trend.direction,
            dashboard.trends.utilization_trend.rate * 100.0
        );

        println!(
            "   Memory Pressure Trend: {:?}",
            dashboard.trends.pressure_trend.direction
        );
        println!(
            "   Heterogeneity Trend: {:?}",
            dashboard.trends.heterogeneity_trend.direction
        );

        // Backend efficiency ratings
        println!("\n🏆 BACKEND EFFICIENCY RATINGS");
        for (backend, efficiency) in &dashboard.backend_efficiency {
            println!("   {:?}:", backend);
            println!(
                "     Utilization Efficiency: {:.1}%",
                efficiency.utilization_efficiency * 100.0
            );
            println!("     Speed Rating: {:?}", efficiency.allocation_speed);
        }

        // Predictions
        println!("\n🔮 PERFORMANCE PREDICTIONS (1H)");
        println!(
            "   Utilization: {:.1}%",
            dashboard.predictions.utilization_1h * 100.0
        );
        println!(
            "   Critical Risk: {:.1}%",
            dashboard.predictions.critical_risk_1h * 100.0
        );
        println!(
            "   Heterogeneity Degradation: {:.3}",
            dashboard.predictions.heterogeneity_degradation
        );

        // NUMA node status
        println!("\n🎯 NUMA NODE STATUS");
        println!(
            "   Node Pressures: {:.1}%",
            dashboard.current_metrics.numa_pressures.iter().sum::<f32>()
                / dashboard.current_metrics.numa_pressures.len().max(1) as f32
                * 100.0
        );

        println!(
            "   NUMA Violations: {}",
            dashboard.current_metrics.numa_violations
        );

        // Threshold breaches
        if !dashboard.threshold_breaches.is_empty() {
            println!("\n⚠️ ACTIVE THRESHOLD BREACHES");
            for breach in &dashboard.threshold_breaches {
                println!(
                    "   {}: {} (Actual: {:.1}, Threshold: {:.1})",
                    breach.metric,
                    if breach.end_timestamp.is_some() {
                        "RESOLVED"
                    } else {
                        "ACTIVE"
                    },
                    breach.actual_value * 100.0,
                    breach.threshold_value * 100.0
                );
            }
        }
    }

    /// Print health summary
    async fn print_health_summary(health: &HealthStatus) {
        println!("\n💚 SYSTEM HEALTH SUMMARY");
        println!("===========================");
        println!("Overall Health Score: {:.1}%", health.overall_score * 100.0);

        let health_indicator = match health.overall_score {
            s if s >= 0.9 => "🟢 EXCELLENT",
            s if s >= 0.8 => "🟡 GOOD",
            s if s >= 0.7 => "🟠 FAIR",
            s if s >= 0.6 => "🔴 POOR",
            _ => "🚨 CRITICAL",
        };
        println!("Health Status: {}", health_indicator);

        println!("\nActive Alerts by Severity:");
        for (level, count) in &health.active_alerts_by_level {
            println!("  {:?}: {}", level, count);
        }

        println!("\nComponent Health:");
        for (component_name, component_health) in &health.component_health {
            let emoji = match component_health.status {
                backend::HealthStatusIndicator::Healthy => "🟢",
                backend::HealthStatusIndicator::Warning => "🟡",
                backend::HealthStatusIndicator::Degraded => "🟠",
                backend::HealthStatusIndicator::Critical => "🔴",
                backend::HealthStatusIndicator::Failed => "❌",
            };
            println!(
                "  {} {}: {:.1}% ({:?})",
                emoji,
                component_name,
                component_health.health_score * 100.0,
                component_health.status
            );

            if !component_health.issues.is_empty() {
                for issue in &component_health.issues {
                    println!("    ⚠️ {}", issue);
                }
            }
        }
    }

    /// Print comprehensive final monitoring report
    async fn print_final_report(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n📋 PRODUCTION MEMORY MONITORING FINAL REPORT");
        println!("=============================================");

        let monitor_guard = self.monitor.read().await;

        // Overall assessment
        let dashboard = monitor_guard.get_dashboard();
        let final_health = monitor_guard.get_health_status().await;

        println!("\n🏆 FINAL ASSESSMENT");
        println!("==================");

        let utilization_achievement = dashboard.current_metrics.system_utilization >= 0.90;
        let heterogeneity_good = dashboard.current_metrics.heterogeneity_score >= 0.85;
        let fragmentation_low = dashboard.current_metrics.avg_fragmentation < 0.2;
        let alerts_minimal = final_health.active_alerts_by_level.values().sum::<usize>() < 3;

        let overall_success =
            utilization_achievement && heterogeneity_good && fragmentation_low && alerts_minimal;

        println!(
            "Production Readiness Score: {:?}",
            if overall_success {
                "✅ PRODUCTION READY"
            } else {
                "⚠️ NEEDS OPTIMIZATION"
            }
        );

        println!("\n📊 Key Metrics Summary:");
        println!(
            "   - Utilization Achievement: {:.1}% {}",
            dashboard.current_metrics.system_utilization * 100.0,
            if utilization_achievement {
                "✅"
            } else {
                "❌"
            }
        );
        println!(
            "   - Heterogeneity Score: {:.3} {}",
            dashboard.current_metrics.heterogeneity_score,
            if heterogeneity_good { "✅" } else { "❌" }
        );
        println!(
            "   - Fragmentation Level: {:.1}% {}",
            dashboard.current_metrics.avg_fragmentation * 100.0,
            if fragmentation_low { "✅" } else { "❌" }
        );
        println!(
            "   - Active Alerts: {} {}",
            final_health.active_alerts_by_level.values().sum::<usize>(),
            if alerts_minimal { "✅" } else { "❌" }
        );
        println!(
            "   - Overall Health Score: {:.1}%",
            final_health.overall_score * 100.0
        );

        println!("\n🔍 Alert Summary:");
        if final_health.active_alerts_by_level.is_empty() {
            println!("   ✅ No active alerts - System stable");
        } else {
            for (level, count) in &final_health.active_alerts_by_level {
                println!("   {:?} Alerts: {} ⚠️", level, count);
            }
        }

        println!("\n🎯 Recommendations for Production:");
        if overall_success {
            println!("   ✅ System is ready for production deployment");
            println!("   📈 Continue monitoring and fine-tuning RL policies");
            println!("   🔄 Consider scaling to multiple nodes");
        } else {
            if !utilization_achievement {
                println!("   ⚠️ Improve utilization - Review RL agent reward functions");
            }
            if !heterogeneity_good {
                println!("   ⚠️ Balance backend utilization - Check workload distribution");
            }
            if !fragmentation_low {
                println!("   ⚠️ Reduce fragmentation - Implement better memory management");
            }
            if !alerts_minimal {
                println!("   ⚠️ Reduce active alerts - Address system health issues");
            }
        }

        println!("\n🔧 RL Agent Performance:");
        // Note: RL performance would be tracked in the full system

        println!("\n✅ Monitoring session completed successfully!");

        Ok(())
    }
}

/// Console alert callback for display purposes
#[derive(Clone)]
pub struct ConsoleAlertCallback;

impl ConsoleAlertCallback {
    /// Create new console callback
    pub fn new() -> Self {
        Self
    }
}

impl AlertCallback for ConsoleAlertCallback {
    /// Handle alert trigger
    fn on_alert(&self, alert: &MemoryAlert) {
        let emoji = match alert.level {
            AlertLevel::Info => "ℹ️",
            AlertLevel::Warning => "⚠️",
            AlertLevel::High => "🔶",
            AlertLevel::Critical => "🚨",
            AlertLevel::Emergency => "🆘",
        };

        println!("\n{} ALERT TRIGGERED: {:?}", emoji, alert.level);
        println!("ID: {}", alert.id);
        println!("Type: {:?}", alert.alert_type);
        println!("Message: {}", alert.message);
        println!("Description: {}", alert.description);

        if !alert.recommended_actions.is_empty() {
            println!("Recommended Actions:");
            for action in &alert.recommended_actions {
                println!("  • {}", action);
            }
        }
    }

    fn on_alert_resolved(&self, alert_id: &str) {
        println!("\n✅ ALERT RESOLVED: {}", alert_id);
    }

    fn on_health_check(&self, status: &HealthStatus) {
        println!(
            "\n🩺 Health Check: score={:.1}% status={:?}",
            status.overall_score * 100.0,
            status.status
        );
    }
}
