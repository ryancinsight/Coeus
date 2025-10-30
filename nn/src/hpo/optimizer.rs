//! Main HPO optimizer interface.
//!
//! This module provides the main interface for hyperparameter optimization,
//! supporting multiple algorithms and evaluation strategies.

use std::time::{Duration, Instant};

use super::space::{HyperparameterSpace, HyperparameterConfig};
use crate::error::{NNError, Result};

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best hyperparameter configuration found
    pub best_config: HyperparameterConfig,
    /// Best objective value achieved
    pub best_value: f64,
    /// Total evaluations performed
    pub evaluations: usize,
    /// Total optimization time
    pub total_time: Duration,
    /// Evaluation history
    pub history: Vec<(HyperparameterConfig, f64, Duration)>,
}

/// Enum-based optimizer to avoid dyn trait issues
#[derive(Debug)]
pub enum HPOptimizer {
    Bayesian(super::bayesian::BayesianOptimizer),
    Bandit(super::bandits::BanditOptimizer),
    Population(super::population::PopulationOptimizer),
    Hyperband(super::multifidelity::HyperbandOptimizer),
    SuccessiveHalving(super::multifidelity::SuccessiveHalving),
    Bohb(super::multifidelity::BohbOptimizer),
    // Production-ready population-based algorithms
    ParticleSwarm(super::population::PopulationOptimizer),
    CmaEs(super::population::PopulationOptimizer),
    DifferentialEvolution(super::population::PopulationOptimizer),
}

impl HPOptimizer {
    /// Run hyperparameter optimization
    pub fn optimize<F>(&mut self, objective: F, max_evaluations: usize) -> Result<OptimizationResult>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        match self {
            HPOptimizer::Bayesian(opt) => opt.optimize(objective, max_evaluations),
            HPOptimizer::Bandit(opt) => opt.optimize(objective, max_evaluations),
            HPOptimizer::Population(opt) => opt.optimize(objective, max_evaluations),
            // New production-ready population-based optimizers
            HPOptimizer::ParticleSwarm(opt) => opt.optimize(objective, max_evaluations),
            HPOptimizer::CmaEs(opt) => opt.optimize(objective, max_evaluations),
            HPOptimizer::DifferentialEvolution(opt) => opt.optimize(objective, max_evaluations),
            HPOptimizer::Hyperband(_) => Err(NNError::NotImplemented { operation: "Hyperband optimizer".to_string() }),
            HPOptimizer::SuccessiveHalving(_) => Err(NNError::NotImplemented { operation: "SuccessiveHalving optimizer".to_string() }),
            HPOptimizer::Bohb(_) => Err(NNError::NotImplemented { operation: "BOHB optimizer".to_string() }),
        }
    }

    /// Get optimizer name
    pub fn name(&self) -> &str {
        match self {
            HPOptimizer::Bayesian(_) => "BayesianOptimization",
            HPOptimizer::Bandit(_) => "BanditOptimization",
            HPOptimizer::Population(_) => "PopulationOptimization",
            // New production-ready population-based optimizers
            HPOptimizer::ParticleSwarm(_) => "ParticleSwarmOptimization",
            HPOptimizer::CmaEs(_) => "CMA-ES",
            HPOptimizer::DifferentialEvolution(_) => "DifferentialEvolution",
            HPOptimizer::Hyperband(_) => "Hyperband",
            HPOptimizer::SuccessiveHalving(_) => "SuccessiveHalving",
            HPOptimizer::Bohb(_) => "BOHB",
        }
    }

    /// Create production-ready PSO optimizer
    pub fn create_pso(space: HyperparameterSpace, population_size: Option<usize>) -> Result<Self> {
        let algorithm = super::population::PopulationAlgorithm::ParticleSwarm {
            inertia_start: 0.9,
            inertia_end: 0.4,
            cognitive: 2.0,
            social: 2.0,
        };

        let mut optimizer = super::population::PopulationOptimizer::new(space);
        optimizer = optimizer.with_algorithm(algorithm)?;
        if let Some(size) = population_size {
            optimizer = optimizer.with_population_size(size)?;
        }

        Ok(HPOptimizer::ParticleSwarm(optimizer))
    }

    /// Create production-ready CMA-ES optimizer
    pub fn create_cmaes(space: HyperparameterSpace, population_size: Option<usize>) -> Result<Self> {
        let algorithm = super::population::PopulationAlgorithm::CmaEs {
            initial_sigma: 0.3,
            tolerance: 1e-6,
        };

        let mut optimizer = super::population::PopulationOptimizer::new(space);
        optimizer = optimizer.with_algorithm(algorithm)?;
        if let Some(size) = population_size {
            optimizer = optimizer.with_population_size(size)?;
        }

        Ok(HPOptimizer::CmaEs(optimizer))
    }

    /// Create production-ready Differential Evolution optimizer
    pub fn create_de(space: HyperparameterSpace, population_size: Option<usize>) -> Result<Self> {
        let algorithm = super::population::PopulationAlgorithm::DifferentialEvolution {
            f: 0.7,
            cr: 0.9,
        };

        let mut optimizer = super::population::PopulationOptimizer::new(space);
        optimizer = optimizer.with_algorithm(algorithm)?;
        if let Some(size) = population_size {
            optimizer = optimizer.with_population_size(size)?;
        }

        Ok(HPOptimizer::DifferentialEvolution(optimizer))
    }
}

/// Comprehensive hyperparameter optimization framework
#[derive(Debug)]
pub struct HyperparameterOptimizer {
    /// Available optimization algorithms
    algorithms: Vec<HPOptimizer>,
    /// Current algorithm index
    current_algorithm: usize,
    /// Hyperparameter space
    space: HyperparameterSpace,
    /// Evaluation budget per run
    budget: usize,
    /// Random seed for reproducibility
    seed: Option<u64>,
}

impl HyperparameterOptimizer {
    /// Create a new hyperparameter optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            algorithms: Vec::new(),
            current_algorithm: 0,
            space,
            budget: 50,
            seed: None,
        }
    }

    /// Add an optimization algorithm
    pub fn add_algorithm(&mut self, algorithm: HPOptimizer) {
        self.algorithms.push(algorithm);
    }

    /// Set evaluation budget
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run optimization with all algorithms
    pub fn optimize_all<F>(&mut self, objective: F) -> Result<Vec<OptimizationResult>>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Clone + Send + Sync,
    {
        let mut results = Vec::new();

        for algorithm in &mut self.algorithms {
            let result = algorithm.optimize(objective.clone(), self.budget)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Run optimization with current algorithm
    pub fn optimize<F>(&mut self, objective: F) -> Result<OptimizationResult>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        if self.algorithms.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "No optimization algorithms configured".to_string(),
            });
        }

        if self.current_algorithm >= self.algorithms.len() {
            self.current_algorithm = 0;
        }

        self.algorithms[self.current_algorithm].optimize(objective, self.budget)
    }

    /// Switch to next algorithm
    pub fn next_algorithm(&mut self) {
        self.current_algorithm = (self.current_algorithm + 1) % self.algorithms.len();
    }

    /// Get current algorithm name
    pub fn current_algorithm_name(&self) -> &str {
        if self.algorithms.is_empty() {
            "None"
        } else {
            self.algorithms[self.current_algorithm].name()
        }
    }

    /// Create a comprehensive optimizer with all algorithms
    pub fn comprehensive(space: HyperparameterSpace) -> Self {
        let mut optimizer = Self::new(space);

        // Add Bayesian optimization
        let bo = super::bayesian::BayesianOptimizer::new(optimizer.space.clone());
        optimizer.add_algorithm(HPOptimizer::Bayesian(bo));

        // Add bandit optimization
        let bandit = super::bandits::BanditOptimizer::new(optimizer.space.clone());
        optimizer.add_algorithm(HPOptimizer::Bandit(bandit));

        // Add population-based optimization
        let population = super::population::PopulationOptimizer::new(optimizer.space.clone());
        optimizer.add_algorithm(HPOptimizer::Population(population));

        // Add Hyperband
        let hyperband = super::multifidelity::HyperbandOptimizer::new(optimizer.space.clone());
        optimizer.add_algorithm(HPOptimizer::Hyperband(hyperband));

        optimizer
    }
}

/// Optimization benchmark utilities
#[derive(Default)]
pub struct BenchmarkRunner {
    /// Available optimizers
    optimizers: Vec<HyperparameterOptimizer>,
    /// Benchmark functions
    functions: Vec<Box<dyn BenchmarkFunction>>,
}


impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an optimizer to benchmark
    pub fn add_optimizer(&mut self, optimizer: HyperparameterOptimizer) {
        self.optimizers.push(optimizer);
    }

    /// Add a benchmark function
    pub fn add_function(&mut self, function: Box<dyn BenchmarkFunction>) {
        self.functions.push(function);
    }

    /// Run comprehensive benchmark
    pub fn run_benchmark(&mut self, _evaluations_per_run: usize) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults {
            results: Vec::new(),
            times: Vec::new(),
            optimizer_names: Vec::new(),
            function_names: Vec::new(),
        };

        for (opt_idx, optimizer) in self.optimizers.iter_mut().enumerate() {
            for (func_idx, function) in self.functions.iter().enumerate() {
                let start_time = Instant::now();

                let opt_result = optimizer.optimize(|config| function.evaluate(config))?;

                let elapsed = start_time.elapsed();

                results.add_result(
                    opt_idx,
                    func_idx,
                    opt_result,
                    elapsed,
                );
            }
        }

        Ok(results)
    }
}

/// Benchmark function trait
pub trait BenchmarkFunction: Send + Sync {
    /// Evaluate the function at a configuration
    fn evaluate(&self, config: &HyperparameterConfig) -> Result<f64>;

    /// Get function name
    fn name(&self) -> &str;

    /// Get optimal value
    fn optimal_value(&self) -> f64;

    /// Get search space for this function
    fn space(&self) -> &HyperparameterSpace;
}

/// Benchmark results
#[derive(Default)]
pub struct BenchmarkResults {
    /// Results for each optimizer-function pair
    pub results: Vec<Vec<Option<OptimizationResult>>>,
    /// Total times for each optimizer-function pair
    pub times: Vec<Vec<Option<Duration>>>,
    /// Optimizer names
    pub optimizer_names: Vec<String>,
    /// Function names
    pub function_names: Vec<String>,
}

impl BenchmarkResults {
    /// Create new benchmark results
    pub fn new() -> Self {
        Self::default()
    }
}


impl BenchmarkResults {
    /// Add a result
    pub fn add_result(
        &mut self,
        optimizer_idx: usize,
        function_idx: usize,
        result: OptimizationResult,
        time: Duration,
    ) {
        // Ensure we have enough space
        while self.results.len() <= optimizer_idx {
            self.results.push(Vec::new());
            self.times.push(Vec::new());
        }

        while self.results[optimizer_idx].len() <= function_idx {
            self.results[optimizer_idx].push(None);
            self.times[optimizer_idx].push(None);
        }

        self.results[optimizer_idx][function_idx] = Some(result);
        self.times[optimizer_idx][function_idx] = Some(time);
    }

    /// Set optimizer names
    pub fn set_optimizer_names(&mut self, names: Vec<String>) {
        self.optimizer_names = names;
    }

    /// Set function names
    pub fn set_function_names(&mut self, names: Vec<String>) {
        self.function_names = names;
    }

    /// Generate performance summary
    pub fn performance_summary(&self) -> Vec<Vec<f64>> {
        let mut summary = Vec::new();

        for opt_results in &self.results {
            let mut opt_summary = Vec::new();

            for func_result in opt_results {
                if let Some(result) = func_result {
                    opt_summary.push(result.best_value);
                } else {
                    opt_summary.push(f64::NAN);
                }
            }

            summary.push(opt_summary);
        }

        summary
    }

    /// Generate ranking table (lower values are better)
    pub fn ranking_table(&self) -> Vec<Vec<usize>> {
        let summary = self.performance_summary();
        let mut rankings = Vec::new();

        for func_idx in 0..self.function_names.len() {
            let mut func_results: Vec<(usize, f64)> = summary.iter()
                .enumerate()
                .map(|(opt_idx, opt_summary)| (opt_idx, opt_summary[func_idx]))
                .filter(|(_, value)| !value.is_nan())
                .collect();

            // Sort by performance (ascending)
            func_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Extract rankings
            let mut func_rankings = vec![usize::MAX; summary.len()];
            for (rank, (opt_idx, _)) in func_results.iter().enumerate() {
                func_rankings[*opt_idx] = rank + 1; // 1-based ranking
            }

            rankings.push(func_rankings);
        }

        rankings
    }
}

/// Simple benchmark functions for testing
pub mod benchmark_functions {
    use super::*;

    /// Rosenbrock function (classic optimization benchmark)
    pub struct Rosenbrock {
        pub space: HyperparameterSpace,
    }

    impl Rosenbrock {
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl Default for Rosenbrock {
        fn default() -> Self {
            let mut space = HyperparameterSpace::new();
            space.add_parameter(super::super::space::Hyperparameter::Float {
                name: "x".to_string(),
                min: -2.0,
                max: 2.0,
                log_scale: false,
            });
            space.add_parameter(super::super::space::Hyperparameter::Float {
                name: "y".to_string(),
                min: -2.0,
                max: 2.0,
                log_scale: false,
            });

            Self { space }
        }
    }

    impl BenchmarkFunction for Rosenbrock {
        fn evaluate(&self, config: &HyperparameterConfig) -> Result<f64> {
            let x = config.get_float("x", 0.0);
            let y = config.get_float("y", 0.0);

            let term1 = (1.0 - x).powi(2);
            let term2 = 100.0 * (y - x.powi(2)).powi(2);

            Ok(term1 + term2)
        }

        fn name(&self) -> &str {
            "Rosenbrock"
        }

        fn optimal_value(&self) -> f64 {
            0.0
        }

        fn space(&self) -> &HyperparameterSpace {
            &self.space
        }
    }

    /// Sphere function (simple quadratic)
    pub struct Sphere {
        pub space: HyperparameterSpace,
    }

    impl Sphere {
        pub fn new(dimension: usize) -> Self {
            let mut space = HyperparameterSpace::new();

            for i in 0..dimension {
                space.add_parameter(super::super::space::Hyperparameter::Float {
                    name: format!("x{}", i),
                    min: -5.0,
                    max: 5.0,
                    log_scale: false,
                });
            }

            Self { space }
        }
    }

    impl BenchmarkFunction for Sphere {
        fn evaluate(&self, config: &HyperparameterConfig) -> Result<f64> {
            let mut sum = 0.0;

            // Sum of squares for all parameters
            for param in &self.space.parameters {
                let value = config.get_float(param.name(), 0.0);
                sum += value * value;
            }

            Ok(sum)
        }

        fn name(&self) -> &str {
            "Sphere"
        }

        fn optimal_value(&self) -> f64 {
            0.0
        }

        fn space(&self) -> &HyperparameterSpace {
            &self.space
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::benchmark_functions::{Rosenbrock, Sphere};

    #[test]
    fn test_optimizer_interface() {
        let space = HyperparameterSpace::neural_network_space();
        let mut optimizer = HyperparameterOptimizer::new(space);

        // Add a simple algorithm (Bayesian)
        let bo = crate::hpo::BayesianOptimizer::new(optimizer.space.clone());
        optimizer.add_algorithm(HPOptimizer::Bayesian(bo));

        let objective = |_config: &HyperparameterConfig| Ok(1.0);
        let result = optimizer.optimize(objective).unwrap();

        // Should respect the budget (50 evaluations)
        assert_eq!(result.evaluations, 50);
    }

    #[test]
    fn test_benchmark_functions() {
        let rosenbrock = Rosenbrock::new();
        let sphere = Sphere::new(2);

        // Test Rosenbrock
        let mut config = HyperparameterConfig::new();
        config.set("x".to_string(), super::super::space::HyperparameterValue::Float(1.0));
        config.set("y".to_string(), super::super::space::HyperparameterValue::Float(1.0));

        let value = rosenbrock.evaluate(&config).unwrap();
        assert_eq!(value, 0.0); // Rosenbrock minimum at (1,1)

        // Test Sphere
        let mut config2 = HyperparameterConfig::new();
        config2.set("x0".to_string(), super::super::space::HyperparameterValue::Float(0.0));
        config2.set("x1".to_string(), super::super::space::HyperparameterValue::Float(0.0));

        let value2 = sphere.evaluate(&config2).unwrap();
        assert_eq!(value2, 0.0); // Sphere minimum at origin
    }

    #[test]
    fn test_benchmark_results() {
        let mut results = BenchmarkResults::new();

        let opt_result = OptimizationResult {
            best_config: HyperparameterConfig::new(),
            best_value: 1.0,
            evaluations: 10,
            total_time: Duration::from_secs(1),
            history: Vec::new(),
        };

        results.add_result(0, 0, opt_result, Duration::from_secs(2));

        let summary = results.performance_summary();
        assert_eq!(summary[0][0], 1.0);
    }
}
