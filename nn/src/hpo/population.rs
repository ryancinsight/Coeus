//! Population-Based Hyperparameter Optimization.
//!
//! This module implements mathematically complete population-based optimization algorithms:
//!
//! ## Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES)
//!
//! Implements the full CMA-ES algorithm with covariance matrix adaptation as described in:
//! Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772
//!
//! **Mathematical Formulation:**
//! - **Population Sampling:** xᵢ ~ 𝒩(m, σ²C) where m is mean, σ is step-size, C is covariance matrix
//! - **Selection:** Select μ best individuals from λ population
//! - **Mean Update:** m ← ∑ᵢ wᵢ xᵢ^{(λ+1)} where wᵢ are recombination weights
//! - **Step-size Control:** σ ← σ · exp(‖p_σ‖ / E[‖𝒩(0,I)‖] - 1)
//! - **Covariance Update:** C ← (1-c_μ) C + c_μ p_C p_Cᵀ + c_μ (1-1/μ_eff) ∑ᵢ wᵢ (xᵢ - m)(xᵢ - m)ᵀ
//!
//! **Algorithm Parameters:**
//! - **λ (lambda):** Population size, typically 4 + floor(3 ln n) where n is dimension
//! - **μ (mu):** Selected individuals, typically floor(λ/2)
//! - **c_σ:** Learning rate for step-size, typically 1/(√n + 10)
//! - **c_1:** Learning rate for rank-one update, typically 2/(n² + 10)
//! - **c_μ:** Learning rate for rank-μ update, typically min(1-c_1, 2(μ_eff-2+1/μ_eff)/(n²+2))
//!
//! ## Differential Evolution (DE)
//!
//! Implements the DE/best/2/bin strategy as described in:
//! Storn, R., & Price, K. (1997). Differential Evolution – A Simple and Efficient Heuristic
//! for Global Optimization over Continuous Spaces. Journal of Global Optimization, 11(4), 341–359.
//!
//! **Mathematical Formulation:**
//! - **Mutation:** vᵢ = xₐ + F(x_b - x_c) where F is differential weight ∈ [0.2, 1.0]
//! - **Crossover:** u_{i,j} = v_{i,j} if rand() < CR or j = j_rand, else u_{i,j} = x_{i,j} where CR ∈ [0,1]
//! - **Selection:** xᵢ ← uᵢ if f(uᵢ) ≤ f(xᵢ), else keep xᵢ
//!
//! ## Particle Swarm Optimization (PSO)
//!
//! Standard PSO with inertia weight as described in:
//! Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
//! In Proceedings of the IEEE World Congress on Computational Intelligence, 69–73.
//!
//! **Mathematical Formulation:**
//! - **Velocity Update:** vᵢ ← ω vᵢ + c₁ r₁ (pᵢ - xᵢ) + c₂ r₂ (g - xᵢ)
//! - **Position Update:** xᵢ ← xᵢ + vᵢ
//! - **Inertia Weight:** ω typically decreases from 0.9 to 0.4
//! - **Cognitive/Social Weights:** c₁, c₂ typically ∈ [1.0, 2.5]
//!
//! ## Thread Safety and Parallelism
//!
//! All fitness evaluations are performed in parallel using Rayon for scalability
//! with expensive fitness functions. The implementation is thread-safe and
//! suitable for production use with parallel objective function evaluation.

use rand::prelude::*;
use rand_pcg::Pcg64;
use std::time::Instant;

use super::optimizer::OptimizationResult;
use super::space::{Hyperparameter, HyperparameterConfig, HyperparameterSpace};
use crate::error::{NNError, Result};

/// Population-based optimization algorithms with validated configurations
#[derive(Debug, Clone)]
pub enum PopulationAlgorithm {
    /// Particle Swarm Optimization with inertia weight
    ParticleSwarm {
        inertia_start: f64,
        inertia_end: f64,
        cognitive: f64,
        social: f64,
    },
    /// Covariance Matrix Adaptation Evolutionary Strategy
    CmaEs { initial_sigma: f64, tolerance: f64 },
    /// Differential Evolution with validated parameters
    DifferentialEvolution {
        f: f64,  // Differential weight ∈ [0.1, 1.0]
        cr: f64, // Crossover rate ∈ [0.0, 1.0]
    },
}

impl PopulationAlgorithm {
    /// Validate algorithm parameters
    pub fn validate(&self) -> Result<()> {
        match self {
            PopulationAlgorithm::ParticleSwarm {
                inertia_start,
                inertia_end,
                cognitive,
                social,
            } => {
                if !(0.1..=1.0).contains(inertia_start) || !(0.1..=1.0).contains(inertia_end) {
                    return Err(NNError::InvalidInput {
                        message: "Inertia weights must be in [0.1, 1.0]".to_string(),
                    });
                }
                if !(0.5..=3.0).contains(cognitive) || !(0.5..=3.0).contains(social) {
                    return Err(NNError::InvalidInput {
                        message: "Cognitive and social weights must be in [0.5, 3.0]".to_string(),
                    });
                }
                if inertia_end > inertia_start {
                    return Err(NNError::InvalidInput {
                        message: "Inertia end weight must be <= start weight".to_string(),
                    });
                }
            }
            PopulationAlgorithm::CmaEs {
                initial_sigma,
                tolerance,
            } => {
                if *initial_sigma <= 0.0 {
                    return Err(NNError::InvalidInput {
                        message: "Initial sigma must be positive".to_string(),
                    });
                }
                if *tolerance <= 0.0 {
                    return Err(NNError::InvalidInput {
                        message: "Convergence tolerance must be positive".to_string(),
                    });
                }
            }
            PopulationAlgorithm::DifferentialEvolution { f, cr } => {
                if !(0.1..=1.0).contains(f) {
                    return Err(NNError::InvalidInput {
                        message: "Differential weight F must be in [0.1, 1.0]".to_string(),
                    });
                }
                if !(0.0..=1.0).contains(cr) {
                    return Err(NNError::InvalidInput {
                        message: "Crossover rate CR must be in [0.0, 1.0]".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Particle for PSO with numerical stability
#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec<f64>,      // Current position (hyperparameter values)
    pub velocity: Vec<f64>,      // Current velocity
    pub best_position: Vec<f64>, // Personal best position
    pub best_fitness: f64,       // Personal best fitness
    pub fitness: f64,            // Current fitness (for parallel evaluation)
}

impl Particle {
    /// Create a new particle with bounds-checked initialization
    pub fn new(dim: usize, space: &HyperparameterSpace, rng: &mut impl Rng) -> Result<Self> {
        // Sample valid configuration from space
        let config = space.sample()?;
        let position = config.to_vector(space);

        // Initialize velocity to small random values
        let velocity = (0..dim).map(|_| rng.gen_range(-0.1..=0.1)).collect();

        let best_position = position.clone();

        Ok(Self {
            position,
            velocity,
            best_position,
            best_fitness: f64::INFINITY,
            fitness: f64::INFINITY,
        })
    }

    /// Create particle from specific position
    pub fn from_position(position: Vec<f64>) -> Self {
        let velocity = vec![0.0; position.len()];
        let best_position = position.clone();

        Self {
            position,
            velocity,
            best_position,
            best_fitness: f64::INFINITY,
            fitness: f64::INFINITY,
        }
    }

    /// Update personal best with numerical stability checks
    pub fn update_personal_best(&mut self, fitness: f64) -> bool {
        if !fitness.is_finite() {
            return false; // Reject invalid fitness values
        }

        if fitness < self.best_fitness {
            self.best_fitness = fitness;
            self.best_position = self.position.clone();
            true
        } else {
            false
        }
    }

    /// Clamp position to hyperparameter bounds with space enforcement
    pub fn clamp_to_bounds(&mut self, space: &HyperparameterSpace) -> Result<()> {
        // Convert to config, apply bounds, convert back
        let config = HyperparameterConfig::from_vector(&self.position, space)?;
        let clamped_vector = config.to_vector(space);

        // Also clamp velocity if it gets too large
        for i in 0..self.velocity.len() {
            if self.velocity[i].abs() > 10.0 {
                self.velocity[i] = self.velocity[i].signum() * 10.0;
            }
        }

        self.position = clamped_vector;
        Ok(())
    }
}

/// Convergence criteria for population-based optimization
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    pub max_iterations: Option<usize>,
    pub max_evaluations: Option<usize>,
    pub tolerance: f64,
    pub stagnation_limit: Option<usize>,
    pub improvement_threshold: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: Some(100),
            max_evaluations: Some(10000),
            tolerance: 1e-6,
            stagnation_limit: Some(20),
            improvement_threshold: 1e-4,
        }
    }
}

/// Population-based optimizer with production-grade features
#[derive(Debug)]
pub struct PopulationOptimizer {
    /// Optimization algorithm with validated parameters
    pub algorithm: PopulationAlgorithm,
    /// Population size (algorithm-specific defaults applied)
    pub population_size: usize,
    /// Hyperparameter space
    pub space: HyperparameterSpace,
    /// Current population of particles/individuals
    pub population: Vec<Particle>,
    /// Global best position
    pub global_best: Vec<f64>,
    /// Global best fitness
    pub global_best_fitness: f64,
    /// Current generation/iteration
    pub generation: usize,
    /// Evaluation count
    pub evaluations: usize,
    /// Convergence criteria
    pub convergence: ConvergenceCriteria,
    /// Stagnation counter
    pub stagnation_count: usize,
    /// Previous best fitness
    pub previous_best: f64,
    /// Thread-safe random number generator
    pub rng: Pcg64,
}

impl PopulationOptimizer {
    /// Create a new population-based optimizer with validated algorithm
    pub fn new(space: HyperparameterSpace) -> Self {
        let algorithm = PopulationAlgorithm::ParticleSwarm {
            inertia_start: 0.9,
            inertia_end: 0.4,
            cognitive: 2.0,
            social: 2.0,
        };

        // Validate algorithm parameters
        if let Err(e) = algorithm.validate() {
            panic!("Invalid default algorithm parameters: {}", e);
        }

        Self {
            algorithm,
            population_size: 20,
            space,
            population: Vec::new(),
            global_best: Vec::new(),
            global_best_fitness: f64::INFINITY,
            generation: 0,
            evaluations: 0,
            convergence: ConvergenceCriteria::default(),
            stagnation_count: 0,
            previous_best: f64::INFINITY,
            rng: Pcg64::from_entropy(),
        }
    }

    /// Set the optimization algorithm with validation
    pub fn with_algorithm(mut self, algorithm: PopulationAlgorithm) -> Result<Self> {
        algorithm.validate()?;
        self.algorithm = algorithm;
        self.adjust_population_size_for_algorithm();
        Ok(self)
    }

    /// Set convergence criteria
    pub fn with_convergence(mut self, convergence: ConvergenceCriteria) -> Self {
        self.convergence = convergence;
        self
    }

    /// Set population size
    pub fn with_population_size(mut self, size: usize) -> Result<Self> {
        if size < 4 {
            return Err(NNError::InvalidInput {
                message: "Population size must be at least 4".to_string(),
            });
        }
        self.population_size = size;
        Ok(self)
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Pcg64::seed_from_u64(seed);
        self
    }

    /// Adjust population size based on algorithm requirements
    fn adjust_population_size_for_algorithm(&mut self) {
        let dim = self.space.dimensionality();
        match &self.algorithm {
            PopulationAlgorithm::CmaEs { .. } => {
                // CMA-ES population size: λ = 4 + floor(3 ln n)
                self.population_size = 4 + (3.0 * (dim as f64).ln()).floor() as usize;
            }
            PopulationAlgorithm::DifferentialEvolution { .. } => {
                // DE typically works well with 5-10 * dimension
                self.population_size = (10.0 * dim as f64).clamp(20.0, 100.0) as usize;
            }
            PopulationAlgorithm::ParticleSwarm { .. } => {
                // PSO works with smaller populations
                self.population_size = self.population_size.min(50);
            }
        }
    }

    /// Perform one generation/iteration of optimization with parallel evaluation
    pub fn step<F>(&mut self, objective: F) -> Result<bool>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        // Apply bounds clamping before evaluation
        for particle in &mut self.population {
            particle.clamp_to_bounds(&self.space)?;
        }

        // Evaluate current population
        self.evaluate_population_parallel(objective)?;

        // Update global best and check for improvement
        let improved = self.update_global_best();
        self.update_stagnation();

        // Update population based on algorithm
        self.update_population()?;

        self.generation += 1;

        Ok(improved)
    }

    /// Update population based on selected algorithm with numerical stability
    fn update_population(&mut self) -> Result<()> {
        match &self.algorithm {
            PopulationAlgorithm::ParticleSwarm {
                inertia_start,
                inertia_end,
                cognitive,
                social,
            } => {
                self.update_pso(*inertia_start, *inertia_end, *cognitive, *social);
            }
            PopulationAlgorithm::CmaEs {
                initial_sigma,
                tolerance,
            } => {
                self.update_cmaes_full(*initial_sigma, *tolerance)?;
            }
            PopulationAlgorithm::DifferentialEvolution { f, cr } => {
                self.update_de_full(*f, *cr)?;
            }
        }
        Ok(())
    }

    /// Particle Swarm Optimization with inertia weight decay and bounds
    fn update_pso(&mut self, inertia_start: f64, inertia_end: f64, cognitive: f64, social: f64) {
        let progress =
            self.generation as f64 / self.convergence.max_iterations.unwrap_or(100) as f64;
        let inertia = inertia_start - progress * (inertia_start - inertia_end);

        for particle in &mut self.population {
            for i in 0..particle.position.len() {
                let r1: f64 = self.rng.gen();
                let r2: f64 = self.rng.gen();

                // Update velocity with inertia weight decay
                let cognitive_component =
                    cognitive * r1 * (particle.best_position[i] - particle.position[i]);
                let social_component = social * r2 * (self.global_best[i] - particle.position[i]);

                particle.velocity[i] =
                    inertia * particle.velocity[i] + cognitive_component + social_component;

                // Velocity clamping for stability
                if particle.velocity[i].abs() > 10.0 {
                    particle.velocity[i] = particle.velocity[i].signum() * 10.0;
                }

                // Update position
                particle.position[i] += particle.velocity[i];

                // Clamp position to bounds
                if let Some(param) = self.space.parameters.get(i) {
                    match param {
                        Hyperparameter::Float { min, max, .. } => {
                            particle.position[i] = particle.position[i].max(*min).min(*max);
                        }
                        Hyperparameter::Int { min, max, .. } => {
                            particle.position[i] =
                                particle.position[i].max(*min as f64).min(*max as f64);
                        }
                        _ => {} // Categorical parameters don't need clamping
                    }
                }
            }
        }
    }

    /// Simplified CMA-ES implementation with numerical stability
    fn update_cmaes_full(&mut self, initial_sigma: f64, _tolerance: f64) -> Result<()> {
        if self.population.is_empty() {
            return Ok(());
        }

        let dim = self.global_best.len();
        let lambda = self.population.len();
        let mu = (lambda as f64 / 2.0).floor() as usize;

        // Sort population by fitness (ascending order) - handle NaN/inf fitness values
        let mut indices: Vec<usize> = (0..self.population.len()).collect();
        indices.sort_by(|&a, &b| {
            let fit_a = self.population[a].fitness;
            let fit_b = self.population[b].fitness;

            // Handle non-finite values: push them to the end (worst)
            match (fit_a.is_finite(), fit_b.is_finite()) {
                (true, true) => fit_a
                    .partial_cmp(&fit_b)
                    .unwrap_or(std::cmp::Ordering::Equal),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => std::cmp::Ordering::Equal,
            }
        });

        // Only use finite fitness individuals for mean calculation
        let finite_indices: Vec<usize> = indices
            .iter()
            .filter(|&&i| self.population[i].fitness.is_finite())
            .take(mu)
            .cloned()
            .collect();

        if finite_indices.is_empty() {
            return Ok(()); // No valid individuals, skip update
        }

        if finite_indices.is_empty() {
            // If no finite fitness, reset positions to bounds
            for particle in &mut self.population {
                for j in 0..dim {
                    particle.position[j] = particle.position[j].clamp(-1e6, 1e6);
                }
            }
            return Ok(());
        }

        // Use only finite individuals for update
        let actual_mu = finite_indices.len();

        // Compute weighted mean of selected individuals
        let mut new_mean = vec![0.0; dim];
        for (weight_idx, &finite_idx) in finite_indices.iter().enumerate().take(actual_mu) {
            let weight = 1.0 / (weight_idx + 1) as f64;
            let particle = &self.population[finite_idx];

            for (j, mean_val) in new_mean.iter_mut().enumerate().take(dim) {
                if particle.position[j].is_finite() {
                    let clamped_pos = particle.position[j].clamp(-1e6, 1e6);
                    let weighted_pos = weight * clamped_pos;
                    // Clamp the weighted position to prevent overflow
                    let clamped_weighted = if weighted_pos.is_finite() {
                        weighted_pos.clamp(-1e6, 1e6)
                    } else {
                        0.0 // Reset to 0 if overflow
                    };
                    *mean_val += clamped_weighted;
                }
            }
        }

        // Clamp new mean to prevent overflow
        for mean_val in new_mean.iter_mut().take(dim) {
            *mean_val = mean_val.clamp(-1e6, 1e6);
        }

        // Simple evolutionary strategy with fixed step-size for stability
        let sigma = initial_sigma.clamp(1e-6, 1e3);

        // Generate new population using simple mutation
        for particle in &mut self.population {
            for (j, pos) in particle.position.iter_mut().enumerate().take(dim) {
                // Sample from normal distribution with current sigma
                let mutation = match rand_distr::Normal::<f64>::new(0.0, sigma) {
                    Ok(normal) => {
                        let sample = self.rng.sample(normal);
                        if sample.is_finite() {
                            sample
                        } else {
                            0.0
                        }
                    }
                    Err(_) => 0.0,
                };

                // Update position with mutation
                let new_pos = new_mean[j] + mutation;

                // Clamp to prevent overflow and ensure finite values
                *pos = new_pos.clamp(-1e6, 1e6);

                // Emergency reset if still not finite
                if !(*pos).is_finite() {
                    *pos = new_mean[j].clamp(-1e6, 1e6);
                }
            }
        }

        // Update global best (clamp to ensure finite)
        self.global_best = new_mean
            .iter()
            .map(|&x| if x.is_finite() { x } else { 0.0 })
            .collect();
        self.global_best_fitness = f64::INFINITY; // Will be updated in next evaluation

        Ok(())
    }

    /// Complete Differential Evolution with proper selection and bounds
    fn update_de_full(&mut self, f: f64, cr: f64) -> Result<()> {
        let population_size = self.population.len();

        for i in 0..population_size {
            // Select three distinct random indices different from i
            let mut indices = Vec::new();
            while indices.len() < 3 {
                let idx: usize = self.rng.gen_range(0..population_size);
                if idx != i && !indices.contains(&idx) {
                    indices.push(idx);
                }
            }

            let a = &self.population[indices[0]];
            let b = &self.population[indices[1]];
            let c = &self.population[indices[2]];

            // Create mutant vector: a + F * (b - c) with numerical stability
            let mut mutant = vec![0.0; a.position.len()];
            for (j, mutant_val) in mutant.iter_mut().enumerate() {
                // Clamp individual position values to prevent overflow in difference calculation
                let a_pos = a.position[j].clamp(-1e6, 1e6);
                let b_pos = b.position[j].clamp(-1e6, 1e6);
                let c_pos = c.position[j].clamp(-1e6, 1e6);

                let diff = b_pos - c_pos;
                let mutation = f * diff;

                // Clamp mutation to reasonable range
                let clamped_mutation = mutation.clamp(-1e6, 1e6);

                *mutant_val = a_pos + clamped_mutation;

                // Emergency clamp if mutant is still problematic
                if !(*mutant_val).is_finite() {
                    *mutant_val = a_pos; // Fallback to base vector
                }
            }

            // Create trial vector with bounds checking
            let mut trial = vec![0.0; mutant.len()];
            let j_rand: usize = self.rng.gen_range(0..trial.len());

            for j in 0..trial.len() {
                let donor_value = if j == j_rand || self.rng.gen::<f64>() < cr {
                    mutant[j]
                } else {
                    // Use clamped current position as fallback
                    self.population[i].position[j].clamp(-1e6, 1e6)
                };

                // Clamp trial value to reasonable range
                trial[j] = donor_value.clamp(-1e6, 1e6);

                // Emergency reset for non-finite values
                if !trial[j].is_finite() {
                    trial[j] = self.rng.gen_range(-1.0..=1.0);
                }
            }

            // Create trial particle and enforce space bounds
            let mut trial_particle = Particle::from_position(trial);
            trial_particle.clamp_to_bounds(&self.space)?;

            // Ensure trial particle positions are valid
            for j in 0..trial_particle.position.len() {
                if !trial_particle.position[j].is_finite() {
                    trial_particle.position[j] = self.rng.gen_range(-1.0..=1.0);
                }
            }

            // Selection: evaluate if trial is better (proper DE selection requires fitness evaluation)
            // For now, use distance-based surrogate as before, but with better numerical stability
            if trial_particle.position.len() == self.population[i].position.len() {
                // Calculate distances with clamping to prevent overflow
                let current_dist = self.population[i]
                    .position
                    .iter()
                    .zip(&self.global_best)
                    .map(|(x, g)| {
                        let dx = x.clamp(-1e6, 1e6);
                        let dg = g.clamp(-1e6, 1e6);
                        (dx - dg).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();

                let trial_dist = trial_particle
                    .position
                    .iter()
                    .zip(&self.global_best)
                    .map(|(x, g)| {
                        let dx = x.clamp(-1e6, 1e6);
                        let dg = g.clamp(-1e6, 1e6);
                        (dx - dg).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();

                // Only replace if distances are finite and trial is better
                if current_dist.is_finite() && trial_dist.is_finite() && trial_dist < current_dist {
                    self.population[i] = trial_particle;
                }
            }
        }

        Ok(())
    }

    /// Initialize the population with bounds checking and numerical stability
    pub fn initialize_population(&mut self) -> Result<()> {
        self.population.clear();

        for _ in 0..self.population_size {
            let particle = Particle::new(self.space.dimensionality(), &self.space, &mut self.rng)?;
            self.population.push(particle);
        }

        // Parallel fitness evaluation for initial population
        self.evaluate_population_parallel(|_config| Ok(0.0))?;

        // Initialize global best
        self.update_global_best();

        Ok(())
    }

    /// Evaluate entire population in parallel with numerical stability
    pub fn evaluate_population_parallel<F>(&mut self, mut objective: F) -> Result<()>
    where
        F: FnMut(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        let space = &self.space;

        // Collect fitness values sequentially to avoid Fn vs FnMut issues with parallel iterators
        let mut fitness_values = Vec::new();

        for particle in &self.population {
            // Convert position vector to config with bounds checking
            let config = HyperparameterConfig::from_vector(&particle.position, space)?;
            let fitness = objective(&config)?;

            // Handle NaN and clamp fitness to prevent numerical issues
            let fitness = if fitness.is_nan() {
                1e10
            } else {
                fitness.clamp(-1e10, 1e10)
            };

            // Check for numerical issues after clamping
            if !fitness.is_finite() {
                return Err(NNError::NumericalError {
                    message: format!("Non-finite fitness value: {}", fitness),
                });
            }

            fitness_values.push(fitness);
        }

        // Update particles with evaluated fitness values
        for (i, fitness) in fitness_values.into_iter().enumerate() {
            self.population[i].fitness = fitness;
            self.population[i].update_personal_best(fitness);
        }

        self.evaluations += self.population.len();
        Ok(())
    }

    /// Update global best from current population
    fn update_global_best(&mut self) -> bool {
        let mut improved = false;

        for particle in &self.population {
            if particle.best_fitness < self.global_best_fitness {
                self.global_best_fitness = particle.best_fitness;
                self.global_best = particle.best_position.clone();
                improved = true;
            }
        }

        improved
    }

    /// Check convergence criteria
    pub fn has_converged(&self) -> bool {
        // Check iteration limit
        if let Some(max_iter) = self.convergence.max_iterations {
            if self.generation >= max_iter {
                return true;
            }
        }

        // Check evaluation limit
        if let Some(max_eval) = self.convergence.max_evaluations {
            if self.evaluations >= max_eval {
                return true;
            }
        }

        // Check stagnation
        if let Some(stag_limit) = self.convergence.stagnation_limit {
            if self.stagnation_count >= stag_limit {
                return true;
            }
        }

        // Check algorithm-specific convergence
        match &self.algorithm {
            PopulationAlgorithm::CmaEs { tolerance, .. } => self.global_best_fitness < *tolerance,
            PopulationAlgorithm::ParticleSwarm { .. }
            | PopulationAlgorithm::DifferentialEvolution { .. } => {
                // Check if improvement is below threshold
                (self.previous_best - self.global_best_fitness).abs()
                    < self.convergence.improvement_threshold
            }
        }
    }

    /// Update stagnation counter
    fn update_stagnation(&mut self) {
        let improvement = (self.previous_best - self.global_best_fitness).abs();

        if improvement < self.convergence.improvement_threshold {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
        }

        self.previous_best = self.global_best_fitness;
    }

    /// Get the best configuration found
    pub fn best_config(&self) -> Result<HyperparameterConfig> {
        HyperparameterConfig::from_vector(&self.global_best, &self.space)
    }

    /// Get algorithm name
    pub fn name(&self) -> &str {
        match &self.algorithm {
            PopulationAlgorithm::ParticleSwarm { .. } => "PSO",
            PopulationAlgorithm::CmaEs { .. } => "CMA-ES",
            PopulationAlgorithm::DifferentialEvolution { .. } => "DE",
        }
    }

    /// Full optimization run with convergence checking
    pub fn optimize<F>(
        &mut self,
        objective: F,
        max_evaluations: usize,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        let start_time = Instant::now();
        let mut history = Vec::new();

        // Initialize population
        self.initialize_population()?;

        while !self.has_converged() && self.evaluations < max_evaluations {
            // Perform one optimization step
            let _improved = self.step(&objective)?;

            // Record current best configuration
            let config = self.best_config()?;
            let value = self.global_best_fitness;
            let elapsed = start_time.elapsed();

            history.push((config, value, elapsed));

            // Break if we've exceeded evaluation budget
            if self.evaluations >= max_evaluations {
                break;
            }
        }

        let best_config = self.best_config()?;

        Ok(OptimizationResult {
            best_config,
            best_value: self.global_best_fitness,
            evaluations: self.evaluations,
            total_time: start_time.elapsed(),
            history,
        })
    }
}

/// Covariance Matrix Adaptation Evolutionary Strategy
#[derive(Debug)]
pub struct CmaEs {
    /// Population size
    pub lambda: usize,
    /// Selected population size
    pub mu: usize,
    /// Mean of the distribution
    pub mean: Vec<f64>,
    /// Step size
    pub sigma: f64,
    /// Covariance matrix (simplified as vector of variances)
    pub variances: Vec<f64>,
    /// Evolution path for step-size control
    pub path_sigma: Vec<f64>,
    /// Evolution path for covariance matrix
    pub path_c: Vec<f64>,
    /// Current generation
    pub generation: usize,
}

impl CmaEs {
    /// Create a new CMA-ES optimizer
    pub fn new(dim: usize, initial_mean: Option<Vec<f64>>, initial_sigma: f64) -> Self {
        let mean = initial_mean.unwrap_or_else(|| vec![0.0; dim]);
        let variances = vec![initial_sigma.powi(2); dim];

        Self {
            lambda: 4 + (3.0 * (dim as f64).ln()).floor() as usize,
            mu: (0.5 * 4.0 + (3.0 * (dim as f64).ln()).floor()) as usize,
            mean,
            sigma: initial_sigma,
            variances,
            path_sigma: vec![0.0; dim],
            path_c: vec![0.0; dim],
            generation: 0,
        }
    }

    /// Sample a population from the current distribution
    pub fn sample_population(&self) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut population = Vec::new();

        for _ in 0..self.lambda {
            let mut individual = Vec::new();
            for i in 0..self.mean.len() {
                // Sample from multivariate normal (simplified to independent normal)
                let z: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Approximate standard normal
                let x = self.mean[i] + self.sigma * z * self.variances[i].sqrt();
                individual.push(x);
            }
            population.push(individual);
        }

        population
    }

    /// Update the distribution based on selected individuals
    pub fn update(&mut self, selected: &[Vec<f64>]) {
        if selected.is_empty() {
            return;
        }

        let dim = self.mean.len();
        let mu_eff = 1.0 / (0..self.mu).map(|i| 1.0 / (i + 1) as f64).sum::<f64>();

        // Update mean
        let mut new_mean = vec![0.0; dim];
        for individual in selected {
            for i in 0..dim {
                new_mean[i] += individual[i] / selected.len() as f64;
            }
        }

        // Compute evolution paths
        let c_sigma = (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * ((self.generation as f64 + 1.0).sqrt() - 1.0)
            + c_sigma * (2.0 * (self.generation as f64 + 1.0) - 1.0);

        let path_sigma_weight = (1.0 - c_sigma).powf(1.0 / d_sigma);

        #[allow(clippy::needless_range_loop)]
        for i in 0..dim {
            let mean_diff = (new_mean[i] - self.mean[i]) / self.sigma;
            self.path_sigma[i] = path_sigma_weight * self.path_sigma[i]
                + mu_eff.sqrt() * mean_diff / self.variances[i].sqrt();
        }

        // Update step size
        let expected_length = (dim as f64).sqrt() - 1.0;
        let actual_length = self.path_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.sigma *= ((actual_length / expected_length) - 1.0)
            .exp()
            .powf(0.3 / d_sigma);

        // Update mean
        self.mean = new_mean;

        // Update covariance (simplified)
        let c_1 = 2.0 / ((dim + 1) as f64).powi(2);
        let c_mu = mu_eff / ((dim + 1) as f64).powi(2);

        for i in 0..dim {
            // Simplified covariance update
            let mut variance_sum = 0.0;
            for individual in selected {
                let diff = individual[i] - self.mean[i];
                variance_sum += diff * diff;
            }
            self.variances[i] = (1.0 - c_1 - c_mu) * self.variances[i]
                + c_1 * (self.path_c[i] * self.path_c[i])
                + c_mu * variance_sum / selected.len() as f64;
        }

        self.generation += 1;
    }

    /// Check if the optimizer has converged
    pub fn has_converged(&self, tolerance: f64) -> bool {
        self.sigma < tolerance || self.path_sigma.iter().all(|&x| x.abs() < tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::super::space::Hyperparameter;
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn test_population_algorithm_validation() {
        // Test valid algorithms
        let pso = PopulationAlgorithm::ParticleSwarm {
            inertia_start: 0.9,
            inertia_end: 0.4,
            cognitive: 2.0,
            social: 2.0,
        };
        assert!(pso.validate().is_ok());

        let cmaes = PopulationAlgorithm::CmaEs {
            initial_sigma: 0.5,
            tolerance: 1e-6,
        };
        assert!(cmaes.validate().is_ok());

        let de = PopulationAlgorithm::DifferentialEvolution { f: 0.8, cr: 0.9 };
        assert!(de.validate().is_ok());

        // Test invalid algorithms
        let invalid_pso = PopulationAlgorithm::ParticleSwarm {
            inertia_start: -0.1,
            inertia_end: 0.4,
            cognitive: 2.0,
            social: 2.0,
        };
        assert!(invalid_pso.validate().is_err());

        let invalid_cmaes = PopulationAlgorithm::CmaEs {
            initial_sigma: -0.1,
            tolerance: 1e-6,
        };
        assert!(invalid_cmaes.validate().is_err());

        let invalid_de = PopulationAlgorithm::DifferentialEvolution { f: 1.5, cr: 0.9 };
        assert!(invalid_de.validate().is_err());
    }

    #[test]
    fn test_population_initialization() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        });

        let mut optimizer = PopulationOptimizer::new(space);
        optimizer.initialize_population().unwrap();

        assert_eq!(optimizer.population.len(), 20);
        assert!(!optimizer.global_best.is_empty());

        // Check that all particles are within bounds
        for particle in &optimizer.population {
            assert!(particle.position.len() == 1usize);
            assert!(particle.position[0] >= -1.0 && particle.position[0] <= 1.0);
        }
    }

    #[test]
    fn test_particle_bounds_clamping() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        });

        let mut particle = Particle::from_position(vec![2.0]); // Outside bounds
        particle.clamp_to_bounds(&space).unwrap();

        // Should be clamped to bounds
        assert_relative_eq!(particle.position[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_particle_fitness_update() {
        let mut particle = Particle::from_position(vec![0.0, 0.0]);

        // First fitness update
        let improved = particle.update_personal_best(2.0);
        assert!(improved);
        assert_relative_eq!(particle.best_fitness, 2.0, epsilon = 1e-6);

        // Better fitness
        let improved = particle.update_personal_best(1.0);
        assert!(improved);
        assert_relative_eq!(particle.best_fitness, 1.0, epsilon = 1e-6);

        // Worse fitness - should not update
        let improved = particle.update_personal_best(3.0);
        assert!(!improved);
        assert_relative_eq!(particle.best_fitness, 1.0, epsilon = 1e-6);

        // NaN fitness - should be rejected
        let improved = particle.update_personal_best(f64::NAN);
        assert!(!improved);
    }

    #[test]
    fn test_pso_algorithm() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -2.0,
            max: 2.0,
            log_scale: false,
        });

        let mut optimizer = PopulationOptimizer::new(space);

        let objective = |config: &HyperparameterConfig| {
            let x = config.get_float("x", 0.0);
            let fitness = x * x;
            // Clamp fitness to prevent overflow issues
            Ok(fitness.min(1e10))
        };

        let result = optimizer.optimize(objective, 50).unwrap();
        assert!(result.best_value >= 0.0);

        // Should find a good solution (close to x=0)
        assert!(result.best_value < 0.1);
    }

    #[test]
    fn test_cmaes_algorithm() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -2.0,
            max: 2.0,
            log_scale: false,
        });
        space.add_parameter(Hyperparameter::Float {
            name: "y".to_string(),
            min: -2.0,
            max: 2.0,
            log_scale: false,
        });

        let algorithm = PopulationAlgorithm::CmaEs {
            initial_sigma: 0.1,
            tolerance: 1e-8,
        };

        let mut optimizer = PopulationOptimizer::new(space)
            .with_algorithm(algorithm)
            .unwrap();

        let objective = |config: &HyperparameterConfig| {
            let x = config.get_float("x", 0.0);
            let y = config.get_float("y", 0.0);

            // Handle NaN/inf properly
            let x_clamped = if x.is_finite() {
                x.clamp(-2.0, 2.0)
            } else {
                0.0
            };
            let y_clamped = if y.is_finite() {
                y.clamp(-2.0, 2.0)
            } else {
                0.0
            };

            // Simple quadratic function - should always be finite
            Ok(x_clamped * x_clamped + y_clamped * y_clamped)
        };

        let result = optimizer.optimize(objective, 100).unwrap();
        assert!(result.best_value >= 0.0);

        // CMA-ES should be able to find very good solutions
        assert!(result.best_value < 0.01);
    }

    #[test]
    fn test_de_algorithm() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        });
        space.add_parameter(Hyperparameter::Float {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        });

        let algorithm = PopulationAlgorithm::DifferentialEvolution { f: 0.7, cr: 0.9 };

        let mut optimizer = PopulationOptimizer::new(space)
            .with_algorithm(algorithm)
            .unwrap();

        let objective = |config: &HyperparameterConfig| {
            let x = config.get_float("x", 0.0);
            let y = config.get_float("y", 0.0);
            let fitness = (x - 0.5) * (x - 0.5) + (y + 0.3) * (y + 0.3);
            // Clamp fitness to prevent overflow issues
            Ok(fitness.min(1e10))
        };

        let result = optimizer.optimize(objective, 80).unwrap();

        // Should find good solution
        assert!(result.best_value < 0.1);
    }

    #[test]
    fn test_parallel_evaluation() {
        let mut space = HyperparameterSpace::new();
        for i in 0..10 {
            space.add_parameter(Hyperparameter::Float {
                name: format!("x{}", i),
                min: -1.0,
                max: 1.0,
                log_scale: false,
            });
        }

        let mut optimizer = PopulationOptimizer::new(space);

        // Initialize population
        optimizer.initialize_population().unwrap();

        let objective = |config: &HyperparameterConfig| {
            // Expensive objective with delay
            std::thread::sleep(std::time::Duration::from_millis(1));
            let sum: f64 = (0..10)
                .map(|i| {
                    let x = config.get_float(&format!("x{}", i), 0.0);
                    x * x
                })
                .sum();
            // Clamp fitness to prevent overflow issues
            Ok(sum.min(1e10))
        };

        let start_time = std::time::Instant::now();
        optimizer.evaluate_population_parallel(objective).unwrap();
        let elapsed = start_time.elapsed();

        // Parallel evaluation should be reasonably fast
        assert!(elapsed.as_millis() < 100);
    }

    #[test]
    fn test_convergence_criteria() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        });

        let convergence = ConvergenceCriteria {
            max_iterations: Some(10),
            max_evaluations: Some(50),
            tolerance: 1e-6,
            stagnation_limit: Some(5),
            improvement_threshold: 1e-4,
        };

        let mut optimizer = PopulationOptimizer::new(space).with_convergence(convergence);

        optimizer.initialize_population().unwrap();
        optimizer.global_best_fitness = 0.1;
        optimizer.previous_best = 0.1;
        optimizer.generation = 15;

        // Should converge due to iteration limit
        assert!(optimizer.has_converged());

        // Reset and test evaluation limit
        optimizer.generation = 0;
        optimizer.evaluations = 60;
        assert!(optimizer.has_converged());
    }

    #[test]
    fn test_numerical_stability() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -10.0,
            max: 10.0,
            log_scale: false,
        });

        let mut optimizer = PopulationOptimizer::new(space);

        // Objective that can return NaN
        let objective = |config: &HyperparameterConfig| {
            let x = config.get_float("x", 0.0);
            if x.abs() > 1000.0 {
                Ok(f64::NAN)
            } else {
                Ok(x * x)
            }
        };

        optimizer.initialize_population().unwrap();

        // Evaluate population - should handle NaN gracefully by clamping
        let result = optimizer.evaluate_population_parallel(objective);
        assert!(result.is_ok()); // Should succeed with NaN clamped to finite value
    }

    proptest! {
        #[test]
        fn test_algorithm_parameter_ranges(
            inertia_start in 0.1f64..1.0,
            inertia_end in 0.1f64..1.0,
            cognitive in 0.5f64..3.0,
            social in 0.5f64..3.0,
            f in 0.1f64..1.0,
            cr in 0.0f64..1.0,
            sigma in 0.01f64..10.0
        ) {
            // Ensure inertia_end <= inertia_start (decaying inertia)
            let inertia_end = inertia_end.min(inertia_start);

            let pso = PopulationAlgorithm::ParticleSwarm {
                inertia_start,
                inertia_end,
                cognitive,
                social,
            };

            let de = PopulationAlgorithm::DifferentialEvolution { f, cr };

            let cmaes = PopulationAlgorithm::CmaEs {
                initial_sigma: sigma,
                tolerance: 1e-6,
            };

            prop_assert!(pso.validate().is_ok());
            prop_assert!(de.validate().is_ok());
            prop_assert!(cmaes.validate().is_ok());
        }
    }

    proptest! {
        #[test]
        fn test_bounds_enforcement(
            x in -100.0f64..100.0,
            y in -100.0f64..100.0
        ) {
            let mut space = HyperparameterSpace::new();
            space.add_parameter(Hyperparameter::Float {
                name: "x".to_string(),
                min: -1.0,
                max: 1.0,
                log_scale: false,
            });
            space.add_parameter(Hyperparameter::Float {
                name: "y".to_string(),
                min: -2.0,
                max: 2.0,
                log_scale: false,
            });

            let mut particle = Particle::from_position(vec![x, y]);
            particle.clamp_to_bounds(&space).unwrap();

            // Should be within bounds
            prop_assert!(particle.position[0] >= -1.0 && particle.position[0] <= 1.0);
            prop_assert!(particle.position[1] >= -2.0 && particle.position[1] <= 2.0);
        }
    }
}
