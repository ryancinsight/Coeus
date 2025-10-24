//! Evolutionary Neural Architecture Search.
//!
//! This module implements evolutionary algorithms for neural architecture search,
//! including population-based evolution with mutation and crossover operators.

use std::cmp::Ordering;

use rand::Rng;

use super::search_space::{Architecture, ArchitectureSpace};
use crate::error::{NNError, Result};

/// Individual in the evolutionary population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Architecture representation
    pub architecture: Architecture,
    /// Fitness score (higher is better)
    pub fitness: f64,
    /// Age of the individual (generations survived)
    pub age: usize,
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness && self.age == other.age
    }
}

impl Eq for Individual {}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Individual {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by fitness first, then by age (younger is better for tie-breaking)
        match self.fitness.partial_cmp(&other.fitness) {
            Some(Ordering::Equal) => other.age.cmp(&self.age), // Reverse age comparison
            Some(ord) => ord,
            None => Ordering::Equal,
        }
    }
}

/// Evolutionary Neural Architecture Search algorithm
#[derive(Debug)]
pub struct EvolutionaryNAS {
    /// Search space definition
    pub search_space: ArchitectureSpace,
    /// Population size
    pub population_size: usize,
    /// Number of elite individuals preserved each generation
    pub num_elites: usize,
    /// Mutation probability
    pub mutation_prob: f64,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Maximum generations
    pub max_generations: usize,
    /// Current generation
    pub current_generation: usize,
    /// Population of individuals
    pub population: Vec<Individual>,
}

impl EvolutionaryNAS {
    /// Create a new evolutionary NAS algorithm
    pub fn new(search_space: ArchitectureSpace) -> Self {
        Self {
            search_space,
            population_size: 50,
            num_elites: 5,
            mutation_prob: 0.1,
            crossover_prob: 0.8,
            max_generations: 100,
            current_generation: 0,
            population: Vec::new(),
        }
    }

    /// Initialize the population with random architectures
    pub fn initialize_population(&mut self) -> Result<()> {
        self.population.clear();

        for _ in 0..self.population_size {
            let architecture = self.search_space.sample_random(10)?;
            let individual = Individual {
                architecture,
                fitness: 0.0, // Will be evaluated later
                age: 0,
            };
            self.population.push(individual);
        }

        Ok(())
    }

    /// Run one generation of evolution
    pub fn evolve_generation<F>(&mut self, fitness_fn: F) -> Result<()>
    where
        F: Fn(&Architecture) -> Result<f64> + Send + Sync,
    {
        // Evaluate fitness for all individuals
        for individual in &mut self.population {
            if individual.fitness == 0.0 {
                // Only evaluate if not already evaluated
                individual.fitness = fitness_fn(&individual.architecture)?;
            }
        }

        // Sort population by fitness (best first)
        self.population.sort_by(|a, b| b.cmp(a));

        // Create new population
        let mut new_population = Vec::with_capacity(self.population_size);

        // Preserve elites
        for i in 0..self.num_elites.min(self.population.len()) {
            let mut elite = self.population[i].clone();
            elite.age += 1;
            new_population.push(elite);
        }

        // Generate offspring through selection, crossover, and mutation
        while new_population.len() < self.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection(3);
            let parent2 = self.tournament_selection(3);

            let mut offspring = if rand::random::<f64>() < self.crossover_prob {
                self.crossover(parent1, parent2)?
            } else {
                parent1.architecture.clone()
            };

            // Apply mutation
            if rand::random::<f64>() < self.mutation_prob {
                self.mutate(&mut offspring)?;
            }

            let child = Individual {
                architecture: offspring,
                fitness: 0.0, // Will be evaluated next generation
                age: 0,
            };

            new_population.push(child);
        }

        self.population = new_population;
        self.current_generation += 1;

        Ok(())
    }

    /// Run the full evolutionary search
    pub fn search<F>(&mut self, fitness_fn: F) -> Result<&Architecture>
    where
        F: Fn(&Architecture) -> Result<f64> + Send + Sync,
    {
        self.initialize_population()?;

        for generation in 0..self.max_generations {
            self.current_generation = generation;
            self.evolve_generation(&fitness_fn)?;

            // Log progress
            if let Some(best) = self.population.first() {
                println!(
                    "Generation {}: Best fitness = {:.4}, Params = {}",
                    generation,
                    best.fitness,
                    best.architecture.num_parameters()
                );
            }
        }

        // Return the best architecture
        self.population
            .first()
            .map(|individual| &individual.architecture)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: "No architectures in population".to_string(),
            })
    }

    /// Tournament selection: select the best individual from a random subset
    fn tournament_selection(&self, tournament_size: usize) -> &Individual {
        let mut best = None;
        let mut rng = rand::thread_rng();

        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            let candidate = &self.population[idx];

            best = match best {
                None => Some(candidate),
                Some(current_best) => {
                    if candidate.fitness > current_best.fitness {
                        Some(candidate)
                    } else {
                        Some(current_best)
                    }
                }
            };
        }

        best.unwrap()
    }

    /// Crossover two parent architectures
    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> Result<Architecture> {
        let mut child = Architecture::new(parent1.architecture.architecture_type);

        // Simple crossover: take layers from parent1, connections from parent2
        let split_point = rand::random::<usize>() % parent1.architecture.layers.len().max(1);

        // Take first part of layers from parent1
        for i in 0..split_point {
            if i < parent1.architecture.layers.len() {
                child.layers.push(parent1.architecture.layers[i].clone());
            }
        }

        // Take second part of layers from parent2
        for i in split_point..parent2.architecture.layers.len() {
            child.layers.push(parent2.architecture.layers[i].clone());
        }

        // Copy connections from parent2 (adjust indices if needed)
        for conn in &parent2.architecture.connections {
            if conn.from < child.layers.len() && conn.to < child.layers.len() {
                child.connections.push(*conn);
            }
        }

        // Copy parameters from parent1
        child.parameters = parent1.architecture.parameters.clone();

        // Ensure the child is valid
        if child.validate().is_ok() {
            Ok(child)
        } else {
            // If invalid, return parent1 as fallback
            Ok(parent1.architecture.clone())
        }
    }

    /// Mutate an architecture
    fn mutate(&self, architecture: &mut Architecture) -> Result<()> {
        let mut rng = rand::thread_rng();

        // Randomly choose mutation type
        let mutation_type = rng.gen_range(0..4);

        match mutation_type {
            0 => self.mutate_add_layer(architecture)?,
            1 => self.mutate_remove_layer(architecture)?,
            2 => self.mutate_modify_layer(architecture)?,
            3 => self.mutate_change_connections(architecture)?,
            _ => {}
        }

        // Validate after mutation
        architecture.validate().or_else(|_| {
            // If invalid after mutation, try to fix it
            self.repair_architecture(architecture)
        })
    }

    /// Add a random layer to the architecture
    fn mutate_add_layer(&self, architecture: &mut Architecture) -> Result<()> {
        if architecture.layers.len() >= self.search_space.max_layers {
            return Ok(());
        }

        let layer_type = &self.search_space.layer_types
            [rand::random::<usize>() % self.search_space.layer_types.len()];
        let new_layer = self.search_space.sample_layer(layer_type)?;

        let insert_pos = rand::random::<usize>() % (architecture.layers.len() + 1);
        architecture.layers.insert(insert_pos, new_layer);

        // Adjust connection indices
        for conn in &mut architecture.connections {
            if conn.from >= insert_pos {
                conn.from += 1;
            }
            if conn.to >= insert_pos {
                conn.to += 1;
            }
        }

        Ok(())
    }

    /// Remove a random layer from the architecture
    fn mutate_remove_layer(&self, architecture: &mut Architecture) -> Result<()> {
        if architecture.layers.len() <= 2 {
            // Keep at least input and output
            return Ok(());
        }

        let remove_pos = rand::random::<usize>() % (architecture.layers.len() - 2) + 1; // Don't remove input/output
        architecture.layers.remove(remove_pos);

        // Remove and adjust connections
        architecture
            .connections
            .retain(|conn| conn.from != remove_pos && conn.to != remove_pos);

        for conn in &mut architecture.connections {
            if conn.from > remove_pos {
                conn.from -= 1;
            }
            if conn.to > remove_pos {
                conn.to -= 1;
            }
        }

        Ok(())
    }

    /// Modify parameters of a random layer
    fn mutate_modify_layer(&self, architecture: &mut Architecture) -> Result<()> {
        if architecture.layers.is_empty() {
            return Ok(());
        }

        let layer_idx = rand::random::<usize>() % architecture.layers.len();
        let layer_type = &self.search_space.layer_types
            [rand::random::<usize>() % self.search_space.layer_types.len()];
        let new_layer = self.search_space.sample_layer(layer_type)?;

        architecture.layers[layer_idx] = new_layer;

        Ok(())
    }

    /// Change connection structure
    fn mutate_change_connections(&self, architecture: &mut Architecture) -> Result<()> {
        if architecture.layers.len() < 2 {
            return Ok(());
        }

        let mut rng = rand::thread_rng();

        // Randomly add or remove connections
        if rng.gen_bool(0.5) {
            // Add connection
            let from = rng.gen_range(0..architecture.layers.len());
            let to = rng.gen_range(0..architecture.layers.len());

            if from != to {
                let new_conn = super::search_space::Connection { from, to };
                architecture.connections.push(new_conn);
            }
        } else {
            // Remove connection
            if !architecture.connections.is_empty() {
                let remove_idx = rng.gen_range(0..architecture.connections.len());
                architecture.connections.remove(remove_idx);
            }
        }

        Ok(())
    }

    /// Try to repair an invalid architecture
    fn repair_architecture(&self, architecture: &mut Architecture) -> Result<()> {
        // Simple repair: remove invalid connections and ensure sequential connectivity
        architecture.connections.clear();

        // Add sequential connections
        for i in 0..architecture.layers.len().saturating_sub(1) {
            architecture
                .connections
                .push(super::search_space::Connection { from: i, to: i + 1 });
        }

        architecture.validate()
    }

    /// Get the best individual in the current population
    pub fn best_individual(&self) -> Option<&Individual> {
        self.population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
    }

    /// Get population statistics
    pub fn population_stats(&self) -> (f64, f64, f64, f64) {
        if self.population.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let fitness_values: Vec<f64> = self.population.iter().map(|ind| ind.fitness).collect();
        let mean = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let min = fitness_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = fitness_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let variance = fitness_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / fitness_values.len() as f64;
        let std = variance.sqrt();

        (mean, min, max, std)
    }
}

#[cfg(test)]
mod tests {
    use super::super::search_space::{ArchitectureType, LayerType, ParameterRange};
    use super::*;

    #[test]
    fn test_evolutionary_nas_initialization() {
        let mut search_space = ArchitectureSpace::new(ArchitectureType::CNN);
        search_space.add_layer_type(LayerType::Conv2D, ParameterRange::default());

        let mut nas = EvolutionaryNAS::new(search_space);
        nas.initialize_population().unwrap();

        assert_eq!(nas.population.len(), 50);
        for individual in &nas.population {
            assert!(individual.architecture.validate().is_ok());
        }
    }

    #[test]
    fn test_evolution_step() {
        let mut search_space = ArchitectureSpace::new(ArchitectureType::CNN);
        search_space.add_layer_type(LayerType::Conv2D, ParameterRange::default());

        let mut nas = EvolutionaryNAS::new(search_space);
        nas.initialize_population().unwrap();

        // Simple fitness function
        let fitness_fn = |_: &Architecture| Ok(rand::random::<f64>());

        nas.evolve_generation(fitness_fn).unwrap();

        assert_eq!(nas.population.len(), 50);
        assert_eq!(nas.current_generation, 1);
    }
}
