//! Task Distribution Learning.
//!
//! This module implements task distribution learning for meta-learning,
//! including task sampling strategies and distribution adaptation.

use crate::core::error::{NNError, Result};
use rand::Rng;

/// Task distribution for meta-training
pub struct TaskDistribution {
    /// Task sampler function
    pub sampler: Box<dyn Fn() -> Result<Task>>,
    /// Distribution parameters
    pub parameters: Vec<f64>,
}

impl TaskDistribution {
    /// Create a new task distribution
    pub fn new<F>(sampler: F) -> Self
    where
        F: Fn() -> Result<Task> + 'static,
    {
        Self {
            sampler: Box::new(sampler),
            parameters: Vec::new(),
        }
    }

    /// Sample a task from the distribution
    pub fn sample_task(&self) -> Result<Task> {
        (self.sampler)()
    }

    /// Sample multiple tasks
    pub fn sample_batch(&self, batch_size: usize) -> Result<Vec<Task>> {
        let mut tasks = Vec::new();
        for _ in 0..batch_size {
            tasks.push(self.sample_task()?);
        }
        Ok(tasks)
    }
}

/// Task definition for meta-learning
#[derive(Debug, Clone)]
pub struct Task {
    /// Task identifier
    pub id: String,
    /// Task difficulty
    pub difficulty: f64,
    /// Task domain
    pub domain: String,
    /// Task-specific parameters
    pub parameters: Vec<f64>,
}

impl Task {
    /// Create a new task
    pub fn new(id: String, difficulty: f64, domain: String) -> Self {
        Self {
            id,
            difficulty,
            domain,
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to the task
    pub fn with_parameter(mut self, param: f64) -> Self {
        self.parameters.push(param);
        self
    }
}

/// Uniform task distribution
pub struct UniformTaskDistribution {
    /// Available tasks
    pub tasks: Vec<Task>,
}

impl UniformTaskDistribution {
    /// Create a new uniform distribution
    pub fn new(tasks: Vec<Task>) -> Self {
        Self { tasks }
    }

    /// Sample a task uniformly at random
    pub fn sample(&self) -> Result<Task> {
        if self.tasks.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "No tasks available in distribution".to_string(),
            });
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.tasks.len());
        Ok(self.tasks[idx].clone())
    }
}

/// Difficulty-weighted task distribution
pub struct DifficultyWeightedDistribution {
    /// Available tasks with weights
    pub tasks: Vec<(Task, f64)>,
}

impl DifficultyWeightedDistribution {
    /// Create a new difficulty-weighted distribution
    pub fn new(tasks: Vec<Task>) -> Self {
        // Weight by inverse difficulty (easier tasks more likely)
        let weighted_tasks = tasks
            .into_iter()
            .map(|task| {
                let weight = 1.0 / (task.difficulty + 1.0); // Avoid division by zero
                (task, weight)
            })
            .collect();

        Self {
            tasks: weighted_tasks,
        }
    }

    /// Sample a task weighted by difficulty
    pub fn sample(&self) -> Result<Task> {
        if self.tasks.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "No tasks available in distribution".to_string(),
            });
        }

        let total_weight: f64 = self.tasks.iter().map(|(_, w)| w).sum();
        let mut rng = rand::thread_rng();
        let target = rng.gen::<f64>() * total_weight;

        let mut cumulative = 0.0;
        for (task, weight) in &self.tasks {
            cumulative += weight;
            if target <= cumulative {
                return Ok(task.clone());
            }
        }

        // Fallback (should not happen)
        Ok(self.tasks[0].0.clone())
    }
}

/// Curriculum learning distribution that increases difficulty over time
pub struct CurriculumDistribution {
    /// Tasks organized by difficulty level
    pub curriculum_stages: Vec<Vec<Task>>,
    /// Current stage
    pub current_stage: usize,
    /// Tasks completed in current stage
    pub stage_progress: usize,
    /// Tasks needed to advance to next stage
    pub advancement_threshold: usize,
}

impl CurriculumDistribution {
    /// Create a new curriculum distribution
    pub fn new(curriculum_stages: Vec<Vec<Task>>, advancement_threshold: usize) -> Self {
        Self {
            curriculum_stages,
            current_stage: 0,
            stage_progress: 0,
            advancement_threshold,
        }
    }

    /// Sample a task from current curriculum stage
    pub fn sample(&self) -> Result<Task> {
        if self.current_stage >= self.curriculum_stages.len() {
            return Err(NNError::InvalidConfiguration {
                message: "Curriculum completed".to_string(),
            });
        }

        let current_tasks = &self.curriculum_stages[self.current_stage];
        if current_tasks.is_empty() {
            return Err(NNError::InvalidConfiguration {
                message: "No tasks in current curriculum stage".to_string(),
            });
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..current_tasks.len());
        Ok(current_tasks[idx].clone())
    }

    /// Report task completion and potentially advance curriculum
    pub fn complete_task(&mut self) -> bool {
        self.stage_progress += 1;

        if self.stage_progress >= self.advancement_threshold {
            self.current_stage += 1;
            self.stage_progress = 0;
            true // Advanced to next stage
        } else {
            false // Stay in current stage
        }
    }

    /// Get current curriculum stage
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Check if curriculum is completed
    pub fn is_completed(&self) -> bool {
        self.current_stage >= self.curriculum_stages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let tasks = vec![
            Task::new("task1".to_string(), 1.0, "domain1".to_string()),
            Task::new("task2".to_string(), 2.0, "domain1".to_string()),
        ];

        let dist = UniformTaskDistribution::new(tasks);

        for _ in 0..10 {
            let task = dist.sample().unwrap();
            assert!(task.id == "task1" || task.id == "task2");
        }
    }

    #[test]
    fn test_difficulty_weighted_distribution() {
        let tasks = vec![
            Task::new("easy".to_string(), 1.0, "domain1".to_string()),
            Task::new("hard".to_string(), 3.0, "domain1".to_string()),
        ];

        let dist = DifficultyWeightedDistribution::new(tasks);

        // Easy task should be sampled more frequently
        let mut easy_count = 0;
        let mut hard_count = 0;

        for _ in 0..1000 {
            let task = dist.sample().unwrap();
            if task.id == "easy" {
                easy_count += 1;
            } else {
                hard_count += 1;
            }
        }

        assert!(easy_count > hard_count); // Easy task should be more frequent
    }

    #[test]
    fn test_curriculum_distribution() {
        let stage1_tasks = vec![Task::new(
            "stage1_task".to_string(),
            1.0,
            "domain1".to_string(),
        )];
        let stage2_tasks = vec![Task::new(
            "stage2_task".to_string(),
            2.0,
            "domain1".to_string(),
        )];

        let mut dist = CurriculumDistribution::new(vec![stage1_tasks, stage2_tasks], 5);

        // Sample from stage 1
        let task1 = dist.sample().unwrap();
        assert_eq!(task1.id, "stage1_task");
        assert_eq!(dist.current_stage(), 0);

        // Complete tasks to advance
        for _ in 0..5 {
            let advanced = dist.complete_task();
            if dist.stage_progress == 0 {
                assert!(advanced);
                break;
            }
        }

        // Should now be in stage 2
        let task2 = dist.sample().unwrap();
        assert_eq!(task2.id, "stage2_task");
        assert_eq!(dist.current_stage(), 1);
    }
}
