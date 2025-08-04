//! Example demonstrating diffusion model capabilities.
//!
//! This example shows how to use the DiffusionModel trait for generative modeling
//! with efficient noise scheduling and sampling.

use rustllm_core::prelude::*;

/// Linear noise schedule implementation.
#[derive(Debug, Clone)]
struct LinearNoiseSchedule {
    num_steps: usize,
    beta_start: f32,
    beta_end: f32,
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alpha_bars: Vec<f32>,
}

impl LinearNoiseSchedule {
    fn new(num_steps: usize, beta_start: f32, beta_end: f32) -> Self {
        if num_steps == 0 {
            return Self {
                num_steps: 0,
                beta_start,
                beta_end,
                betas: Vec::new(),
                alphas: Vec::new(),
                alpha_bars: Vec::new(),
            };
        }
        
        let betas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let divisor = if num_steps > 1 { num_steps as f32 - 1.0 } else { 1.0 };
                beta_start + (beta_end - beta_start) * (i as f32) / divisor
            })
            .collect();
        
        let alphas: Vec<f32> = betas.iter().map(|&beta| 1.0 - beta).collect();
        
        // Use scan to compute cumulative product more idiomatically
        let alpha_bars: Vec<f32> = alphas.iter()
            .scan(1.0_f32, |acc, &alpha| {
                *acc *= alpha;
                Some(*acc)
            })
            .collect();
        
        Self {
            num_steps,
            beta_start,
            beta_end,
            betas,
            alphas,
            alpha_bars,
        }
    }
}

impl NoiseSchedule for LinearNoiseSchedule {
    fn num_steps(&self) -> usize {
        self.num_steps
    }
    
    fn beta(&self, timestep: usize) -> ModelFloat {
        self.betas.get(timestep).copied().unwrap_or(0.0)
    }
    
    fn alpha_bar(&self, timestep: usize) -> ModelFloat {
        self.alpha_bars.get(timestep).copied().unwrap_or(1.0)
    }
}

/// DDPM (Denoising Diffusion Probabilistic Models) sampler.
#[derive(Debug, Clone)]
struct DDPMSampler {
    deterministic: bool,
}

impl DDPMSampler {
    fn new(deterministic: bool) -> Self {
        Self { deterministic }
    }
}

impl DiffusionSampler for DDPMSampler {
    fn sample_step(
        &self,
        current: &[ModelFloat],
        predicted_noise: &[ModelFloat],
        timestep: usize,
        noise_schedule: &dyn NoiseSchedule,
    ) -> Result<Vec<ModelFloat>> {
        let beta = noise_schedule.beta(timestep);
        let alpha = 1.0 - beta;
        let alpha_bar = noise_schedule.alpha_bar(timestep);
        let alpha_bar_prev = if timestep > 0 {
            noise_schedule.alpha_bar(timestep - 1)
        } else {
            1.0
        };
        
        // Compute predicted x0
        let predicted_x0: Vec<f32> = current.iter()
            .zip(predicted_noise.iter())
            .map(|(&x_t, &noise)| {
                (x_t - noise * (1.0 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            })
            .collect();
        
        // Compute variance
        let variance = if self.deterministic || timestep == 0 {
            0.0
        } else {
            beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        };
        
        // Sample next step
        let next_sample: Vec<f32> = predicted_x0.iter()
            .zip(current.iter())
            .map(|(&x0_pred, &x_t)| {
                let mean = (alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar)) * x0_pred
                    + (alpha.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) * x_t;
                
                if variance > 0.0 {
                    // Add noise (simplified - in practice would use proper random sampling)
                    mean + variance.sqrt() * 0.1
                } else {
                    mean
                }
            })
            .collect();
        
        Ok(next_sample)
    }
    
    fn is_deterministic(&self) -> bool {
        self.deterministic
    }
}

/// Simple diffusion model for demonstration.
struct SimpleDiffusionModel {
    dim: usize,
}

impl SimpleDiffusionModel {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Model for SimpleDiffusionModel {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type Config = BasicModelConfig;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Simplified noise prediction - in practice would be a neural network
        Ok(input.iter().map(|&x| x * 0.9).collect())
    }
    
    fn config(&self) -> &Self::Config {
        static CONFIG: BasicModelConfig = BasicModelConfig {
            model_dim: 128,
            head_count: 8,
            layer_count: 6,
            max_seq_len: 512,
            vocab_size: 1000,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        };
        &CONFIG
    }
    
    fn num_parameters(&self) -> usize {
        self.dim * self.dim // Simplified
    }
}

impl DiffusionModel for SimpleDiffusionModel {
    type NoiseSchedule = LinearNoiseSchedule;
    type Sampler = DDPMSampler;
    
    fn add_noise(
        &self,
        input: &Self::Input,
        timestep: usize,
        noise_schedule: &Self::NoiseSchedule,
    ) -> Result<Self::Input> {
        let alpha_bar = noise_schedule.alpha_bar(timestep);
        
        // Add noise according to schedule
        let noisy_input: Vec<f32> = input.iter()
            .map(|&x| {
                let noise = ((timestep + 1) as f32 * 0.1).sin(); // Simplified noise
                x * alpha_bar.sqrt() + noise * (1.0 - alpha_bar).sqrt()
            })
            .collect();
        
        Ok(noisy_input)
    }
    
    fn predict_noise(
        &self,
        noisy_input: Self::Input,
        timestep: usize,
    ) -> Result<Self::Output> {
        // In practice, this would condition on timestep
        let scale = 1.0 - (timestep as f32 / 1000.0);
        Ok(noisy_input.iter().map(|&x| x * scale * 0.1).collect())
    }
    
    fn denoise_step(
        &self,
        noisy_input: Self::Input,
        timestep: usize,
        predicted_noise: Self::Output,
        noise_schedule: &Self::NoiseSchedule,
    ) -> Result<Self::Input> {
        let alpha_bar = noise_schedule.alpha_bar(timestep);
        
        // Remove predicted noise
        let denoised: Vec<f32> = noisy_input.iter()
            .zip(predicted_noise.iter())
            .map(|(&noisy, &noise)| {
                (noisy - noise * (1.0 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            })
            .collect();
        
        Ok(denoised)
    }
    
    fn generate_samples(
        &self,
        initial_noise: Self::Input,
        sampler: &Self::Sampler,
        noise_schedule: &Self::NoiseSchedule,
    ) -> Result<Self::Output> {
        let mut current = initial_noise;
        
        // Reverse diffusion process using iterator
        for timestep in (0..noise_schedule.num_steps()).rev() {
            let predicted_noise = self.predict_noise(current.clone(), timestep)?;
            current = sampler.sample_step(&current, &predicted_noise, timestep, noise_schedule)?;
        }
        
        Ok(current)
    }
}

fn main() -> Result<()> {
    println!("RustLLM Core Diffusion Model Example");
    println!("====================================\n");
    
    // Create diffusion model and noise schedule
    let model = SimpleDiffusionModel::new(10);
    let noise_schedule = LinearNoiseSchedule::new(1000, 0.0001, 0.02);
    let sampler = DDPMSampler::new(false);
    
    println!("Model initialized with {} parameters", model.num_parameters());
    println!("Noise schedule: {} steps, beta range [{:.4}, {:.4}]", 
             noise_schedule.num_steps(), 
             noise_schedule.beta_start, 
             noise_schedule.beta_end);
    
    // Demonstrate forward diffusion process
    println!("\nForward diffusion process:");
    let original_data = vec![1.0; 10];
    println!("Original data: {:?}", original_data);
    
    for t in [0, 250, 500, 750, 999] {
        let noisy_data = model.add_noise(&original_data, t, &noise_schedule)?;
        let noise_level = noise_schedule.alpha_bar(t);
        println!("t={:3}: noise_level={:.4}, data_std={:.4}", 
                 t, 
                 1.0 - noise_level,
                 noisy_data.iter().map(|&x| x * x).sum::<f32>().sqrt() / noisy_data.len() as f32);
    }
    
    // Demonstrate reverse diffusion (generation)
    println!("\nReverse diffusion (generation):");
    let initial_noise: Vec<f32> = (0..10).map(|i| ((i as f32 + 1.0) * 0.3).sin()).collect();
    println!("Initial noise: {:?}", initial_noise);
    
    let generated = model.generate_samples(initial_noise, &sampler, &noise_schedule)?;
    println!("Generated samples: {:?}", generated);
    
    // Demonstrate efficient batch processing with iterators
    println!("\nBatch diffusion with iterator combinators:");
    let batch_results: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            let noise: Vec<f32> = (0..10)
                .map(|j| ((i * 10 + j) as f32 * 0.1).cos())
                .collect();
            noise
        })
        .stream_map(|noise| {
            model.generate_samples(noise, &sampler, &noise_schedule).unwrap_or_default()
        })
        .collect();
    
    for (i, result) in batch_results.iter().enumerate() {
        println!("Batch {}: mean={:.4}, std={:.4}", 
                 i,
                 result.iter().sum::<f32>() / result.len() as f32,
                 (result.iter().map(|&x| x * x).sum::<f32>() / result.len() as f32).sqrt());
    }
    
    // Demonstrate memory-efficient processing
    println!("\nMemory-efficient diffusion with SliceView:");
    let data = vec![0.5; 100];
    let view = SliceView::new(&data)
        .map(|x| x * 2.0);
    
    println!("Original data[0]: {}, Transformed view[0]: {:?}", 
             data[0], view.get(0));
    
    Ok(())
}