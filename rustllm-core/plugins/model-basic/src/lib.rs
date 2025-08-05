//! Basic model plugin implementation.

use rustllm_core::core::plugin::{Plugin, ModelBuilderPlugin, PluginCapabilities};
use rustllm_core::core::model::{Model, ForwardModel, ModelBuilder, ModelConfig, BasicModelConfig};
use rustllm_core::foundation::{
    error::Result,
    types::Version,
};

/// Basic model plugin.
#[derive(Debug, Default)]
pub struct BasicModelPlugin;

impl Plugin for BasicModelPlugin {
    fn name(&self) -> &str {
        "basic_model"
    }
    
    fn version(&self) -> Version {
        Version::new(0, 1, 0)
    }
    
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::standard()
            .with_feature("model_building")
    }
}

impl ModelBuilderPlugin for BasicModelPlugin {
    type Builder = BasicModelBuilder;
    
    fn create_builder(&self) -> Result<Self::Builder> {
        Ok(BasicModelBuilder::new())
    }
}

/// Basic model builder.
#[derive(Debug, Clone)]
pub struct BasicModelBuilder;

impl BasicModelBuilder {
    /// Creates a new basic model builder.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BasicModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder for BasicModelBuilder {
    type Model = BasicModel;
    type Config = BasicModelConfig;
    
    fn build(&self, config: Self::Config) -> Result<Self::Model> {
        config.validate()?;
        Ok(BasicModel::new(config))
    }
}

/// Basic model implementation.
#[derive(Debug)]
pub struct BasicModel {
    config: BasicModelConfig,
    num_parameters: usize,
}

impl BasicModel {
    /// Creates a new basic model.
    pub fn new(config: BasicModelConfig) -> Self {
        // Calculate parameters once during construction
        let embed_params = config.vocab_size * config.model_dim;
        let attn_params = config.layer_count * config.model_dim * config.model_dim * 4;
        let ffn_params = config.layer_count * config.model_dim * config.model_dim * 8;
        let num_parameters = embed_params + attn_params + ffn_params;
        
        Self { 
            config,
            num_parameters,
        }
    }
    
    /// Returns the number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.num_parameters
    }
}

impl Model for BasicModel {
    type Config = BasicModelConfig;
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn name(&self) -> &str {
        "BasicModel"
    }
}

impl ForwardModel for BasicModel {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Simple identity function for demonstration
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_model() {
        let config = BasicModelConfig::default();
        let builder = BasicModelBuilder::new();
        let model = builder.build(config).unwrap();
        
        let input = vec![1.0, 2.0, 3.0];
        let output = model.forward(input.clone()).unwrap();
        
        assert_eq!(output, input);
        assert!(model.num_parameters() > 0);
    }
}