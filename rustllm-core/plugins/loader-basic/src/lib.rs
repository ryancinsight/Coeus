//! Basic loader plugin implementation.

use rustllm_core::core::plugin::{Plugin, ModelLoaderPlugin};
use rustllm_core::core::model::{Model, BasicModelConfig};
use rustllm_core::foundation::{
    error::{Error, Result},
    types::Version,
};

/// Basic model loader plugin.
#[derive(Debug, Default)]
pub struct BasicLoaderPlugin;

impl Plugin for BasicLoaderPlugin {
    fn name(&self) -> &str {
        "basic_loader"
    }
    
    fn version(&self) -> Version {
        Version::new(0, 1, 0)
    }
    
    fn description(&self) -> &str {
        "Basic model file loader"
    }
    
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl ModelLoaderPlugin for BasicLoaderPlugin {
    fn supported_formats(&self) -> Vec<&'static str> {
        vec!["txt", "json"]
    }
    
    fn load_model(&self, path: &str) -> Result<Box<dyn Model<
        Input = Vec<f32>,
        Output = Vec<f32>,
        Config = BasicModelConfig,
    >>> {
        // For demonstration, we just create a dummy model
        if !path.ends_with(".txt") && !path.ends_with(".json") {
            return Err(Error::NotSupported(format!("Unsupported file format: {}", path)));
        }
        
        // In a real implementation, we would read the file and deserialize the model
        let config = BasicModelConfig::default();
        let model = DummyModel::new(config);
        
        Ok(Box::new(model))
    }
}

/// Dummy model for demonstration.
#[derive(Debug)]
struct DummyModel {
    config: BasicModelConfig,
}

impl DummyModel {
    fn new(config: BasicModelConfig) -> Self {
        Self { config }
    }
}

impl Model for DummyModel {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type Config = BasicModelConfig;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Simple pass-through
        Ok(input)
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn num_parameters(&self) -> usize {
        1000 // Dummy value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_loader() {
        let loader = BasicLoaderPlugin::default();
        assert_eq!(loader.supported_formats(), vec!["txt", "json"]);
        
        // Test loading a supported format
        let model = loader.load_model("model.txt").unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let output = model.forward(input.clone()).unwrap();
        assert_eq!(output, input);
        
        // Test unsupported format
        assert!(loader.load_model("model.bin").is_err());
    }
}