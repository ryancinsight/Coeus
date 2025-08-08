//! Modeling domain - Neural network architecture and model building.
//!
//! This domain handles model architecture definition and construction.

use crate::core::traits::Identity;
use crate::foundation::error::Result;

/// Model architecture types.
#[derive(Debug, Clone)]
pub enum Architecture {
    Transformer { layers: usize, heads: usize },
    RNN { layers: usize, hidden_size: usize },
    CNN { layers: Vec<usize> },
}

/// Model aggregate.
#[derive(Debug)]
pub struct ModelAggregate {
    id: String,
    architecture: Architecture,
}

impl Identity for ModelAggregate {
    fn id(&self) -> &str {
        &self.id
    }
}

/// Modeling service.
pub struct ModelingService;

/// Modeling events.
#[derive(Debug, Clone)]
pub enum ModelingEvent {
    Created { model_id: String },
    Updated { model_id: String },
}

/// Model repository trait.
pub trait ModelRepository: Send + Sync {
    fn save(&self, model: &ModelAggregate) -> Result<()>;
    fn load(&self, id: &str) -> Result<ModelAggregate>;
}
