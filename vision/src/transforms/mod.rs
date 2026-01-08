mod color_jitter;
mod crop;
mod flip;
mod normalize;
mod random_resized_crop;
mod resize;
mod totensor;

use crate::ImageTensor;

// Re-export specific transforms
pub use color_jitter::ColorJitter;
pub use crop::{CenterCrop, RandomCrop};
pub use flip::RandomHorizontalFlip;
pub use normalize::Normalize;
pub use random_resized_crop::RandomResizedCrop;
pub use resize::Resize;
pub use totensor::ToTensor;

// Unify Transform trait from utils
pub use utils::{Transform, TransformError};

pub struct Compose {
    transforms: Vec<Box<dyn Transform<ImageTensor, ImageTensor>>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform<ImageTensor, ImageTensor>>>) -> Self {
        Self { transforms }
    }

    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    pub fn apply(&self, input: ImageTensor) -> Result<ImageTensor, TransformError> {
        let mut iter = self.transforms.iter();
        let Some(first) = iter.next() else {
            return Err(TransformError::TransformError {
                message: "cannot apply empty transform pipeline".to_string(),
            });
        };

        // Note: Transform trait in utils might behave differently.
        // utils::Transform::apply takes self and input.
        // Let's check utils::Transform signature.
        let mut current = first.apply(input)?;
        for t in iter {
            current = t.apply(current)?;
        }
        Ok(current)
    }
}
