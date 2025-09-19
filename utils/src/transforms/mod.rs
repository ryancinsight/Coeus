//! Data transformation utilities
//!
//! Provides common preprocessing operations for tensors,
//! compatible with PyTorch's torchvision.transforms interface.

// Core transforms
pub mod core;
pub use core::{
    Compose, Identity, Lambda, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor, Transform,
};

// Vision transforms
pub mod vision;
pub use vision::{
    ColorJitter, InterpolationMode, RandomAffine, RandomErasing, RandomPerspective, RandomRotation,
    RandomVerticalFlip,
};

// General transforms
pub mod general;
pub use general::{RandomApply, RandomChoice, RandomOrder};
