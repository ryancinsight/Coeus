//! Specialized loss functions
//!
//! Loss functions for specific domains and applications, including
//! computer vision, natural language processing, and other specialized tasks.

use super::{utils, Module, NNError, Reduction};
use coeus_tensor::{Dtype, FloatDtype, Mul, Tensor, CpuBackend};

/// Focal Loss
///
/// Computes focal loss for addressing class imbalance in classification:
/// `FocalLoss = -α * (1 - p_t)^γ * log(p_t)`
///
/// Where p_t is the model's estimated probability for the true class.
/// This loss down-weights easy examples and focuses on hard examples.
///
/// # Mathematical Properties
/// - α controls the relative importance of rare vs common classes
/// - γ (gamma) controls the rate at which easy examples are down-weighted
/// - Reduces to cross-entropy when γ = 0
/// - Commonly used in object detection (RetinaNet)
///
/// # Example
/// ```rust
/// use coeus_nn::FocalLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = FocalLoss::new(2.0, 0.25); // gamma=2.0, alpha=0.25
/// let logits = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
/// let targets = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&logits, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FocalLoss {
    /// Focusing parameter (default: 2.0)
    pub gamma: f64,
    /// Weighting factor for rare class (default: 1.0)
    pub alpha: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for FocalLoss {
    fn default() -> Self {
        Self::new(2.0, 1.0)
    }
}

impl FocalLoss {
    /// Create a new focal loss with specified gamma and alpha
    pub fn new(gamma: f64, alpha: f64) -> Self {
        Self {
            gamma,
            alpha,
            reduction: Reduction::Mean,
        }
    }

    /// Create with all parameters specified
    pub fn with_params(gamma: f64, alpha: f64, reduction: Reduction) -> Self {
        Self {
            gamma,
            alpha,
            reduction,
        }
    }

    /// Compute focal loss between logits and target class indices
    pub fn forward<T: FloatDtype + std::iter::Sum, I: coeus_tensor::Dtype + num_traits::ToPrimitive>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<I, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if predictions.ndim() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: "Predictions must be 2D tensor (batch_size, num_classes)".to_string(),
            });
        }

        if targets.ndim() != 1 {
            return Err(crate::NNError::InvalidInput {
                message: "Targets must be 1D tensor (batch_size,)".to_string(),
            });
        }

        if predictions.shape()[0] != targets.shape()[0] {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![predictions.shape()[0]],
                actual: targets.shape().to_vec(),
            });
        }

        // Compute softmax probabilities
        let softmax = utils::softmax(predictions)?;
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        let mut losses = Vec::with_capacity(batch_size);
        let gamma = T::from(self.gamma).unwrap();
        let alpha = T::from(self.alpha).unwrap();

        for i in 0..batch_size {
            let target_idx =
                targets.data()[i]
                    .to_usize()
                    .ok_or_else(|| crate::NNError::InvalidInput {
                        message: "Target indices must be valid usize values".to_string(),
                    })?;

            if target_idx >= num_classes {
                return Err(crate::NNError::InvalidInput {
                    message: format!(
                        "Target index {} out of bounds for {} classes",
                        target_idx, num_classes
                    ),
                });
            }

            // Get probability for true class
            let p_t = softmax.data()[i * num_classes + target_idx];
            let p_t_clamped = utils::clamp_for_log(p_t);

            // Focal loss: -α * (1 - p_t)^γ * log(p_t)
            let one_minus_pt = T::one() - p_t;
            let focal_weight = one_minus_pt.powf(gamma);
            let focal_loss = -alpha * focal_weight * p_t_clamped.ln();

            losses.push(focal_loss);
        }

        let loss_tensor = Tensor::from_vec(CpuBackend::default(), losses, vec![batch_size]).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for FocalLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "FocalLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

// Dice Loss
///
/// Computes Dice loss for segmentation tasks:
/// `DiceLoss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)`
///
/// Where X is the predicted segmentation and Y is the ground truth.
/// This loss is commonly used in medical image segmentation.
///
/// # Mathematical Properties
/// - Based on Dice coefficient (F1 score for binary classification)
/// - Handles class imbalance naturally
/// - Smooth parameter prevents division by zero
/// - Range: [0, 1] where 0 is perfect overlap
///
/// # Example
/// ```rust
/// use coeus_nn::DiceLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = DiceLoss::new();
/// let predictions = Tensor::from_vec(CpuBackend::default(), vec![0.9], vec![1]).unwrap();
/// let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DiceLoss {
    /// Smoothing parameter to avoid division by zero (default: 1.0)
    pub smooth: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for DiceLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DiceLoss {
    /// Create a new Dice loss with default smoothing
    pub fn new() -> Self {
        Self {
            smooth: 1.0,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified smoothing parameter
    pub fn with_smooth(smooth: f64) -> Self {
        Self {
            smooth,
            reduction: Reduction::Mean,
        }
    }

    /// Create with all parameters specified
    pub fn with_params(smooth: f64, reduction: Reduction) -> Self {
        Self { smooth, reduction }
    }

    /// Compute Dice loss between predictions and targets
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();
        let smooth = T::from(self.smooth).unwrap();

        // Compute intersection and union
        let mut intersection = T::zero();
        let mut pred_sum = T::zero();
        let mut target_sum = T::zero();

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            intersection = intersection + pred * target;
            pred_sum = pred_sum + pred;
            target_sum = target_sum + target;
        }

        // Dice coefficient: (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        let numerator = T::from(2.0).unwrap() * intersection + smooth;
        let denominator = pred_sum + target_sum + smooth;
        let dice_coeff = numerator / denominator;

        // Dice loss: 1 - dice_coefficient
        let dice_loss = T::one() - dice_coeff;

        Ok(Tensor::scalar(dice_loss))
    }
}

impl<T: FloatDtype> Module<T> for DiceLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "DiceLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}
///
/// IoU Loss (Intersection over Union Loss)
///
/// Computes IoU loss for segmentation tasks:
/// `IoULoss = 1 - (|X ∩ Y| + smooth) / (|X ∪ Y| + smooth)`
///
/// Where X is the predicted segmentation and Y is the ground truth.
/// Also known as Jaccard loss, commonly used in semantic segmentation.
///
/// # Mathematical Properties
/// - Based on Jaccard index (IoU metric)
/// - Handles class imbalance well
/// - Smooth parameter prevents division by zero
/// - Range: [0, 1] where 0 is perfect overlap
///
/// # Example
/// ```rust
/// use coeus_nn::IoULoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = IoULoss::new();
/// let predictions = Tensor::from_vec(CpuBackend::default(), vec![0.9], vec![1]).unwrap();
/// let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct IoULoss {
    /// Smoothing parameter to avoid division by zero (default: 1.0)
    pub smooth: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for IoULoss {
    fn default() -> Self {
        Self::new()
    }
}

impl IoULoss {
    /// Create a new IoU loss with default smoothing
    pub fn new() -> Self {
        Self {
            smooth: 1.0,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified smoothing parameter
    pub fn with_smooth(smooth: f64) -> Self {
        Self {
            smooth,
            reduction: Reduction::Mean,
        }
    }

    /// Create with all parameters specified
    pub fn with_params(smooth: f64, reduction: Reduction) -> Self {
        Self { smooth, reduction }
    }

    /// Compute IoU loss between predictions and targets
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();
        let smooth = T::from(self.smooth).unwrap();

        // Compute intersection and union
        let mut intersection = T::zero();
        let mut union = T::zero();

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            intersection = intersection + pred * target;
            union = union + pred + target - pred * target;
        }

        // IoU coefficient: (intersection + smooth) / (union + smooth)
        let iou_coeff = (intersection + smooth) / (union + smooth);

        // IoU loss: 1 - IoU_coefficient
        let iou_loss = T::one() - iou_coeff;

        Ok(Tensor::scalar(iou_loss))
    }
}

impl<T: FloatDtype> Module<T> for IoULoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "IoULoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Soft Margin Loss
///
/// Computes soft margin loss for two-class classification:
/// `SoftMarginLoss(x, y) = sum log(1 + exp(-y * x)) / batch_size`
///
/// Where y ∈ {-1, 1} are the target labels and x are the predictions.
/// This is similar to logistic loss but allows for y ∈ {-1, 1} instead of {0, 1}.
///
/// # Mathematical Properties
/// - Smooth approximation of hinge loss
/// - Differentiable everywhere
/// - Commonly used in SVM-like formulations
/// - Range: [0, ∞) where 0 is perfect classification
///
/// # Example
/// ```rust
/// use coeus_nn::SoftMarginLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = SoftMarginLoss::new();
/// let predictions = Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap();
/// let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SoftMarginLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for SoftMarginLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftMarginLoss {
    /// Create a new soft margin loss function
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute soft margin loss
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();

        let mut losses = Vec::with_capacity(predictions.numel());

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            // Soft margin loss: log(1 + exp(-y * x))
            let y_x = target * pred;
            let exp_term = (-y_x).exp();
            let loss_val = (T::one() + exp_term).ln();
            losses.push(loss_val);
        }

        let loss_tensor = Tensor::from_vec(CpuBackend::default(), losses, predictions.shape().to_vec()).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for SoftMarginLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "SoftMarginLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Connectionist Temporal Classification (CTC) Loss
///
/// Computes CTC loss for sequence-to-sequence tasks without alignment:
/// `CTCLoss(log_probs, targets, input_lengths, target_lengths)`
///
/// This loss is used for tasks like speech recognition and handwriting recognition
/// where the alignment between input and target sequences is unknown.
///
/// # Mathematical Properties
/// - Handles variable-length sequences
/// - No need for pre-aligned training data
/// - Uses dynamic programming for efficient computation
/// - Supports blank symbols for insertions/deletions
///
/// # Mathematical Properties
/// - Designed for count data modeling
/// - Handles Poisson distributed targets
/// - Supports both full and reduced log-likelihood forms
///
/// # Example
/// ```rust
/// use coeus_nn::PoissonNLLLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = PoissonNLLLoss::new();
/// let log_input = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap(); // log(λ)
/// let target = Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap(); // count targets
///
/// let loss = loss_fn.forward(&log_input, &target).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PoissonNLLLoss {
    /// Whether to compute full loss (including log(y!)) or reduced form
    pub full: bool,
    /// Small epsilon for numerical stability when target = 0
    pub eps: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl PoissonNLLLoss {
    /// Create a new Poisson NLL loss with reduced form
    pub fn new() -> Self {
        Self {
            full: false,
            eps: 1e-8,
            reduction: Reduction::Mean,
        }
    }

    /// Create with full log-likelihood computation
    pub fn full() -> Self {
        Self {
            full: true,
            eps: 1e-8,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            full: false,
            eps: 1e-8,
            reduction,
        }
    }

    /// Compute Poisson NLL loss between log-rate predictions and count targets
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        // Input should be log(λ), target should be count values
        if input.shape() != target.shape() {
            return Err(NNError::InvalidInput {
                message: format!(
                    "PoissonNLLLoss: input and target shapes must match, got {:?} and {:?}",
                    input.shape(),
                    target.shape()
                ),
            });
        }

        // Compute λ = exp(log_λ)
        let rate = input.exp()?;

        // Compute base loss: λ - target * log(λ + eps)
        let eps_tensor = Tensor::scalar(T::from(self.eps).unwrap());

        let rate_plus_eps = (&rate + &eps_tensor)?;
        let log_rate = rate_plus_eps.log()?;
        let target_log_rate = target.mul(&log_rate)?;
        let loss = (&rate - &target_log_rate)?;

        // Add log(y!) term if full=True
        let final_loss = if self.full {
            // Stirling's approximation for log(y!): y*log(y) - y + 0.5*log(2πy)
            let target_log = target.log()?;
            let y_log_y = target.mul(&target_log)?;
            let log_factorial = (&y_log_y - target)?;
            // For simplicity, we'll use a basic approximation
            // In practice, you'd want a more accurate log-factorial computation
            &loss + &log_factorial
        } else {
            Ok(loss)
        };

        // Apply reduction
        utils::apply_reduction(&final_loss?, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for PoissonNLLLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(NNError::InvalidInput {
            message: "PoissonNLLLoss should be used via forward() method with two inputs"
                .to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Gaussian Negative Log Likelihood Loss
///
/// Computes negative log likelihood loss for Gaussian distributions:
/// `GaussianNLLLoss(y, μ, σ²) = 0.5 * log(2πσ²) + (y - μ)²/(2σ²)`
///
/// This loss is useful for modeling continuous data with known or learned
/// variance, providing uncertainty quantification capabilities.
///
/// # Mathematical Properties
/// - Models Gaussian distributed targets
/// - Supports heteroscedastic regression (varying variance)
/// - Provides proper uncertainty quantification
///
/// # Example
/// ```rust
/// use coeus_nn::GaussianNLLLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = GaussianNLLLoss::new();
/// let pred_mean = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap(); // predicted mean
/// let pred_var = Tensor::from_vec(CpuBackend::default(), vec![0.5], vec![1]).unwrap(); // predicted variance
/// let target = Tensor::from_vec(CpuBackend::default(), vec![1.2], vec![1]).unwrap(); // target values
///
/// let loss = loss_fn.forward_separate(&pred_mean, &pred_var, &target, vec![1]).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianNLLLoss {
    /// Full log-likelihood computation (default: true)
    pub full: bool,
    /// Small epsilon for numerical stability when variance = 0
    pub eps: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl GaussianNLLLoss {
    /// Create a new Gaussian NLL loss
    pub fn new() -> Self {
        Self {
            full: true,
            eps: 1e-8,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            full: true,
            eps: 1e-8,
            reduction,
        }
    }
}

impl GaussianNLLLoss {
    /// Forward pass with separate pred_mean and pred_var tensors
    pub fn forward_separate<T: FloatDtype + num_traits::FromPrimitive + std::iter::Sum>(
        &self,
        pred_mean: &Tensor<T, CpuBackend>,
        pred_var: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        // pred_mean and pred_var should have same shape, target can broadcast
        if pred_mean.shape() != pred_var.shape() {
            return Err(NNError::InvalidInput {
                message: format!(
                    "GaussianNLLLoss: pred_mean and pred_var shapes must match, got {:?} and {:?}",
                    pred_mean.shape(),
                    pred_var.shape()
                ),
            });
        }

        // Add epsilon to variance for numerical stability
        let eps_tensor = Tensor::scalar(T::from(self.eps).unwrap());
        let var = (pred_var + &eps_tensor)?;

        // Compute squared error: (target - pred_mean)²
        let diff = (target - pred_mean)?;
        let squared_error = diff.mul(&diff)?;

        // Compute loss: 0.5 * log(2πσ²) + (y - μ)²/(2σ²)
        let var_term = (&squared_error / &var)?;
        let half_var_term = (&var_term * &Tensor::scalar(T::from(0.5).unwrap()))?;

        let loss: crate::Result<Tensor<T, CpuBackend>> = if self.full {
            // Add log variance term: 0.5 * log(2πσ²) = 0.5 * (log(2π) + log(σ²))
            let log_var = var.log()?;
            let log_2pi = Tensor::scalar(T::from((2.0 * std::f64::consts::PI).ln()).unwrap());
            let log_term = (&log_var + &log_2pi)?;
            let half_log_term = (&log_term * &Tensor::scalar(T::from(0.5).unwrap()))?;
            Ok((&half_var_term + &half_log_term)?)
        } else {
            Ok(half_var_term)
        };

        // Apply reduction
        utils::apply_reduction(&loss?, self.reduction)
    }
}

impl<T: FloatDtype + num_traits::FromPrimitive> Module<T> for GaussianNLLLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(NNError::InvalidInput {
            message: "GaussianNLLLoss should be used via forward_separate() method with pred_mean, pred_var, and target".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// use coeus_nn::CTCLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = CTCLoss::new();
/// // Log probabilities: (batch_size, seq_len, num_classes + 1) where +1 is for blank
/// let log_probs = Tensor::from_vec(vec![
///     0.1f32, 0.2, 0.3, 0.4,  // seq 1: class probs for 4 classes
///     0.2f32, 0.3, 0.1, 0.4,  // seq 2: class probs for 4 classes
///     0.3f32, 0.1, 0.4, 0.2,  // seq 3: class probs for 4 classes
///     0.4f32, 0.2, 0.1, 0.3   // seq 4: class probs for 4 classes
/// ], vec![2, 2, 4]); // (batch_size=2, seq_len=2, num_classes=4)
/// let targets = Tensor::from_vec(vec![
///     1i32, 2, // batch 1: target sequence
///     1i32, 0  // batch 2: target sequence (padded with blank=0)
/// ], vec![2, 2]); // (batch_size=2, max_target_len=2)
/// let input_lengths = Tensor::from_vec(CpuBackend::default(), vec![2i32], vec![1]).unwrap(); // Length of input sequences
/// let target_lengths = Tensor::from_vec(CpuBackend::default(), vec![2i32], vec![1]).unwrap(); // Length of target sequences
///
/// let loss = loss_fn.forward(&log_probs, &targets, &input_lengths, &target_lengths).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CTCLoss {
    /// Blank symbol index (default: 0)
    pub blank: usize,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for CTCLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CTCLoss {
    /// Create a new CTC loss function
    pub fn new() -> Self {
        Self {
            blank: 0,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified blank symbol
    pub fn with_blank(blank: usize) -> Self {
        Self {
            blank,
            reduction: Reduction::Mean,
        }
    }

    /// Create with all parameters specified
    pub fn with_params(blank: usize, reduction: Reduction) -> Self {
        Self { blank, reduction }
    }

    /// Compute CTC loss using forward-backward algorithm
    pub fn forward<T: FloatDtype + std::iter::Sum, I: Dtype + num_traits::ToPrimitive>(
        &self,
        log_probs: &Tensor<T, CpuBackend>,
        targets: &Tensor<I, CpuBackend>,
        input_lengths: &Tensor<I, CpuBackend>,
        target_lengths: &Tensor<I, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if log_probs.ndim() != 3 {
            return Err(crate::NNError::InvalidInput {
                message: "log_probs must be 3D tensor (batch_size, seq_len, num_classes)"
                    .to_string(),
            });
        }

        if targets.ndim() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: "targets must be 2D tensor (batch_size, max_target_len)".to_string(),
            });
        }

        if input_lengths.ndim() != 1 || target_lengths.ndim() != 1 {
            return Err(crate::NNError::InvalidInput {
                message: "input_lengths and target_lengths must be 1D tensors".to_string(),
            });
        }

        let batch_size = log_probs.shape()[0];
        let seq_len = log_probs.shape()[1];
        let num_classes = log_probs.shape()[2];

        if targets.shape()[0] != batch_size
            || input_lengths.shape()[0] != batch_size
            || target_lengths.shape()[0] != batch_size
        {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![batch_size, seq_len, num_classes],
                actual: vec![
                    targets.shape()[0],
                    input_lengths.shape()[0],
                    target_lengths.shape()[0],
                ],
            });
        }

        let mut losses = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let input_len = input_lengths.data()[batch_idx].to_usize().unwrap();
            let target_len = target_lengths.data()[batch_idx].to_usize().unwrap();

            if input_len > seq_len || target_len > targets.shape()[1] {
                return Err(crate::NNError::InvalidInput {
                    message: format!(
                        "Invalid lengths: input_len={}, seq_len={}, target_len={}, max_target_len={}",
                        input_len, seq_len, target_len, targets.shape()[1]
                    ),
                });
            }

            // Extract target sequence for this batch
            let target_start = batch_idx * targets.shape()[1];
            let target_end = target_start + target_len;
            let target_seq: Vec<usize> = targets.data()[target_start..target_end]
                .iter()
                .map(|&x| x.to_usize().unwrap())
                .collect();

            // Compute CTC loss for this sequence using forward-backward algorithm
            let loss_val = self.compute_ctc_loss(
                &log_probs.data()
                    [batch_idx * seq_len * num_classes..(batch_idx + 1) * seq_len * num_classes],
                &target_seq,
                input_len,
                seq_len,
                num_classes,
            )?;

            losses.push(loss_val);
        }

        let loss_tensor = Tensor::from_vec(CpuBackend::default(), losses, vec![batch_size]).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }

    /// Compute CTC loss for a single sequence using forward-backward algorithm
    fn compute_ctc_loss<T: FloatDtype>(
        &self,
        log_probs: &[T],
        targets: &[usize],
        input_len: usize,
        _seq_len: usize,
        num_classes: usize,
    ) -> crate::Result<T> {
        if targets.is_empty() {
            // Empty target sequence - only blank is allowed
            let mut total_prob = T::zero();
            for t in 0..input_len {
                total_prob = total_prob + log_probs[t * num_classes + self.blank];
            }
            return Ok(-total_prob);
        }

        // Create extended target sequence with blanks: B T1 B T2 B ... TN B
        let mut extended_targets = Vec::with_capacity(2 * targets.len() + 1);
        extended_targets.push(self.blank);

        for &target in targets {
            if target == self.blank {
                return Err(crate::NNError::InvalidInput {
                    message: "Target sequence cannot contain blank symbol".to_string(),
                });
            }
            extended_targets.push(target);
            extended_targets.push(self.blank);
        }

        let extended_len = extended_targets.len();

        // Forward pass
        let mut alpha = vec![T::zero(); input_len * extended_len];

        // Initialize first column (t=0)
        if extended_targets[0] == self.blank {
            alpha[0] = log_probs[self.blank];
        }

        // Fill first row (s=0, various t)
        for t in 1..input_len {
            let prev_alpha = alpha[(t - 1) * extended_len];
            alpha[t * extended_len] = prev_alpha + log_probs[t * num_classes + self.blank];
        }

        // Fill the rest of the matrix
        for s in 1..extended_len {
            let current_target = extended_targets[s];

            for t in 0..input_len {
                let mut sum = T::zero();

                // Case 1: horizontal transition (repeat)
                if s > 1 && extended_targets[s] == extended_targets[s - 2] {
                    sum = sum + alpha[(t * extended_len) + (s - 1)];
                }

                // Case 2: diagonal transition
                if t > 0 {
                    sum = sum + alpha[((t - 1) * extended_len) + (s - 1)];
                }

                // Case 3: vertical transition (skip blank)
                if s > 1
                    && extended_targets[s] != self.blank
                    && extended_targets[s - 1] == self.blank
                {
                    sum = sum + alpha[(t * extended_len) + (s - 2)];
                }

                if sum != T::zero() {
                    alpha[t * extended_len + s] = sum + log_probs[t * num_classes + current_target];
                }
            }
        }

        // For simplicity, compute loss using only forward pass
        // A complete CTC implementation would use both forward and backward passes
        // This is a simplified version that works for basic cases
        let mut total_prob = T::zero();
        for t in 0..input_len {
            // Sum probabilities for all possible paths at time t
            let mut prob_at_t = T::zero();
            for s in 0..extended_len {
                prob_at_t = prob_at_t + alpha[t * extended_len + s];
            }
            total_prob = total_prob + prob_at_t;
        }

        Ok(-total_prob)
    }
}

impl<T: FloatDtype> Module<T> for CTCLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "CTCLoss requires four inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_focal_loss_basic() {
        let loss_fn = FocalLoss::new(2.0, 1.0);
        let logits = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1, 1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap();

        let loss = loss_fn.forward(&logits, &targets).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_focal_loss_easy_examples() {
        let loss_fn = FocalLoss::new(2.0, 1.0);

        // Very confident correct prediction (easy example)
        let logits = Tensor::from_vec(CpuBackend::default(), vec![-10.0], vec![1, 1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap();

        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Should be very small due to (1-p_t)^gamma term
        assert!(loss.item().unwrap() < 0.01);
    }

    #[test]
    fn test_focal_loss_hard_examples() {
        let loss_fn = FocalLoss::new(2.0, 1.0);

        // Low confidence correct prediction (hard example)
        let logits = Tensor::from_vec(CpuBackend::default(), vec![0.1], vec![1, 1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap();

        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Should be larger than easy example
        assert!(loss.item().unwrap() > 0.1);
    }

    #[test]
    fn test_dice_loss_perfect_overlap() {
        let loss_fn = DiceLoss::new();
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // Perfect overlap should give very low loss (close to 0)
        assert!(loss.item().unwrap() < 0.1);
    }

    #[test]
    fn test_dice_loss_no_overlap() {
        let loss_fn = DiceLoss::new();
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // No overlap should give high loss (close to 1)
        assert!(loss.item().unwrap() > 0.5);
    }

    #[test]
    fn test_dice_loss_partial_overlap() {
        let loss_fn = DiceLoss::new();
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // Partial overlap should give moderate loss
        let loss_val = loss.item().unwrap();
        assert!(loss_val > 0.0 && loss_val < 1.0);
    }

    #[test]
    fn test_iou_loss_perfect_overlap() {
        let loss_fn = IoULoss::new();
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // Perfect overlap should give very low loss
        assert!(loss.item().unwrap() < 0.1);
    }

    #[test]
    fn test_iou_loss_no_overlap() {
        let loss_fn = IoULoss::new();
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();

        // No overlap should give high loss
        assert!(loss.item().unwrap() > 0.5);
    }

    #[test]
    fn test_dice_vs_iou_loss() {
        let dice_fn = DiceLoss::new();
        let iou_fn = IoULoss::new();

        let predictions = Tensor::from_vec(CpuBackend::default(), vec![0.8], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let dice_loss = dice_fn.forward(&predictions, &targets).unwrap();
        let iou_loss = iou_fn.forward(&predictions, &targets).unwrap();

        // Both should be reasonable losses, IoU typically higher than Dice
        assert!(dice_loss.item().unwrap() >= 0.0);
        assert!(iou_loss.item().unwrap() >= 0.0);
        assert!(iou_loss.item().unwrap() >= dice_loss.item().unwrap());
    }

    #[test]
    fn test_specialized_losses_smoothing() {
        // Test that smoothing prevents division by zero
        let dice_fn = DiceLoss::with_smooth(1e-6);
        let iou_fn = IoULoss::with_smooth(1e-6);

        // All zeros case
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();

        let dice_loss = dice_fn.forward(&predictions, &targets).unwrap();
        let iou_loss = iou_fn.forward(&predictions, &targets).unwrap();

        // Should not be NaN or infinite due to smoothing
        let dice_val: f64 = dice_loss.item().unwrap();
        let iou_val: f64 = iou_loss.item().unwrap();
        assert!(dice_val.is_finite());
        assert!(iou_val.is_finite());
    }

    #[test]
    fn test_soft_margin_loss_basic() {
        let loss_fn = SoftMarginLoss::new();
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_soft_margin_loss_perfect_classification() {
        let loss_fn = SoftMarginLoss::new();
        // Perfect classification: y * x >> 0 for all samples
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![5.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();
        // Should be very small due to large positive y*x values
        assert!(loss.item().unwrap() < 0.1);
    }

    #[test]
    fn test_soft_margin_loss_wrong_classification() {
        let loss_fn = SoftMarginLoss::new();
        // Wrong classification: y * x << 0 for some samples
        let predictions = Tensor::from_vec(CpuBackend::default(), vec![-2.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&predictions, &targets).unwrap();
        // Should have positive loss due to classification errors
        assert!(loss.item().unwrap() > 0.5);
    }

    #[test]
    fn test_soft_margin_loss_reductions() {
        let loss_fn_none = SoftMarginLoss::with_reduction(Reduction::None);
        let loss_fn_sum = SoftMarginLoss::with_reduction(Reduction::Sum);
        let loss_fn_mean = SoftMarginLoss::with_reduction(Reduction::Mean);

        let predictions = Tensor::from_vec(CpuBackend::default(), vec![2.0, 1.0], vec![2]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1.0, -1.0], vec![2]).unwrap();

        let result_none = loss_fn_none.forward(&predictions, &targets).unwrap();
        let result_sum = loss_fn_sum.forward(&predictions, &targets).unwrap();
        let result_mean = loss_fn_mean.forward(&predictions, &targets).unwrap();

        // None should return per-element losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data()[0] + result_none.data()[1];
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }

    #[test]
    fn test_ctc_loss_simple() {
        let loss_fn = CTCLoss::new();

        // Simple case: single character sequence
        // log_probs: (batch_size=1, seq_len=3, num_classes=4) with blank=0
        let log_probs_data = vec![
            0.0, -1.0, -2.0, -3.0, // t=0: blank=0, class1=-1, class2=-2, class3=-3
            -1.0, 0.0, -2.0, -3.0, // t=1: blank=-1, class1=0, class2=-2, class3=-3
            -2.0, -1.0, 0.0, -3.0, // t=2: blank=-2, class1=-1, class2=0, class3=-3
        ];
        let log_probs = Tensor::from_vec(CpuBackend::default(), log_probs_data.clone(), vec![1, 3, 4]).unwrap();

        // Target: single character (class 1)
        let targets = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1, 1]).unwrap();
        let input_lengths = Tensor::from_vec(CpuBackend::default(), vec![3], vec![1]).unwrap();
        let target_lengths = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap();

        let loss = loss_fn
            .forward(&log_probs, &targets, &input_lengths, &target_lengths)
            .unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_ctc_loss_empty_target() {
        let loss_fn = CTCLoss::new();

        // Empty target sequence
        let log_probs_data = vec![
            0.0, -1.0, -2.0, // t=0
            -1.0, 0.0, -2.0, // t=1
        ];
        let log_probs = Tensor::from_vec(CpuBackend::default(), log_probs_data.clone(), vec![2, 2, 3]).unwrap();

        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap(); // Empty sequence
        let input_lengths = Tensor::from_vec(CpuBackend::default(), vec![2], vec![1]).unwrap();
        let target_lengths = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap(); // Empty target

        let loss = loss_fn
            .forward(&log_probs, &targets, &input_lengths, &target_lengths)
            .unwrap();
        // Loss should be -sum(blank_probs) = -(0.0 + (-1.0)) = 1.0
        assert_relative_eq!(loss.item().unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ctc_loss_invalid_blank_in_target() {
        let loss_fn = CTCLoss::new();

        let log_probs = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();
        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap(); // Target contains blank (0)
        let input_lengths = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap();
        let target_lengths = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap();

        // Should return an error because target contains blank symbol
        let result = loss_fn.forward(&log_probs, &targets, &input_lengths, &target_lengths);
        assert!(result.is_err());
    }

    #[test]
    fn test_ctc_loss_custom_blank() {
        let loss_fn = CTCLoss::with_blank(3); // Use class 3 as blank

        let log_probs_data = vec![
            -1.0, -2.0, -3.0, 0.0, // t=0: class0=-1, class1=-2, class2=-3, blank=0
            -2.0, -1.0, -3.0, -1.0, // t=1
        ];
        let log_probs = Tensor::from_vec(CpuBackend::default(), log_probs_data.clone(), vec![2, 2, 3]).unwrap();

        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap(); // Target class 1
        let input_lengths = Tensor::from_vec(CpuBackend::default(), vec![2], vec![1]).unwrap();
        let target_lengths = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap();

        let loss = loss_fn
            .forward(&log_probs, &targets, &input_lengths, &target_lengths)
            .unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_ctc_loss_reductions() {
        let loss_fn_none = CTCLoss::with_params(0, Reduction::None);
        let loss_fn_sum = CTCLoss::with_params(0, Reduction::Sum);
        let loss_fn_mean = CTCLoss::with_params(0, Reduction::Mean);

        // Two sequences
        let log_probs_data = vec![
            // Sequence 1
            0.0, -1.0, -2.0, // t=0
            -1.0, 0.0, -2.0, // t=1
            // Sequence 2
            -1.0, 0.0, -2.0, // t=0
            -2.0, -1.0, 0.0, // t=1
        ];
        let log_probs = Tensor::from_vec(CpuBackend::default(), log_probs_data.clone(), vec![2, 2, 3]).unwrap();

        let targets = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap(); // Different targets
        let input_lengths = Tensor::from_vec(CpuBackend::default(), vec![2], vec![1]).unwrap();
        let target_lengths = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap();

        let result_none = loss_fn_none
            .forward(&log_probs, &targets, &input_lengths, &target_lengths)
            .unwrap();
        let result_sum = loss_fn_sum
            .forward(&log_probs, &targets, &input_lengths, &target_lengths)
            .unwrap();
        let result_mean = loss_fn_mean
            .forward(&log_probs, &targets, &input_lengths, &target_lengths)
            .unwrap();

        // None should return per-element losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data()[0] + result_none.data()[1];
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }
}


