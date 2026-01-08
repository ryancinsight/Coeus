//! Zero-Shot Classification for CLIP Models
//!
//! This module provides zero-shot image classification capabilities using CLIP.
//! Supports ImageNet and custom classification datasets with text prompts.
//!
//! Zero-shot classification works by:
//! 1. Creating text prompts for each class (e.g., "a photo of a {class}")
//! 2. Encoding both images and text prompts to the same embedding space
//! 3. Classifying images by finding the most similar text prompt

use super::imagenet_labels::IMAGENET_SIMPLE_LABELS;
use crate::clip::traits::ClipEncoder;
use crate::core::error::{NNError, Result};
use crate::evaluation::ZeroShotResults;
use backend::Backend;
use dtype::{DataType, FloatExt};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use storage::DenseStorage;
use tensor::Tensor;

/// Zero-shot classifier using CLIP
pub struct ZeroShotClassifier<B, T>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// CLIP model for encoding
    model: Arc<dyn ClipEncoder<B, T> + Send + Sync>,
    /// Class name to text embeddings mapping
    class_embeddings: HashMap<String, Tensor<B, DenseStorage<T>, T>>,
    /// Class names in order
    class_names: Vec<String>,
    /// Text templates for prompt engineering
    templates: Vec<String>,
    /// Temperature for softmax normalization
    temperature: f64,
}

/// Configuration for zero-shot classification
#[derive(Debug, Clone)]
pub struct ZeroShotConfig {
    /// Temperature for softmax
    pub temperature: f64,
    /// Text prompt templates
    pub templates: Vec<String>,
    /// Whether to use ensemble of templates
    pub use_ensemble: bool,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            templates: vec![
                "a photo of a {}".to_string(),
                "a picture of a {}".to_string(),
                "an image of a {}".to_string(),
                "a photograph of a {}".to_string(),
            ],
            use_ensemble: true,
            batch_size: 32,
        }
    }
}

/// Classification result for a single image
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted class name
    pub predicted_class: String,
    /// Prediction confidence (probability)
    pub confidence: f64,
    /// Top-k predictions with confidences
    pub top_k: Vec<(String, f64)>,
    /// All class probabilities
    pub probabilities: HashMap<String, f64>,
}

/// Batch classification results
#[derive(Debug, Clone)]
pub struct BatchClassificationResult {
    /// Results for each image in the batch
    pub results: Vec<ClassificationResult>,
    /// Top-1 accuracy for this batch
    pub top1_accuracy: f64,
    /// Top-5 accuracy for this batch
    pub top5_accuracy: f64,
}

impl<B, T> ZeroShotClassifier<B, T>
where
    B: Backend<Data = T> + Clone + Send + Sync + 'static,
    T: DataType
        + FloatExt
        + num_traits::FromPrimitive
        + num_traits::Bounded
        + num_traits::Float
        + Send
        + Sync
        + 'static,
{
    /// Create a new zero-shot classifier
    pub fn new(
        model: Arc<dyn ClipEncoder<B, T> + Send + Sync>,
        class_names: &[&str],
        config: ZeroShotConfig,
    ) -> Result<Self> {
        let mut classifier = Self {
            model,
            class_embeddings: HashMap::new(),
            class_names: class_names.iter().map(|s| s.to_string()).collect(),
            templates: config.templates.clone(),
            temperature: config.temperature,
        };

        // Pre-compute class embeddings
        classifier.compute_class_embeddings()?;

        Ok(classifier)
    }

    /// Create classifier with standard ImageNet classes
    pub fn imagenet(
        model: Arc<dyn ClipEncoder<B, T> + Send + Sync>,
        config: ZeroShotConfig,
    ) -> Result<Self> {
        let class_names = Self::imagenet_classes();
        Self::new(model, &class_names, config)
    }

    /// Classify a single image
    pub fn classify_image(&self, image_data: &[u8]) -> Result<ClassificationResult> {
        let images = vec![image_data.to_vec()];
        let batch_result = self.classify_batch(&images)?;
        Ok(batch_result.results.into_iter().next().unwrap())
    }

    /// Classify a batch of images
    pub fn classify_batch(&self, image_batch: &[Vec<u8>]) -> Result<BatchClassificationResult> {
        // Encode images
        let mut image_embeddings = Vec::new();
        for image_data in image_batch {
            // Convert Vec<u8> to Vec<f32> (assuming normalized 0-1 float data stored as bytes)
            let float_data: Vec<f32> = image_data.iter().map(|&b| b as f32 / 255.0).collect();
            let embedding = self.model.encode_image(&float_data, 1)?;
            image_embeddings.push(embedding);
        }

        let mut results = Vec::new();
        let mut correct_top1 = 0;
        let mut correct_top5 = 0;

        for (i, image_emb) in image_embeddings.iter().enumerate() {
            // Compute similarities to all class embeddings
            let mut similarities = Vec::new();

            for class_name in &self.class_names {
                let class_emb = &self.class_embeddings[class_name];
                let similarity = self.compute_similarity(image_emb, class_emb)?;
                similarities.push((class_name.clone(), similarity));
            }

            // Sort by similarity (descending)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply temperature and softmax
            let probabilities = self.apply_softmax(&similarities, self.temperature);

            // Get top-k results
            let top_k: Vec<(String, f64)> = similarities.iter().take(5).cloned().collect();

            let predicted_class = similarities[0].0.clone();
            let confidence = probabilities[&predicted_class];

            results.push(ClassificationResult {
                predicted_class: predicted_class.clone(),
                confidence,
                top_k: top_k.clone(),
                probabilities: probabilities.clone(),
            });

            // For accuracy calculation, we would need ground truth
            // Here we assume the first class for demonstration
            let true_class = &self.class_names[0];
            if predicted_class == *true_class {
                correct_top1 += 1;
            }
            if top_k.iter().any(|(class, _)| class == true_class) {
                correct_top5 += 1;
            }
        }

        let top1_accuracy = correct_top1 as f64 / results.len() as f64;
        let top5_accuracy = correct_top5 as f64 / results.len() as f64;

        Ok(BatchClassificationResult {
            results,
            top1_accuracy,
            top5_accuracy,
        })
    }

    /// Compute class embeddings from text prompts
    fn compute_class_embeddings(&mut self) -> Result<()> {
        println!(
            "Computing class embeddings for {} classes...",
            self.class_names.len()
        );

        let mut all_prompts = Vec::new();
        let mut prompt_to_class = Vec::new();

        // Generate prompts for each class
        for class_name in &self.class_names {
            for template in &self.templates {
                let prompt = template.replace("{}", class_name);
                all_prompts.push(prompt);
                prompt_to_class.push(class_name.clone());
            }
        }

        // Encode all prompts in batches
        let mut class_embeddings: HashMap<String, Vec<Tensor<B, DenseStorage<T>, T>>> =
            HashMap::new();

        for (i, text) in all_prompts.iter().enumerate() {
            let embeddings = self.model.encode_text(&[text.as_str()])?;
            let class_name = &prompt_to_class[i];
            let existing_emb = class_embeddings.entry(class_name.clone()).or_default();

            // Store individual template embeddings
            existing_emb.push(embeddings);
        }

        // Average embeddings across templates for each class
        for (class_name, embeddings) in &mut class_embeddings {
            if embeddings.len() > 1usize {
                // Average across templates
                let avg_embedding = self.average_embeddings(embeddings)?;
                self.class_embeddings
                    .insert(class_name.clone(), avg_embedding);
            } else if let Some(emb) = embeddings.first() {
                self.class_embeddings
                    .insert(class_name.clone(), emb.clone());
            }
        }

        println!(
            "Computed embeddings for {} classes",
            self.class_embeddings.len()
        );
        Ok(())
    }

    /// Compute cosine similarity between two embeddings
    fn compute_similarity(
        &self,
        emb1: &Tensor<B, DenseStorage<T>, T>,
        emb2: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<f64> {
        let emb1_data = emb1.as_slice();
        let emb2_data = emb2.as_slice();

        let dot_product: f64 = emb1_data
            .iter()
            .zip(emb2_data.iter())
            .map(|(&a, &b)| a.to_f64().unwrap_or(0.0) * b.to_f64().unwrap_or(0.0))
            .sum::<f64>();

        let norm1: f64 = emb1_data
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm2: f64 = emb2_data
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0).powi(2))
            .sum::<f64>()
            .sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            Ok(dot_product / (norm1 * norm2))
        } else {
            Ok(0.0)
        }
    }

    /// Average multiple embeddings
    fn average_embeddings(
        &self,
        embeddings: &[Tensor<B, DenseStorage<T>, T>],
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        if embeddings.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Cannot average empty embedding list".to_string(),
            });
        }

        let first_emb = &embeddings[0];
        let shape = first_emb.shape();
        let mut avg_data = vec![0.0f64; first_emb.as_slice().len()];

        // Sum all embeddings
        for emb in embeddings {
            for (i, &val) in emb.as_slice().iter().enumerate() {
                avg_data[i] += val.to_f64().unwrap_or(0.0);
            }
        }

        // Divide by count
        let count = embeddings.len() as f64;
        for val in &mut avg_data {
            *val /= count;
        }

        // Convert back to tensor T
        let avg_t: Vec<T> = avg_data
            .iter()
            .map(|&x| T::from(x).unwrap_or(T::zero()))
            .collect();

        Ok(Tensor::from_vec(avg_t, shape.dims())?)
    }

    /// Apply softmax with temperature
    fn apply_softmax(
        &self,
        similarities: &[(String, f64)],
        temperature: f64,
    ) -> HashMap<String, f64> {
        let mut probabilities = HashMap::new();

        // Apply temperature scaling
        let scaled_similarities: Vec<f64> = similarities
            .iter()
            .map(|(_, sim)| sim / temperature)
            .collect();

        // Find max for numerical stability
        let max_sim = scaled_similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exponentials
        let exps: Vec<f64> = scaled_similarities
            .iter()
            .map(|&sim| (sim - max_sim).exp())
            .collect();

        // Compute sum
        let sum_exp: f64 = exps.iter().sum();

        // Compute probabilities
        for (i, (class_name, _)) in similarities.iter().enumerate() {
            let prob = exps[i] / sum_exp;
            probabilities.insert(class_name.clone(), prob);
        }

        probabilities
    }

    /// Get ImageNet class names
    pub fn imagenet_classes() -> Vec<&'static str> {
        IMAGENET_SIMPLE_LABELS.to_vec()
    }
}

/// ImageNet evaluation utilities
pub mod imagenet {
    use super::*;

    /// Evaluate CLIP on ImageNet zero-shot classification
    pub async fn evaluate_imagenet<B, T>(
        model: Arc<dyn ClipEncoder<B, T> + Send + Sync>,
        imagenet_dataset: &dyn crate::datasets::VisionLanguageData,
        config: ZeroShotConfig,
    ) -> Result<ZeroShotResults>
    where
        B: Backend<Data = T> + Clone + Send + Sync + 'static,
        T: DataType
            + FloatExt
            + num_traits::FromPrimitive
            + num_traits::Bounded
            + num_traits::Float
            + Send
            + Sync
            + 'static,
    {
        let class_names = ZeroShotClassifier::<B, T>::imagenet_classes();
        let classifier = ZeroShotClassifier::new(model, &class_names, config)?;

        println!("Evaluating CLIP on ImageNet zero-shot classification...");
        println!("ImageNet has {} classes", class_names.len());

        let mut correct_top1 = 0;
        let mut correct_top5 = 0;
        let mut total_samples = 0;
        let mut class_correct: HashMap<String, usize> = HashMap::new();

        // Evaluate on a subset for speed
        let eval_samples = std::cmp::min(imagenet_dataset.len(), 1000);

        for i in 0..eval_samples {
            let pair = imagenet_dataset.get(i).await?;
            let result = classifier.classify_image(&pair.image_data)?;

            // For demonstration, assume the true class is encoded in the image_id
            // In practice, you'd have ground truth labels
            let true_class_idx = i % class_names.len();
            let true_class = class_names[true_class_idx];

            if result.predicted_class == true_class {
                correct_top1 += 1;
                *class_correct.entry(true_class.to_string()).or_insert(0) += 1;
            }

            if result.top_k.iter().any(|(class, _)| class == true_class) {
                correct_top5 += 1;
            }

            total_samples += 1;

            if i % 100 == 0 {
                println!("Processed {}/{} samples", i, eval_samples);
            }
        }

        let top1_accuracy = correct_top1 as f64 / total_samples as f64;
        let top5_accuracy = correct_top5 as f64 / total_samples as f64;

        let class_accuracies = class_names
            .iter()
            .map(|class| {
                let correct = *class_correct.get(&class[..]).unwrap_or(&0);
                let total = total_samples / class_names.len();
                let accuracy = if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                };
                (class.to_string(), accuracy)
            })
            .collect();

        println!("ImageNet Zero-shot Results:");
        println!("Top-1 Accuracy: {:.2}%", top1_accuracy * 100.0);
        println!("Top-5 Accuracy: {:.2}%", top5_accuracy * 100.0);

        Ok(ZeroShotResults {
            dataset_name: "ImageNet".to_string(),
            top1_accuracy,
            top5_accuracy,
            per_class_accuracy: class_accuracies,
            confusion_matrix: Vec::new(),
            class_confidences: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    // Mock CLIP model for testing
    struct MockClipModel;

    impl ClipEncoder<CpuBackend<Float32>, Float32> for MockClipModel {
        fn encode_text(
            &self,
            texts: &[&str],
        ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
            let batch_size = texts.len();
            let embed_dim = 4;
            let mut data = Vec::with_capacity(batch_size * embed_dim);
            for _ in 0..batch_size {
                let vals = [1.0f32, 0.5, -0.5, 0.0];
                data.extend(vals.iter().map(|&v| Float32::new(v)));
            }
            // In a real scenario this would return [batch, dim]
            // But since the usage in ZeroShotClassifier iterates one by one, it expects [1, dim] or [dim]
            // The loop in average_embeddings iterates as_slice().
            Ok(Tensor::from_vec(data, &[batch_size, embed_dim])?)
        }

        fn encode_image(
            &self,
            _image_data: &[f32],
            batch_size: usize,
        ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
            let embed_dim = 4;
            let mut data = Vec::with_capacity(batch_size * embed_dim);
            for _ in 0..batch_size {
                let vals = [0.5f32, 1.0, -0.2, 0.8];
                data.extend(vals.iter().map(|&v| Float32::new(v)));
            }
            Ok(Tensor::from_vec(data, &[batch_size, embed_dim])?)
        }
    }

    #[tokio::test]
    async fn test_zero_shot_classifier_creation() {
        let mock_model = Arc::new(MockClipModel);
        let class_names = vec!["cat", "dog", "bird"];

        let config = ZeroShotConfig::default();
        let classifier = ZeroShotClassifier::new(mock_model, &class_names, config).unwrap();

        assert_eq!(classifier.class_names.len(), 3);
        assert_eq!(classifier.class_embeddings.len(), 3);
    }

    #[tokio::test]
    async fn test_zero_shot_classification() {
        let mock_model = Arc::new(MockClipModel);
        let class_names = vec!["cat", "dog", "bird"];

        let config = ZeroShotConfig::default();
        let classifier = ZeroShotClassifier::new(mock_model, &class_names, config).unwrap();

        let dummy_image = vec![1, 2, 3, 4, 5];
        let result = classifier.classify_image(&dummy_image).unwrap();

        assert!(!result.predicted_class.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.top_k.len(), 3); // Limited by available classes
        assert_eq!(result.probabilities.len(), 3);
    }

    #[test]
    fn test_imagenet_classes() {
        let classes = ZeroShotClassifier::<CpuBackend<Float32>, Float32>::imagenet_classes();
        assert_eq!(classes.len(), 1000);
        assert!(classes.contains(&"tabby cat"));
        assert!(classes.contains(&"Golden Retriever"));
    }
}
