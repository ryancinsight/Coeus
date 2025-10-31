//! CLIP Zero-shot Classification Evaluation
//!
//! Implements zero-shot image classification using CLIP text embeddings
//! to evaluate model performance on downstream classification tasks.

use std::collections::HashMap;
use std::time::Instant;

use crate::error::{NNError, Result};
use super::{ClipEvaluationConfig, BenchmarkDataset, ClipModelEvaluator};
use backend::Backend;
use storage::Storage;
use dtype::{DataType, traits::FloatExt};

/// Zero-shot classification results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroShotResults {
    /// Dataset name
    pub dataset_name: String,
    /// Top-1 classification accuracy
    pub top1_accuracy: f64,
    /// Top-5 classification accuracy
    pub top5_accuracy: f64,
    /// Per-class accuracy breakdown
    pub per_class_accuracy: HashMap<String, f64>,
    /// Confusion matrix data
    pub confusion_matrix: Vec<Vec<f64>>,
    /// Class prediction confidences
    pub class_confidences: Vec<Vec<f64>>,
}

/// Text template for zero-shot classification prompts
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// Template string with {class} placeholder
    pub template: String,
    /// List of class names to fill the template
    pub class_names: Vec<String>,
}

impl PromptTemplate {
    /// Create standard CLIP prompt template
    pub fn clip_default(class_names: Vec<String>) -> Self {
        Self {
            template: "a photo of a {class}".to_string(),
            class_names,
        }
    }

    /// Create custom prompt template
    pub fn custom(template: String, class_names: Vec<String>) -> Self {
        Self {
            template,
            class_names,
        }
    }

    /// Generate prompts for all classes
    pub fn generate_prompts(&self) -> Vec<String> {
        self.class_names
            .iter()
            .map(|class| self.template.replace("{class}", class))
            .collect()
    }
}

/// Zero-shot classification evaluator
pub struct ZeroShotEvaluator {
    config: ClipEvaluationConfig,
}

impl ZeroShotEvaluator {
    /// Create new zero-shot evaluator
    pub fn new(config: ClipEvaluationConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Evaluate zero-shot classification on multiple datasets
    pub fn evaluate_zeroshot<B, S, T, M>(
        &self,
        model: &M,
        datasets: &[BenchmarkDataset],
    ) -> Result<Vec<ZeroShotResults>>
    where
        B: Backend<Data = T> + Clone + Send + Sync,
        S: Storage<T> + Clone + Send + Sync,
        T: DataType + FloatExt + Clone + Send + Sync,
        M: ClipModelEvaluator<B, S, T>,
    {
        let mut results = Vec::new();
        let start_time = Instant::now();

        println!("🎯 Starting CLIP zero-shot classification evaluation on {} datasets", datasets.len());

        for dataset in datasets {
            println!("  Evaluating zero-shot on dataset: {}", dataset.name);
            let dataset_result = self.evaluate_single_dataset(model, dataset)?;
            results.push(dataset_result);
        }

        let total_time = start_time.elapsed().as_secs_f64();
        println!("✅ Zero-shot evaluation completed in {:.2}s", total_time);

        Ok(results)
    }

    /// Evaluate zero-shot classification on single dataset
    pub fn evaluate_single_dataset<B, S, T, M>(
        &self,
        model: &M,
        dataset: &BenchmarkDataset,
    ) -> Result<ZeroShotResults>
    where
        B: Backend<Data = T> + Clone + Send + Sync,
        S: Storage<T> + Clone + Send + Sync,
        T: DataType + FloatExt + Clone + Send + Sync,
        M: ClipModelEvaluator<B, S, T>,
    {
        let start_time = Instant::now();

        println!("  Generating prompts for {} classes...", dataset.class_names.len());

        // Create prompt template and generate text prompts
        let template = PromptTemplate::clip_default(dataset.class_names.clone());
        let text_prompts = template.generate_prompts();

        // Encode text prompts to embeddings
        println!("  Encoding text prompts...");
        let text_embeddings = model.encode_texts(
            &text_prompts,
            self.config.eval_batch_size,
        )?;

        // Encode image embeddings (use provided normalized embeddings)
        let image_embeddings = &dataset.image_embeddings;
        let labels = &dataset.labels;

        println!("  Computing image-text similarities...");
        let similarities = self.compute_image_text_similarities(image_embeddings, &text_embeddings)?;

        println!("  Computing classification predictions...");
        let predictions = self.compute_predictions(&similarities, dataset.class_names.len())?;

        println!("  Computing accuracy metrics...");
        let top1_accuracy = self.compute_top_k_accuracy(&predictions, labels, 1);
        let top5_accuracy = self.compute_top_k_accuracy(&predictions, labels, 5);

        let per_class_accuracy = self.compute_per_class_accuracy(&predictions, labels, &dataset.class_names)?;
        let confusion_matrix = self.compute_confusion_matrix(&predictions, labels, dataset.class_names.len())?;
        let class_confidences = self.extract_class_confidences(&similarities, dataset.class_names.len())?;

        let eval_time = start_time.elapsed().as_secs_f64();
        println!("  Zero-shot evaluation completed in {:.2}s", eval_time);

        Ok(ZeroShotResults {
            dataset_name: dataset.name.clone(),
            top1_accuracy,
            top5_accuracy,
            per_class_accuracy,
            confusion_matrix,
            class_confidences,
        })
    }

    /// Compute similarities between all images and text prompts
    fn compute_image_text_similarities(
        &self,
        image_embeddings: &[Vec<f32>],
        text_embeddings: &[Vec<f32>],
    ) -> Result<Vec<Vec<f64>>> {
        let mut similarities = Vec::with_capacity(image_embeddings.len());

        for img_emb in image_embeddings {
            let mut img_similarities = Vec::with_capacity(text_embeddings.len());
            for text_emb in text_embeddings {
                let similarity = self.cosine_similarity(img_emb, text_emb);
                img_similarities.push(similarity);
            }
            similarities.push(img_similarities);
        }

        Ok(similarities)
    }

    /// Cosine similarity between two normalized vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            dot_product += (*x as f64) * (*y as f64);
            norm_a += (*x as f64).powi(2);
            norm_b += (*y as f64).powi(2);
        }

        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Compute predictions from similarity matrix
    fn compute_predictions(&self, similarities: &[Vec<f64>], num_classes: usize) -> Result<Vec<Vec<(usize, f64)>>> {
        let mut predictions = Vec::with_capacity(similarities.len());

        for similarity_row in similarities {
            // Create (class_idx, similarity) pairs
            let mut class_similarities: Vec<(usize, f64)> = similarity_row
                .iter()
                .enumerate()
                .take(num_classes)
                .map(|(idx, &sim)| (idx, sim))
                .collect();

            // Sort by similarity descending
            class_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            predictions.push(class_similarities);
        }

        Ok(predictions)
    }

    /// Compute top-k accuracy
    fn compute_top_k_accuracy(&self, predictions: &[Vec<(usize, f64)>], labels: &[String], k: usize) -> f64 {
        let mut correct = 0;

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            // Parse label as class index (assuming labels are numeric strings)
            if let Ok(true_class) = label.parse::<usize>() {
                // Check if true class is in top-k predictions
                let is_correct = pred.iter().take(k).any(|(pred_class, _)| *pred_class == true_class);
                if is_correct {
                    correct += 1;
                }
            }
        }

        correct as f64 / labels.len() as f64
    }

    /// Compute per-class accuracy
    fn compute_per_class_accuracy(
        &self,
        predictions: &[Vec<(usize, f64)>],
        labels: &[String],
        class_names: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut class_correct = HashMap::new();
        let mut class_total = HashMap::new();

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            if let Ok(true_class) = label.parse::<usize>() {
                let class_name = class_names.get(true_class).unwrap_or(&format!("class_{}", true_class)).clone();
                let is_correct = pred.iter().take(1).any(|(pred_class, _)| *pred_class == true_class);

                *class_total.entry(class_name).or_insert(0) += 1;
                if is_correct {
                    *class_correct.entry(class_name).or_insert(0) += 1;
                }
            }
        }

        let mut per_class_accuracy = HashMap::new();
        for (class_name, total) in class_total {
            let correct = class_correct.get(&class_name).copied().unwrap_or(0);
            let accuracy = correct as f64 / total as f64;
            per_class_accuracy.insert(class_name, accuracy);
        }

        Ok(per_class_accuracy)
    }

    /// Compute confusion matrix
    fn compute_confusion_matrix(
        &self,
        predictions: &[Vec<(usize, f64)>],
        labels: &[String],
        num_classes: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let mut confusion_matrix = vec![vec![0.0; num_classes]; num_classes];

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            if let Ok(true_class) = label.parse::<usize>() {
                let pred_class = pred[0].0;
                if pred_class < num_classes && true_class < num_classes {
                    confusion_matrix[true_class][pred_class] += 1.0;
                }
            }
        }

        Ok(confusion_matrix)
    }

    /// Extract class prediction confidences
    fn extract_class_confidences(
        &self,
        similarities: &[Vec<f64>],
        num_classes: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let mut confidences = Vec::with_capacity(similarities.len());

        for similarity_row in similarities {
            let class_similarities: Vec<f64> = similarity_row.iter()
                .take(num_classes)
                .map(|&sim| sim)
                .collect();

            // Convert to softmax probabilities for better interpretation
            let softmax_confidences = self.compute_softmax(&class_similarities)?;
            confidences.push(softmax_confidences);
        }

        Ok(confidences)
    }

    /// Compute softmax probabilities
    fn compute_softmax(&self, values: &[f64]) -> Result<Vec<f64>> {
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f64>();

        if sum_exp == 0.0 {
            return Err(NNError::InvalidInput {
                message: "Softmax computation resulted in zero sum".to_string(),
            });
        }

        Ok(exp_values.iter().map(|&x| x / sum_exp).collect())
    }
}

/// Prompt ensemble for improved zero-shot performance
pub struct PromptEnsemble {
    templates: Vec<String>,
    class_names: Vec<String>,
}

impl PromptEnsemble {
    /// Create ensemble with multiple prompt templates
    pub fn new(templates: Vec<String>, class_names: Vec<String>) -> Self {
        Self {
            templates,
            class_names,
        }
    }

    /// Create standard CLIP ensemble
    pub fn clip_standard(class_names: Vec<String>) -> Self {
        let templates = vec![
            "a photo of a {class}".to_string(),
            "a picture of a {class}".to_string(),
            "an image of a {class}".to_string(),
            "a photograph of a {class}".to_string(),
            "a photo of the {class}".to_string(),
        ];

        Self::new(templates, class_names)
    }

    /// Generate all ensemble prompts
    pub fn generate_ensemble_prompts(&self) -> Vec<String> {
        let mut all_prompts = Vec::new();

        for template in &self.templates {
            for class in &self.class_names {
                let prompt = template.replace("{class}", class);
                all_prompts.push(prompt);
            }
        }

        all_prompts
    }

    /// Average embeddings across ensemble prompts for each class
    pub fn average_class_embeddings(&self, all_embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let prompts_per_class = self.templates.len();
        let num_classes = self.class_names.len();

        let mut averaged_embeddings = Vec::with_capacity(num_classes);

        for class_idx in 0..num_classes {
            let mut class_embeddings = Vec::new();
            for prompt_idx in 0..prompts_per_class {
                let embedding_idx = class_idx * prompts_per_class + prompt_idx;
                if embedding_idx < all_embeddings.len() {
                    class_embeddings.push(all_embeddings[embedding_idx].clone());
                }
            }

            // Average embeddings
            if !class_embeddings.is_empty() {
                let averaged = Self::average_embeddings(&class_embeddings);
                averaged_embeddings.push(averaged);
            }
        }

        averaged_embeddings
    }

    /// Average multiple embeddings
    fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let embedding_dim = embeddings[0].len();
        let num_embeddings = embeddings.len();

        let mut averaged = vec![0.0; embedding_dim];

        for embedding in embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                averaged[i] += val;
            }
        }

        for val in averaged.iter_mut() {
            *val /= num_embeddings as f32;
        }

        averaged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> BenchmarkDataset {
        BenchmarkDataset {
            name: "test_dataset".to_string(),
            image_embeddings: vec![
                vec![1.0, 0.0, 0.0], // Should match class 0
                vec![0.0, 1.0, 0.0], // Should match class 1
                vec![0.0, 0.0, 1.0], // Should match class 2
            ],
            labels: vec!["0".to_string(), "1".to_string(), "2".to_string()],
            class_names: vec!["red".to_string(), "green".to_string(), "blue".to_string()],
        }
    }

    fn create_test_text_embeddings() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0], // "red" embedding
            vec![0.0, 1.0, 0.0], // "green" embedding
            vec![0.0, 0.0, 1.0], // "blue" embedding
        ]
    }

    #[test]
    fn test_cosine_similarity_perfect_match() {
        let evaluator = ZeroShotEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let similarity = evaluator.cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_no_match() {
        let evaluator = ZeroShotEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let similarity = evaluator.cosine_similarity(&a, &b);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_prompt_template_generation() {
        let class_names = vec!["dog".to_string(), "cat".to_string()];
        let template = PromptTemplate::clip_default(class_names);

        let prompts = template.generate_prompts();
        assert_eq!(prompts, vec!["a photo of a dog", "a photo of a cat"]);
    }

    #[test]
    fn test_top_k_accuracy_perfect() {
        let evaluator = ZeroShotEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        // Perfect predictions: each prediction has the correct class as first choice
        let predictions = vec![
            vec![(0, 1.0), (1, 0.8)], // Correct class 0 is top-1
            vec![(1, 1.0), (0, 0.8)], // Correct class 1 is top-1
        ];
        let labels = vec!["0".to_string(), "1".to_string()];

        let top1_acc = evaluator.compute_top_k_accuracy(&predictions, &labels, 1);
        assert!((top1_acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_computation() {
        let evaluator = ZeroShotEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        let values = vec![0.0, 1.0, 2.0];
        let softmax = evaluator.compute_softmax(&values).unwrap();

        // Softmax should sum to 1
        let sum: f64 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Third value should be highest probability (exp(2) > exp(1) > exp(0))
        assert!(softmax[2] > softmax[1] && softmax[1] > softmax[0]);
    }

    #[test]
    fn test_prompt_ensemble_generation() {
        let ensemble = PromptEnsemble::clip_standard(vec!["dog".to_string(), "cat".to_string()]);

        let prompts = ensemble.generate_ensemble_prompts();
        assert_eq!(prompts.len(), 10); // 2 classes × 5 templates

        assert!(prompts.contains(&"a photo of a dog".to_string()));
        assert!(prompts.contains(&"a picture of a dog".to_string()));
        assert!(prompts.contains(&"a photo of a cat".to_string()));
        assert!(prompts.contains(&"a photograph of a cat".to_string()));
    }

    #[test]
    fn test_embedding_averaging() {
        let ensemble = PromptEnsemble::clip_standard(vec!["class1".to_string()]);

        let embeddings = vec![
            vec![1.0, 2.0], // First prompt
            vec![3.0, 4.0], // Second prompt
            vec![5.0, 6.0], // Third prompt
        ];

        let averaged = ensemble.average_class_embeddings(&embeddings);
        assert_eq!(averaged.len(), 1);
        assert_eq!(averaged[0], vec![3.0, 4.0]); // Average: (1+3+5)/3=3, (2+4+6)/3=4
    }
}
