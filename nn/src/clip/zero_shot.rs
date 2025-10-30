//! Zero-Shot CLIP Framework
//!
//! This module provides zero-shot classification, image-text retrieval,
//! and similarity search capabilities using pre-trained CLIP models.

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};

use crate::error::{NNError, Result};

use super::config::ClipConfig;
use super::model::ClipModel;
use super::preprocessing::{ImageProcessor, TextProcessor};

/// Zero-shot classification results
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted class labels
    pub labels: Vec<String>,
    /// Prediction probabilities (softmax normalized)
    pub probabilities: Vec<f32>,
    /// Top-k predictions with indices and scores
    pub top_k: Vec<(usize, f32)>,
}

/// Image-text retrieval results
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Top-k similar texts for image query
    pub image_to_text: Vec<(String, f32)>,
    /// Top-k similar images for text query
    pub text_to_image: Vec<(usize, f32)>,
    /// Similarity matrix [num_images, num_texts]
    pub similarities: Vec<Vec<f32>>,
}

/// Similarity search result
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Most similar items with indices and scores
    pub similarities: Vec<(usize, f32)>,
    /// Thresholded results above similarity cutoff
    pub above_threshold: Vec<(usize, f32)>,
}

/// Zero-shot CLIP classifier
pub struct ClipClassifier<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// CLIP model for zero-shot classification
    model: ClipModel<B, S, T>,
    /// Image processor
    image_processor: ImageProcessor,
    /// Text processor
    text_processor: TextProcessor,
    /// Class name templates for prompt engineering
    templates: Vec<String>,
}

impl<B, S, T> ClipClassifier<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create new classifier from CLIP model
    pub fn new(model: ClipModel<B, S, T>) -> Self {
        Self {
            model,
            image_processor: ImageProcessor::default(),
            text_processor: TextProcessor::default(),
            templates: Self::default_templates(),
        }
    }

    /// Zero-shot image classification
    ///
    /// # Arguments
    /// * `images` - Batch of images [batch_size, height, width, channels]
    /// * `class_names` - List of class names to classify against
    /// * `batch_size` - Number of images to process
    ///
    /// # Returns
    /// Classification results for each image
    pub fn classify(
        &self,
        images: &[f32],
        class_names: &[String],
        batch_size: usize,
    ) -> Result<Vec<ClassificationResult>> {
        if images.is_empty() || class_names.is_empty() {
            return Err(NNError::InvalidInput {
                message: "images and class_names cannot be empty".to_string(),
            });
        }

        // Process images
        let processed_images = self.image_processor.preprocess_batch(
            images,
            224, // Standard CLIP input size
            224,
            batch_size,
        );

        // Get image embeddings
        let image_embeddings = self.model.encode_image(&processed_images, batch_size)?;

        // Create text prompts for each class
        let text_prompts = self.create_text_prompts(class_names);

        // Get text embeddings for all prompts
        let mut all_text_embeddings = Vec::new();
        for prompts in &text_prompts {
            let text_batch: Vec<String> = prompts.iter()
                .map(|s| s.as_str())
                .collect();

            let embeddings = self.model.encode_text(&text_batch)?;
            all_text_embeddings.push(embeddings);
        }

        // Average embeddings across templates for each class
        let class_text_embeddings = self.average_template_embeddings(&all_text_embeddings, class_names.len())?;

        // Compute similarities and classify
        let mut results = Vec::new();
        for image_emb in image_embeddings.chunks(self.model.config().embed_dim) {
            let result = self.classify_single_image(image_emb, &class_text_embeddings, class_names)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Create text prompts using templates and class names
    fn create_text_prompts(&self, class_names: &[String]) -> Vec<Vec<String>> {
        class_names.iter().enumerate().map(|(i, class_name)| {
            self.templates.iter().map(|template| {
                template.replace("{}", class_name)
            }).collect()
        }).collect()
    }

    /// Average text embeddings across templates for each class
    fn average_template_embeddings(
        &self,
        all_embeddings: &[Vec<f32>],
        num_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let mut averaged_embeddings = Vec::new();

        for class_idx in 0..num_classes {
            let mut class_embedding = vec![0.0f32; self.model.config().embed_dim];

            for template_idx in 0..self.templates.len() {
                let embedding_idx = class_idx * self.templates.len() + template_idx;
                if embedding_idx < all_embeddings.len() {
                    let embedding = &all_embeddings[embedding_idx];
                    for (i, &val) in embedding.iter().enumerate() {
                        class_embedding[i] += val;
                    }
                }
            }

            // Average across templates
            let num_templates = self.templates.len() as f32;
            for val in &mut class_embedding {
                *val /= num_templates;
            }

            // L2 normalize
            self.l2_normalize(&mut class_embedding)?;

            averaged_embeddings.push(class_embedding);
        }

        Ok(averaged_embeddings)
    }

    /// Classify single image against all classes
    fn classify_single_image(
        &self,
        image_embedding: &[f32],
        class_embeddings: &[Vec<f32>],
        class_names: &[String],
    ) -> Result<ClassificationResult> {
        let mut similarities = Vec::new();

        // Compute similarity to each class
        for class_emb in class_embeddings {
            let similarity = self.cosine_similarity(image_embedding, class_emb);
            similarities.push(similarity);
        }

        // Apply softmax to get probabilities
        let probabilities = self.softmax(&similarities)?;

        // Get top-k results
        let mut indexed_similarities: Vec<(usize, f32)> = similarities
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k = indexed_similarities.into_iter().take(5).collect();

        let labels = class_names.to_vec();

        Ok(ClassificationResult {
            labels,
            probabilities,
            top_k,
        })
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            dot_product += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Apply softmax to logits
    fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        if sum_exp == 0.0 {
            return Err(NNError::InvalidInput {
                message: "Softmax sum is zero".to_string(),
            });
        }

        Ok(exp_logits.iter().map(|&x| x / sum_exp).collect())
    }

    /// L2 normalize a vector in-place
    fn l2_normalize(&self, vector: &mut [f32]) -> Result<()> {
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();

        if norm == 0.0 {
            return Err(NNError::InvalidInput {
                message: "Cannot normalize zero vector".to_string(),
            });
        }

        for val in vector {
            *val /= norm;
        }

        Ok(())
    }

    /// Default prompt templates for zero-shot classification
    fn default_templates() -> Vec<String> {
        vec![
            "a photo of a {}".to_string(),
            "a picture of a {}".to_string(),
            "an image of a {}".to_string(),
            "a {} in a photo".to_string(),
            "a {} in a picture".to_string(),
        ]
    }

    /// Get mutable access to templates for customization
    pub fn templates_mut(&mut self) -> &mut Vec<String> {
        &mut self.templates
    }
}

/// Image-text retrieval system
pub struct ImageTextRetriever<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// CLIP model for retrieval
    model: ClipModel<B, S, T>,
    /// Image processor
    image_processor: ImageProcessor,
}

impl<B, S, T> ImageTextRetriever<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create new retriever
    pub fn new(model: ClipModel<B, S, T>) -> Self {
        Self {
            model,
            image_processor: ImageProcessor::default(),
        }
    }

    /// Retrieve similar texts for given images
    pub fn retrieve_similar_texts(
        &self,
        query_images: &[f32],
        candidate_texts: &[String],
        image_batch_size: usize,
        top_k: usize,
    ) -> Result<RetrievalResult> {
        if query_images.is_empty() || candidate_texts.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Query images and candidate texts cannot be empty".to_string(),
            });
        }

        // Process query images
        let processed_images = self.image_processor.preprocess_batch(
            query_images,
            224,
            224,
            image_batch_size,
        );

        // Get image embeddings
        let image_embeddings = self.model.encode_image(&processed_images, image_batch_size)?;

        // Get text embeddings
        let text_embeddings = self.model.encode_text(&candidate_texts.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;

        // Compute similarity matrix
        let similarities = self.model.get_similarity(&image_embeddings, &text_embeddings)?;

        // Extract top-k results for each image
        let mut image_to_text_results = Vec::new();

        // For each image, find top-k most similar texts
        for image_idx in 0..image_batch_size {
            let mut text_similarities = Vec::new();

            for text_idx in 0..candidate_texts.len() {
                let similarity = self.extract_similarity(&similarities, image_idx, text_idx)?;
                text_similarities.push((text_idx, similarity));
            }

            text_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_k_texts: Vec<(String, f32)> = text_similarities
                .into_iter()
                .take(top_k)
                .map(|(idx, score)| (candidate_texts[idx].clone(), score))
                .collect();

            image_to_text_results.push(top_k_texts);
        }

        // For text-to-image retrieval (simplified - using first image as representative)
        let mut text_to_image_results = Vec::new();
        if let Some(first_image_sims) = image_to_text_results.first() {
            let mut image_similarities: Vec<(usize, f32)> = first_image_sims
                .iter()
                .enumerate()
                .map(|(i, (_, score))| (i, *score))
                .collect();

            image_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            text_to_image_results = image_similarities.into_iter().take(top_k).collect();
        }

        // Convert similarity matrix to 2D vec
        let similarity_matrix = self.similarity_tensor_to_vec(&similarities, image_batch_size, candidate_texts.len())?;

        Ok(RetrievalResult {
            image_to_text: image_to_text_results.into_iter().flatten().collect(),
            text_to_image: text_to_image_results,
            similarities: similarity_matrix,
        })
    }

    /// Extract similarity value from tensor
    fn extract_similarity(
        &self,
        similarities: &Tensor<crate::backend::CpuBackend<T>, DenseStorage<T>, T>,
        image_idx: usize,
        text_idx: usize,
    ) -> Result<f32> {
        // Extract similarity value (simplified implementation)
        let flat_idx = image_idx * image_idx + text_idx; // Assuming square matrix
        if flat_idx < similarities.as_slice().len() {
            Ok(similarities.as_slice()[flat_idx] as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Convert similarity tensor to 2D vector
    fn similarity_tensor_to_vec(
        &self,
        _similarities: &Tensor<crate::backend::CpuBackend<T>, DenseStorage<T>, T>,
        _num_images: usize,
        _num_texts: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // Placeholder implementation
        Ok(vec![vec![0.5; 10]; 5])
    }
}

/// CLIP inference API for embedding extraction and similarity search
pub struct ClipInference<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// CLIP model
    model: ClipModel<B, S, T>,
    /// Image processor
    image_processor: ImageProcessor,
}

impl<B, S, T> ClipInference<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create new inference API
    pub fn new(model: ClipModel<B, S, T>) -> Self {
        Self {
            model,
            image_processor: ImageProcessor::default(),
        }
    }

    /// Extract embeddings from images
    pub fn embed_images(
        &self,
        images: &[f32],
        batch_size: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let processed_images = self.image_processor.preprocess_batch(images, height, width, batch_size);
        let embeddings = self.model.encode_image(&processed_images, batch_size)?;

        // Convert to Vec<f32> (simplified)
        Ok(embeddings.as_slice().iter().map(|&x| x as f32).collect())
    }

    /// Extract embeddings from texts
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<f32>> {
        let embeddings = self.model.encode_text(texts)?;
        Ok(embeddings.as_slice().iter().map(|&x| x as f32).collect())
    }

    /// Find most similar items using embeddings
    pub fn similarity_search(
        &self,
        query_embedding: &[f32],
        candidate_embeddings: &[f32],
        top_k: usize,
        threshold: Option<f32>,
    ) -> Result<SimilarityResult> {
        if candidate_embeddings.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Candidate embeddings cannot be empty".to_string(),
            });
        }

        let embed_dim = self.model.config().embed_dim;
        let num_candidates = candidate_embeddings.len() / embed_dim;

        let mut similarities = Vec::new();

        // Compute similarities to all candidates
        for i in 0..num_candidates {
            let start = i * embed_dim;
            let end = start + embed_dim;
            let candidate_emb = &candidate_embeddings[start..end];

            let similarity = self.cosine_similarity(query_embedding, candidate_emb);
            similarities.push((i, similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_similarities = similarities.iter().take(top_k).cloned().collect();

        // Filter by threshold
        let above_threshold = if let Some(thresh) = threshold {
            similarities.into_iter()
                .filter(|(_, sim)| *sim >= thresh)
                .collect()
        } else {
            similarities.into_iter().take(top_k).collect()
        };

        Ok(SimilarityResult {
            similarities: top_similarities,
            above_threshold,
        })
    }

    /// Cosine similarity helper
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (&x, &y) in a.iter().zip(b.iter()) {
            dot_product += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Prompt engineering utilities for zero-shot classification
pub struct PromptEngineer {
    /// Available prompt templates
    templates: Vec<String>,
    /// Template categories
    categories: HashMap<String, Vec<String>>,
}

impl PromptEngineer {
    /// Create new prompt engineer
    pub fn new() -> Self {
        let mut categories = HashMap::new();

        // Image classification templates
        categories.insert("image_classification".to_string(), vec![
            "a photo of a {}".to_string(),
            "a photograph of a {}".to_string(),
            "an image of a {}".to_string(),
            "a picture of a {}".to_string(),
            "a {} in the image".to_string(),
            "a {} shown in a photo".to_string(),
        ]);

        // Object detection templates
        categories.insert("object_detection".to_string(), vec![
            "there is a {} in the image".to_string(),
            "the image contains a {}".to_string(),
            "a {} is present in the picture".to_string(),
        ]);

        // Scene understanding templates
        categories.insert("scene".to_string(), vec![
            "a photo of a {}".to_string(),
            "this is {} in the image".to_string(),
            "the scene shows {}".to_string(),
        ]);

        Self {
            templates: categories.values().flatten().cloned().collect(),
            categories,
        }
    }

    /// Generate prompts for class names using specified category
    pub fn generate_prompts(&self, class_names: &[String], category: &str) -> Result<Vec<String>> {
        let category_templates = self.categories.get(category).ok_or_else(|| {
            NNError::InvalidInput {
                message: format!("Unknown category: {}", category),
            }
        })?;

        let mut prompts = Vec::new();

        for class_name in class_names {
            for template in category_templates {
                let prompt = template.replace("{}", class_name);
                prompts.push(prompt);
            }
        }

        Ok(prompts)
    }

    /// Generate contextual prompts using surrounding text
    pub fn generate_contextual_prompts(
        &self,
        class_names: &[String],
        context: &str,
    ) -> Vec<String> {
        class_names.iter().map(|class_name| {
            format!("{} showing {}", context, class_name)
        }).collect()
    }

    /// Add custom template
    pub fn add_template(&mut self, template: String, category: Option<String>) {
        self.templates.push(template.clone());

        if let Some(cat) = category {
            self.categories.entry(cat).or_insert_with(Vec::new).push(template);
        }
    }

    /// Get all available templates
    pub fn templates(&self) -> &[String] {
        &self.templates
    }

    /// Get categories
    pub fn categories(&self) -> &HashMap<String, Vec<String>> {
        &self.categories
    }
}

/// CLIP model loader for loading pretrained checkpoints
pub struct ClipModelLoader {
    // Placeholder for model loading functionality
    checkpoint_dir: String,
    supported_configs: Vec<ClipConfig>,
}

impl ClipModelLoader {
    /// Create new model loader
    pub fn new(checkpoint_dir: String) -> Self {
        Self {
            checkpoint_dir,
            supported_configs: vec![
                ClipConfig::vit_b32(),
                ClipConfig::vit_b16(),
                ClipConfig::vit_l14(),
            ],
        }
    }

    /// Load CLIP model from checkpoint
    pub fn load_model<B, S, T>(
        &self,
        model_name: &str,
        backend: B,
    ) -> Result<ClipModel<B, S, T>>
    where
        B: Backend<Data = T> + Clone + Default,
        S: Storage<T> + Clone + StorageFromVec<T> + 'static,
        T: DataType + FloatExt,
    {
        // Find matching config
        let config = self.supported_configs.iter().find(|c| {
            // Simple name matching (would be more sophisticated)
            model_name.contains("B32") && c.vision_config.patch_size == 32 ||
            model_name.contains("B16") && c.vision_config.patch_size == 16 ||
            model_name.contains("L14") && c.vision_config.patch_size == 14
        }).cloned().unwrap_or_else(|| ClipConfig::vit_b32());

        println!("Loading CLIP model: {} from {}", model_name, self.checkpoint_dir);
        println!("Using config: {:?}", config);

        // Create model (in practice, would load weights from checkpoint)
        // For now, return initialized model
        ClipModel::new(config)
    }

    /// List available pretrained models
    pub fn available_models(&self) -> Vec<String> {
        vec![
            "CLIP-ViT-B32".to_string(),
            "CLIP-ViT-B16".to_string(),
            "CLIP-ViT-L14".to_string(),
        ]
    }

    /// Get supported configurations
    pub fn supported_configs(&self) -> &[ClipConfig] {
        &self.supported_configs
    }
}

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Classification Result:")?;
        for (i, (idx, prob)) in self.top_k.iter().enumerate() {
            if i < self.labels.len() {
                writeln!(f, "  {}: {:.2}% ({})", i + 1, prob * 100.0, self.labels[*idx])?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for RetrievalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Retrieval Result:")?;
        writeln!(f, "Image→Text matches: {}", self.image_to_text.len())?;
        writeln!(f, "Text→Image matches: {}", self.text_to_image.len())?;
        Ok(())
    }
}

impl PromptEngineer {
    /// Create specialized prompt engineer for specific domains
    pub fn for_domain(domain: &str) -> Self {
        let mut engineer = Self::new();

        match domain {
            "medical" => {
                engineer.add_template(
                    "medical image showing {}".to_string(),
                    Some("medical".to_string())
                );
                engineer.add_template(
                    "radiology scan of {}".to_string(),
                    Some("medical".to_string())
                );
            },
            "food" => {
                engineer.add_template(
                    "food image of {}".to_string(),
                    Some("food".to_string())
                );
                engineer.add_template(
                    "dish showing {}".to_string(),
                    Some("food".to_string())
                );
            },
            "nature" => {
                engineer.add_template(
                    "nature photo of {}".to_string(),
                    Some("nature".to_string())
                );
                engineer.add_template(
                    "wildlife image of {}".to_string(),
                    Some("nature".to_string())
                );
            },
            _ => {} // Use defaults
        }

        engineer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::dtype::float::Float32;
    use crate::storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;

    #[test]
    fn test_clip_model_loading() {
        let loader = ClipModelLoader::new("checkpoints".to_string());
        let models = loader.available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"CLIP-ViT-B32".to_string()));
    }

    #[test]
    fn test_prompt_engineer_basic() {
        let engineer = PromptEngineer::new();
        let prompts = engineer.generate_prompts(
            &["cat".to_string(), "dog".to_string()],
            "image_classification"
        ).unwrap();

        assert!(!prompts.is_empty());
        assert!(prompts.iter().any(|p| p.contains("photo of a cat")));
        assert!(prompts.iter().any(|p| p.contains("photo of a dog")));
    }

    #[test]
    fn test_prompt_engineer_domain() {
        let engineer = PromptEngineer::for_domain("medical");
        assert!(engineer.categories().contains_key("medical"));

        let medical_templates = engineer.categories().get("medical").unwrap();
        assert!(medical_templates.iter().any(|t| t.contains("medical image")));
    }

    #[test]
    fn test_similarity_search() {
        let mut classifier = ClipClassifier::<TestBackend, TestStorage, Float32> {
            model: ClipModel::new(ClipConfig::vit_b32()).unwrap(),
            image_processor: ImageProcessor::default(),
            text_processor: TextProcessor::default(),
            templates: vec!["a photo of {}".to_string()],
        };

        // Test similarity search
        let inference = ClipInference::new(classifier.model);
        let query_emb = vec![0.5; 512]; // Mock embedding
        let candidate_embs = vec![0.6; 512 * 3]; // 3 candidates

        let result = inference.similarity_search(&query_emb, &candidate_embs, 2, Some(0.0)).unwrap();
        assert_eq!(result.similarities.len(), 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let inference = ClipInference::<TestBackend, TestStorage, Float32> {
            model: ClipModel::new(ClipConfig::vit_b32()).unwrap(),
            image_processor: ImageProcessor::default(),
        };

        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let sim = inference.cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);

        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let sim = inference.cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);
    }
}
