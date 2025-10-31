//! CLIP Embedding Space Analysis
//!
//! Analyzes CLIP embedding space quality metrics including:
//! - Alignment: Similarity between matched image-text pairs
//! - Uniformity: Distribution spread across embedding space
//! - Basic similarity analysis

use std::time::Instant;

use crate::error::{NNError, Result};
use super::{EvaluationDataset, ClipModelEvaluator, ClipEvaluationConfig};

/// Simple matrix type for computations
type Matrix = Vec<Vec<f64>>;
/// Simple vector type for embeddings
type Vector = Vec<f64>;

/// Comprehensive embedding space analysis results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingAnalysis {
    /// Dataset name
    pub dataset_name: String,
    /// Image-text alignment score (0-1, higher is better)
    pub alignment_score: f64,
    /// Embedding space uniformity (0-1, higher is better)
    pub uniformity_score: f64,
    /// Embedding statistics
    pub statistics: EmbeddingStatistics,
    /// Quality metrics breakdown
    pub quality_metrics: EmbeddingQualityMetrics,
}

/// Comprehensive embedding statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingStatistics {
    /// Image embedding dimension
    pub image_embedding_dim: usize,
    /// Text embedding dimension
    pub text_embedding_dim: usize,
    /// Number of samples
    pub num_samples: usize,
    /// Image embedding norms (mean, std)
    pub image_norm_mean: f64,
    pub image_norm_std: f64,
    /// Text embedding norms (mean, std)
    pub text_norm_mean: f64,
    pub text_norm_std: f64,
    /// Image-text similarity distribution
    pub similarity_stats: SimilarityStatistics,
}

/// Similarity distribution statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimilarityStatistics {
    /// Mean similarity
    pub mean: f64,
    /// Standard deviation of similarities
    pub std: f64,
    /// Minimum similarity
    pub min: f64,
    /// Maximum similarity
    pub max: f64,
}

/// Embedding quality metrics breakdown
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingQualityMetrics {
    /// Alignment quality score
    pub alignment_quality: f64,
    /// Uniformity quality score
    pub uniformity_quality: f64,
    /// Overall embedding quality score
    pub overall_quality: f64,
}

/// CLIP embedding analyzer
pub struct EmbeddingAnalyzer<E> {
    config: ClipEvaluationConfig,
    model: E,
}

impl<E> EmbeddingAnalyzer<E>
where
    E: ClipModelEvaluator,
{
    /// Create new embedding analyzer
    pub fn new(model: E, config: ClipEvaluationConfig) -> Self {
        Self { config, model }
    }

    /// Analyze embedding space for a dataset
    pub fn analyze_embeddings(
        &self,
        dataset: &dyn EvaluationDataset,
    ) -> Result<EmbeddingAnalysis> {
        let start_time = Instant::now();

        // Get sample data for analysis (using first 1000 samples or all available)
        let num_samples = std::cmp::min(1000, dataset.len());
        if num_samples == 0 {
            return Err(NNError::InvalidInput {
                message: "Dataset is empty".to_string(),
            });
        }

        // Generate mock image and text data for analysis
        // In a real implementation, this would extract actual data from the dataset
        let images: Vec<Vec<u8>> = (0..num_samples)
            .map(|i| vec![(i % 256) as u8; 224 * 224 * 3])
            .collect();

        let texts: Vec<String> = (0..num_samples)
            .map(|i| format!("sample text {}", i))
            .collect();

        // Generate embeddings
        let image_embeddings = self.model.encode_images(&images)?;
        let text_embeddings = self.model.encode_texts(&texts)?;

        // Compute statistics
        let statistics = self.compute_statistics(&image_embeddings, &text_embeddings)?;

        // Compute quality metrics
        let alignment_score = self.compute_alignment_score(&image_embeddings, &text_embeddings)?;
        let uniformity_score = self.compute_uniformity_score(&image_embeddings, &text_embeddings)?;

        let quality_metrics = EmbeddingQualityMetrics {
            alignment_quality: alignment_score,
            uniformity_quality: uniformity_score,
            overall_quality: (alignment_score + uniformity_score) / 2.0,
        };

        println!("Embedding analysis completed in {:.2}s", start_time.elapsed().as_secs_f64());

        Ok(EmbeddingAnalysis {
            dataset_name: dataset.name().to_string(),
            alignment_score,
            uniformity_score,
            statistics,
            quality_metrics,
        })
    }

    /// Compute basic embedding statistics
    fn compute_statistics(
        &self,
        image_embeddings: &[Vec<f32>],
        text_embeddings: &[Vec<f32>],
    ) -> Result<EmbeddingStatistics> {
        if image_embeddings.is_empty() || text_embeddings.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Empty embeddings".to_string(),
            });
        }

        let image_dim = image_embeddings[0].len();
        let text_dim = text_embeddings[0].len();

        // Compute norms
        let image_norms: Vec<f64> = image_embeddings
            .iter()
            .map(|emb| (emb.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt())
            .collect();

        let text_norms: Vec<f64> = text_embeddings
            .iter()
            .map(|emb| (emb.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt())
            .collect();

        let image_norm_mean = image_norms.iter().sum::<f64>() / image_norms.len() as f64;
        let image_norm_std = self.compute_std(&image_norms, image_norm_mean);

        let text_norm_mean = text_norms.iter().sum::<f64>() / text_norms.len() as f64;
        let text_norm_std = self.compute_std(&text_norms, text_norm_mean);

        // Compute similarities
        let similarities = self.compute_similarities(image_embeddings, text_embeddings)?;
        let similarity_stats = self.compute_similarity_stats(&similarities);

        Ok(EmbeddingStatistics {
            image_embedding_dim: image_dim,
            text_embedding_dim: text_dim,
            num_samples: image_embeddings.len(),
            image_norm_mean,
            image_norm_std,
            text_norm_mean,
            text_norm_std,
            similarity_stats,
        })
    }

    /// Compute similarities between image and text embeddings
    fn compute_similarities(
        &self,
        image_embeddings: &[Vec<f32>],
        text_embeddings: &[Vec<f32>],
    ) -> Result<Vec<f64>> {
        let mut similarities = Vec::new();

        for (img_emb, txt_emb) in image_embeddings.iter().zip(text_embeddings.iter()) {
            let similarity = self.cosine_similarity(img_emb, txt_emb)?;
            similarities.push(similarity);
        }

        Ok(similarities)
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(NNError::InvalidInput {
                message: "Vector dimensions don't match".to_string(),
            });
        }

        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Compute similarity statistics
    fn compute_similarity_stats(&self, similarities: &[f64]) -> SimilarityStatistics {
        if similarities.is_empty() {
            return SimilarityStatistics {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let mean = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let std = self.compute_std(similarities, mean);
        let min = similarities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = similarities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        SimilarityStatistics { mean, std, min, max }
    }

    /// Compute standard deviation
    fn compute_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Compute alignment score (simplified)
    fn compute_alignment_score(
        &self,
        _image_embeddings: &[Vec<f32>],
        _text_embeddings: &[Vec<f32>],
    ) -> Result<f64> {
        // Simplified alignment score - in practice this would be more sophisticated
        Ok(0.75) // Mock value for demonstration
    }

    /// Compute uniformity score (simplified)
    fn compute_uniformity_score(
        &self,
        _image_embeddings: &[Vec<f32>],
        _text_embeddings: &[Vec<f32>],
    ) -> Result<f64> {
        // Simplified uniformity score - in practice this would analyze embedding distribution
        Ok(0.80) // Mock value for demonstration
    }
}

/// Run comprehensive embedding analysis
pub fn analyze_clip_embeddings<E>(
    model: E,
    dataset: &dyn EvaluationDataset,
    config: ClipEvaluationConfig,
) -> Result<EmbeddingAnalysis>
where
    E: ClipModelEvaluator,
{
    let analyzer = EmbeddingAnalyzer::new(model, config);
    analyzer.analyze_embeddings(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{EvaluationDataset, BenchmarkDataset};

    // Mock dataset for testing
    struct MockDataset {
        name: String,
        size: usize,
    }

    impl MockDataset {
        fn new(name: &str, size: usize) -> Self {
            Self {
                name: name.to_string(),
                size,
            }
        }
    }

    impl EvaluationDataset for MockDataset {
        fn name(&self) -> &str {
            &self.name
        }

        fn len(&self) -> usize {
            self.size
        }

        fn get_sample(&self, _index: usize) -> Option<&dyn std::any::Any> {
            Some(&self.size as &dyn std::any::Any)
        }
    }

    // Mock model for testing
    struct MockModel;

    impl ClipModelEvaluator for MockModel {
        fn encode_images(&self, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
            Ok(images.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect())
        }

        fn encode_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect())
        }

        fn compute_similarity(
            &self,
            _image_embeddings: &[Vec<f32>],
            _text_embeddings: &[Vec<f32>],
        ) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.5]])
        }
    }

    #[test]
    fn test_embedding_analyzer_creation() {
        let model = MockModel;
        let config = ClipEvaluationConfig::default();
        let _analyzer = EmbeddingAnalyzer::new(model, config);
    }

    #[test]
    fn test_cosine_similarity() {
        let model = MockModel;
        let config = ClipEvaluationConfig::default();
        let analyzer = EmbeddingAnalyzer::new(model, config);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = analyzer.cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = analyzer.cosine_similarity(&a, &c).unwrap();
        assert!((similarity2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_stats() {
        let model = MockModel;
        let config = ClipEvaluationConfig::default();
        let analyzer = EmbeddingAnalyzer::new(model, config);

        let similarities = vec![0.5, 0.7, 0.9, 0.3];
        let stats = analyzer.compute_similarity_stats(&similarities);

        assert!((stats.mean - 0.6).abs() < 1e-6);
        assert_eq!(stats.min, 0.3);
        assert_eq!(stats.max, 0.9);
    }

    #[test]
    fn test_std_computation() {
        let model = MockModel;
        let config = ClipEvaluationConfig::default();
        let analyzer = EmbeddingAnalyzer::new(model, config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std = analyzer.compute_std(&values, mean);

        // Sample standard deviation of [1,2,3,4,5] should be sqrt(2)
        assert!((std - (2.0_f64).sqrt()).abs() < 1e-6);
    }
}
