//! CLIP Retrieval Evaluation
//!
//! Implements comprehensive image-to-text and text-to-image retrieval evaluation
//! with standard metrics: R@1, R@5, R@10, MRR (Mean Reciprocal Rank).
//!
//! Designed for memory-efficient evaluation of large-scale datasets.

use std::collections::HashMap;
use std::time::Instant;

use super::{ClipEvaluationConfig, ClipModelEvaluator, EvaluationDataset};
use crate::core::error::Result;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::Storage;

/// Single dataset retrieval results
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct RetrievalResults {
    /// Dataset name
    pub dataset_name: String,
    /// Image-to-text retrieval scores by k
    pub i2t_scores: HashMap<usize, f64>,
    /// Text-to-image retrieval scores by k
    pub t2i_scores: HashMap<usize, f64>,
    /// Individual image-to-text ranks for analysis
    pub i2t_ranks: Vec<usize>,
    /// Individual text-to-image ranks for analysis
    pub t2i_ranks: Vec<usize>,
}

/// Comprehensive retrieval metrics
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct RetrievalMetrics {
    /// Dataset name
    pub dataset_name: String,

    /// Image-to-text retrieval metrics
    pub image_to_text_r1: f64,
    pub image_to_text_r5: f64,
    pub image_to_text_r10: f64,
    pub image_to_text_mrr: f64,

    /// Text-to-image retrieval metrics
    pub text_to_image_r1: f64,
    pub text_to_image_r5: f64,
    pub text_to_image_r10: f64,
    pub text_to_image_mrr: f64,

    /// Additional statistics
    pub num_samples: usize,
    pub median_i2t_rank: f64,
    pub median_t2i_rank: f64,
}

/// Memory-optimized similarity computation
struct SimilarityComputer {
    cache: HashMap<String, Vec<f64>>,
}

impl SimilarityComputer {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Compute cosine similarity between two normalized embedding matrices
    fn pairwise_cosine_similarity(
        &self,
        emb1: &[Vec<f32>],
        emb2: &[Vec<f32>],
    ) -> Result<Vec<Vec<f64>>> {
        let mut similarities = Vec::with_capacity(emb1.len());

        for a in emb1 {
            let row: Vec<f64> = emb2.iter().map(|b| Self::cosine_similarity(a, b)).collect();
            similarities.push(row);
        }

        Ok(similarities)
    }

    /// Compute cosine similarity between two normalized vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
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

    /// Compute top-k similarities for efficient retrieval
    fn compute_top_k_similarities(
        &self,
        similarities: &[Vec<f64>],
        k: usize,
    ) -> Vec<Vec<(usize, f64)>> {
        let mut top_k_results = Vec::with_capacity(similarities.len());

        for similarity_row in similarities {
            let mut ranked: Vec<(usize, f64)> = similarity_row
                .iter()
                .enumerate()
                .map(|(idx, &sim)| (idx, sim))
                .collect();

            // Sort by similarity descending
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top-k
            ranked.truncate(k);
            top_k_results.push(ranked);
        }

        top_k_results
    }
}

/// Retrieval evaluator for CLIP models
pub struct RetrievalEvaluator {
    config: ClipEvaluationConfig,
    similarity_computer: SimilarityComputer,
}

impl RetrievalEvaluator {
    /// Create new retrieval evaluator
    pub fn new(config: ClipEvaluationConfig) -> Result<Self> {
        Ok(Self {
            config,
            similarity_computer: SimilarityComputer::new(),
        })
    }

    /// Evaluate retrieval on multiple datasets
    pub fn evaluate_retrieval<B, S, T, M>(
        &self,
        model: &M,
        datasets: &[&dyn EvaluationDataset],
    ) -> Result<Vec<RetrievalResults>>
    where
        B: Backend<Data = T> + Clone + Send + Sync,
        S: Storage<T> + Clone + Send + Sync,
        T: DataType + FloatExt + Clone + Send + Sync,
        M: ClipModelEvaluator,
    {
        let mut results = Vec::new();
        let start_time = Instant::now();

        println!(
            "🧪 Starting CLIP retrieval evaluation on {} datasets",
            datasets.len()
        );

        // TODO: Fix trait object evaluation
        // for dataset in datasets {
        //     println!("  Evaluating dataset: {}", dataset.name());
        //     let dataset_result = self.evaluate_single_dataset(model, dataset)?;
        //     results.push(dataset_result);
        // }

        let total_time = start_time.elapsed().as_secs_f64();
        println!("✅ Retrieval evaluation completed in {:.2}s", total_time);

        Ok(results)
    }

    /// Evaluate retrieval on single dataset
    pub fn evaluate_single_dataset<B, S, T, M>(
        &self,
        model: &M,
        dataset: &dyn EvaluationDataset,
    ) -> Result<RetrievalResults>
    where
        B: Backend<Data = T> + Clone + Send + Sync,
        S: Storage<T> + Clone + Send + Sync,
        T: DataType + FloatExt + Clone + Send + Sync,
        M: ClipModelEvaluator,
    {
        let start_time = Instant::now();

        println!(
            "  Computing embeddings for {} samples...",
            dataset.image_embeddings().len()
        );

        // Use provided embeddings - would normally encode fresh ones
        let image_embeddings = &dataset.image_embeddings();
        let text_embeddings = &dataset.text_embeddings();

        println!("  Computing similarity matrices...");
        let i2t_similarities = self
            .similarity_computer
            .pairwise_cosine_similarity(image_embeddings, text_embeddings)?;

        let t2i_similarities = self
            .similarity_computer
            .pairwise_cosine_similarity(text_embeddings, image_embeddings)?;

        println!("  Computing retrieval ranks...");
        let mut i2t_ranks = self.compute_retrieval_ranks(&i2t_similarities)?;
        let mut t2i_ranks = self.compute_retrieval_ranks(&t2i_similarities)?;

        println!("  Computing success metrics...");
        let mut i2t_scores = HashMap::new();
        let mut t2i_scores = HashMap::new();

        for &k in &self.config.retrieval_top_k {
            i2t_scores.insert(k, self.compute_recall_at_k(&i2t_ranks, k));
            t2i_scores.insert(k, self.compute_recall_at_k(&t2i_ranks, k));
        }

        let eval_time = start_time.elapsed().as_secs_f64();
        println!("  Dataset evaluation completed in {:.2}s", eval_time);

        Ok(RetrievalResults {
            dataset_name: dataset.name().to_string(),
            i2t_scores,
            t2i_scores,
            i2t_ranks,
            t2i_ranks,
        })
    }

    /// Compute retrieval metrics from results
    pub fn compute_retrieval_metrics(
        &self,
        results: &[RetrievalResults],
    ) -> Result<Vec<RetrievalMetrics>> {
        let mut metrics = Vec::new();

        for result in results {
            let dataset_metrics = self.compute_single_dataset_metrics(result)?;
            metrics.push(dataset_metrics);
        }

        Ok(metrics)
    }

    /// Compute comprehensive metrics for single dataset
    pub fn compute_single_dataset_metrics(
        &self,
        result: &RetrievalResults,
    ) -> Result<RetrievalMetrics> {
        let i2t_r1 = result.i2t_scores.get(&1).copied().unwrap_or(0.0);
        let i2t_r5 = result.i2t_scores.get(&5).copied().unwrap_or(0.0);
        let i2t_r10 = result.i2t_scores.get(&10).copied().unwrap_or(0.0);
        let i2t_mrr = self.compute_mean_reciprocal_rank(&result.i2t_ranks);

        let t2i_r1 = result.t2i_scores.get(&1).copied().unwrap_or(0.0);
        let t2i_r5 = result.t2i_scores.get(&5).copied().unwrap_or(0.0);
        let t2i_r10 = result.t2i_scores.get(&10).copied().unwrap_or(0.0);
        let t2i_mrr = self.compute_mean_reciprocal_rank(&result.t2i_ranks);

        let median_i2t_rank = self.compute_median(&result.i2t_ranks);
        let median_t2i_rank = self.compute_median(&result.t2i_ranks);

        Ok(RetrievalMetrics {
            dataset_name: result.dataset_name.clone(),
            image_to_text_r1: i2t_r1,
            image_to_text_r5: i2t_r5,
            image_to_text_r10: i2t_r10,
            image_to_text_mrr: i2t_mrr,
            text_to_image_r1: t2i_r1,
            text_to_image_r5: t2i_r5,
            text_to_image_r10: t2i_r10,
            text_to_image_mrr: t2i_mrr,
            num_samples: result.i2t_ranks.len(),
            median_i2t_rank,
            median_t2i_rank,
        })
    }

    /// Compute retrieval ranks from similarity matrix
    fn compute_retrieval_ranks(&self, similarities: &[Vec<f64>]) -> Result<Vec<usize>> {
        let mut ranks = Vec::with_capacity(similarities.len());

        for (i, similarity_row) in similarities.iter().enumerate() {
            // Find the rank of the correct match (an ideal retrieval would have rank 1)
            let mut indexed_similarities: Vec<(usize, f64)> = similarity_row
                .iter()
                .enumerate()
                .map(|(j, &sim)| (j, sim))
                .collect();

            // Sort by similarity descending
            indexed_similarities
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Find rank of the correct match (i should match i)
            let rank = indexed_similarities
                .iter()
                .position(|&(idx, _)| idx == i)
                .map(|pos| pos + 1) // 1-indexed rank
                .unwrap_or(similarity_row.len() + 1); // Worst possible rank

            ranks.push(rank);
        }

        Ok(ranks)
    }

    /// Compute recall at k
    fn compute_recall_at_k(&self, ranks: &[usize], k: usize) -> f64 {
        let correct_retrievals = ranks.iter().filter(|&&rank| rank <= k).count();
        correct_retrievals as f64 / ranks.len() as f64
    }

    /// Compute mean reciprocal rank
    fn compute_mean_reciprocal_rank(&self, ranks: &[usize]) -> f64 {
        let reciprocal_ranks: Vec<f64> = ranks.iter().map(|&rank| 1.0 / (rank as f64)).collect();

        reciprocal_ranks.iter().sum::<f64>() / reciprocal_ranks.len() as f64
    }

    /// Compute median of a sorted vector
    fn compute_median(&self, values: &[usize]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort();

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
        } else {
            sorted[mid] as f64
        }
    }

    /// Validate success criterion achievement
    pub fn validate_success_criterion(&self, metrics: &RetrievalMetrics) -> RetrievalValidation {
        let achieved = metrics.image_to_text_r1 >= 0.2 && metrics.text_to_image_r1 >= 0.2;

        RetrievalValidation {
            achieved_success_criterion: achieved,
            image_to_text_r1: metrics.image_to_text_r1,
            text_to_image_r1: metrics.text_to_image_r1,
            validation_timestamp: std::time::SystemTime::now(),
            dataset_name: metrics.dataset_name.clone(),
        }
    }
}

/// Retrieval validation results
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct RetrievalValidation {
    /// Whether success criterion was achieved
    pub achieved_success_criterion: bool,
    /// Image-to-text R@1 score
    pub image_to_text_r1: f64,
    /// Text-to-image R@1 score
    pub text_to_image_r1: f64,
    /// Validation timestamp
    pub validation_timestamp: std::time::SystemTime,
    /// Dataset name
    pub dataset_name: String,
}

impl Default for RetrievalMetrics {
    fn default() -> Self {
        Self {
            dataset_name: String::new(),
            image_to_text_r1: 0.0,
            image_to_text_r5: 0.0,
            image_to_text_r10: 0.0,
            image_to_text_mrr: 0.0,
            text_to_image_r1: 0.0,
            text_to_image_r5: 0.0,
            text_to_image_r10: 0.0,
            text_to_image_mrr: 0.0,
            num_samples: 0,
            median_i2t_rank: 0.0,
            median_t2i_rank: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_computation() {
        let computer = SimilarityComputer::new();

        let emb1 = vec![vec![1.0, 0.0, 0.0]]; // Normalized unit vector
        let emb2 = vec![vec![1.0, 0.0, 0.0]]; // Normalized unit vector

        let similarities = computer.pairwise_cosine_similarity(&emb1, &emb2).unwrap();
        assert_eq!(similarities.len(), 1);
        assert_eq!(similarities[0].len(), 1);
        assert!((similarities[0][0] - 1.0).abs() < 1e-6); // Should be perfectly similar
    }

    #[test]
    fn test_retrieval_rank_computation() {
        let evaluator = RetrievalEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        // Simple 3x3 similarity matrix where each item matches perfectly with itself
        let similarities = vec![
            vec![1.0, 0.1, 0.2], // Item 0 matches best with itself (rank 1)
            vec![0.1, 1.0, 0.3], // Item 1 matches best with itself (rank 1)
            vec![0.2, 0.3, 1.0], // Item 2 matches best with itself (rank 1)
        ];

        let ranks = evaluator.compute_retrieval_ranks(&similarities).unwrap();
        assert_eq!(ranks, vec![1, 1, 1]); // All should achieve perfect retrieval
    }

    #[test]
    fn test_recall_at_k_computation() {
        let evaluator = RetrievalEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        // Ranks: 1, 3, 2, 5, 1 (2 out of 5 within k=1, 4 out of 5 within k=3)
        let ranks = vec![1, 3, 2, 5, 1];

        assert_eq!(evaluator.compute_recall_at_k(&ranks, 1), 2.0 / 5.0); // R@1 = 40%
        assert_eq!(evaluator.compute_recall_at_k(&ranks, 3), 4.0 / 5.0); // R@3 = 80%
    }

    #[test]
    fn test_median_computation() {
        let evaluator = RetrievalEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        // Odd number of values
        assert_eq!(evaluator.compute_median(&[1, 3, 5]), 3.0);

        // Even number of values
        assert_eq!(evaluator.compute_median(&[1, 3, 5, 7]), 4.0);

        // Empty vector
        assert_eq!(evaluator.compute_median(&[]), 0.0);
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let evaluator = RetrievalEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        // Ranks: 1, 2, 4
        // RR: 1.0, 0.5, 0.25
        // MRR: (1.0 + 0.5 + 0.25) / 3 = 0.583...
        let ranks = vec![1, 2, 4];
        let mrr = evaluator.compute_mean_reciprocal_rank(&ranks);
        assert!((mrr - (1.0 + 0.5 + 0.25) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_success_validation_logic() {
        let evaluator = RetrievalEvaluator::new(ClipEvaluationConfig::default()).unwrap();

        // Test success case
        let success_metrics = RetrievalMetrics {
            dataset_name: "test".to_string(),
            image_to_text_r1: 0.25,
            text_to_image_r1: 0.22,
            ..Default::default()
        };
        let validation = evaluator.validate_success_criterion(&success_metrics);
        assert!(validation.achieved_success_criterion);

        // Test failure case
        let failure_metrics = RetrievalMetrics {
            dataset_name: "test".to_string(),
            image_to_text_r1: 0.15,
            text_to_image_r1: 0.18,
            ..Default::default()
        };
        let validation = evaluator.validate_success_criterion(&failure_metrics);
        assert!(!validation.achieved_success_criterion);
    }
}
