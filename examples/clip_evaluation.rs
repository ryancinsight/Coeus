//! CLIP Model Evaluation Example
//!
//! This example demonstrates comprehensive CLIP model evaluation including:
//! - Retrieval metrics (R@1, R@5, R@10) for text-image similarity
//! - Zero-shot classification on ImageNet
//! - Embedding space quality analysis
//! - Performance benchmarking
//!
//! Run with: cargo run --example clip_evaluation

use std::collections::HashMap;
use std::time::Instant;

// Backend and tensor dependencies
use backend::CpuBackend;
use dtype::float::Float32;
use storage::{DenseStorage, StorageFromVec, StorageToDense};
use tensor::Tensor;

// NN modules
use nn::clip::{
    ClipConfig, ClipModel, ClipValidator, ValidationConfig,
    validation::{EvaluationType, ValidationReport},
    zero_shot::{ZeroShotClassifier, ZeroShotConfig},
};
use nn::datasets::{
    CocoDataset, Flickr30kDataset, VisionLanguageBatchLoader, BatchConfig, DatasetSplit
};
use nn::error::Result;

// Type aliases for clarity
type Backend = CpuBackend<Float32>;
type Storage = DenseStorage<Float32>;
type Model = ClipModel<Backend, Storage, Float32>;

/// Comprehensive CLIP evaluation
async fn run_clip_evaluation() -> Result<()> {
    println!("🧪 CLIP Model Evaluation Suite");
    println!("================================");

    let start_time = Instant::now();

    // Phase 1: Load CLIP Model
    println!("\n📦 Phase 1: Loading CLIP Model");
    println!("-------------------------------");

    let model_config = ClipConfig {
        vision_config: nn::clip::VisionConfig {
            image_size: 224,
            patch_size: 16,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
        },
        text_config: nn::clip::TextConfig {
            vocab_size: 49408,
            max_position_embeddings: 77,
            hidden_size: 512,
            num_layers: 12,
            num_heads: 8,
            mlp_ratio: 4.0,
        },
        projection_dim: 512,
        temperature: 0.07,
    };

    let model = Model::new(model_config)?;
    println!("✅ CLIP model loaded successfully");

    // Phase 2: Load Dataset
    println!("\n📚 Phase 2: Loading Evaluation Dataset");
    println!("---------------------------------------");

    // Try to load COCO first, fall back to synthetic data if not available
    let dataset_result = CocoDataset::new("datasets/coco").await;
    let dataset: Box<dyn nn::datasets::VisionLanguageData> = match dataset_result {
        Ok(dataset) => {
            println!("✅ Loaded COCO dataset with {} pairs", dataset.len());
            Box::new(dataset)
        },
        Err(_) => {
            println!("⚠️  COCO dataset not available, using Flickr30K...");
            match Flickr30kDataset::new("datasets/flickr30k").await {
                Ok(dataset) => {
                    println!("✅ Loaded Flickr30K dataset with {} pairs", dataset.len());
                    Box::new(dataset)
                },
                Err(_) => {
                    println!("⚠️  No real datasets available, evaluation will be limited");
                    println!("💡 To run full evaluation:");
                    println!("   1. Download COCO 2017: https://cocodataset.org/#download");
                    println!("   2. Extract to datasets/coco/ with proper structure");
                    return Ok(());
                }
            }
        }
    };

    // Phase 3: Comprehensive Validation
    println!("\n🔍 Phase 3: Running Comprehensive Validation");
    println!("---------------------------------------------");

    let validator = ClipValidator::new(
        std::sync::Arc::new(model),
        ValidationConfig::default()
    );

    // Run full evaluation suite
    let report = validator.validate(&*dataset, EvaluationType::Full).await?;

    // Display results
    display_validation_report(&report);

    // Phase 4: Zero-Shot Classification Demo
    println!("\n🎯 Phase 4: Zero-Shot Classification Demo");
    println!("------------------------------------------");

    let zero_shot_config = ZeroShotConfig {
        templates: vec![
            "a photo of a {}".to_string(),
            "a picture of a {}".to_string(),
        ],
        use_ensemble: true,
        ..Default::default()
    };

    // Create zero-shot classifier for a subset of classes
    let class_names = ["cat", "dog", "car", "bird", "house"];
    let zero_shot_classifier = ZeroShotClassifier::new(
        std::sync::Arc::new(Model::new(model_config)?),
        &class_names,
        zero_shot_config,
    )?;

    println!("✅ Created zero-shot classifier for {} classes", class_names.len());

    // Demonstrate classification on a few samples
    println!("\n🖼️  Zero-Shot Classification Examples:");
    println!("-------------------------------------");

    let eval_samples = std::cmp::min(dataset.len(), 5);
    for i in 0..eval_samples {
        let pair = dataset.get(i).await?;
        let result = zero_shot_classifier.classify_image(&pair.image_data)?;

        println!("Sample {}: {} (confidence: {:.2}%)",
                i + 1,
                result.predicted_class,
                result.confidence * 100.0);

        println!("  Top-3: {}",
                result.top_k.iter()
                    .take(3)
                    .map(|(class, conf)| format!("{} ({:.1}%)", class, conf * 100.0))
                    .collect::<Vec<_>>()
                    .join(", "));
        println!();
    }

    // Phase 5: Performance Analysis
    println!("⚡ Phase 5: Performance Analysis");
    println!("---------------------------------");

    let total_time = start_time.elapsed();
    println!("Total evaluation time: {:.2}s", total_time.as_secs_f64());
    println!("Evaluation throughput: {:.1} samples/second",
            dataset.len() as f64 / total_time.as_secs_f64());

    // Memory usage estimate
    let memory_mb = estimate_evaluation_memory(dataset.len());
    println!("Estimated memory usage: {:.1} MB", memory_mb);

    println!("\n🎉 CLIP Evaluation Complete!");
    println!("============================");

    Ok(())
}

/// Display comprehensive validation report
fn display_validation_report(report: &ValidationReport) {
    println!("\n📊 VALIDATION REPORT");
    println!("===================");

    if let Some(ref retrieval) = report.retrieval {
        println!("\n🔍 RETRIEVAL PERFORMANCE");
        println!("------------------------");
        println!("Text-to-Image Retrieval:");
        println!("  R@1:  {:.2}%", retrieval.text_to_image.r1 * 100.0);
        println!("  R@5:  {:.2}%", retrieval.text_to_image.r5 * 100.0);
        println!("  R@10: {:.2}%", retrieval.text_to_image.r10 * 100.0);
        println!("  Mean Rank: {:.1}", retrieval.text_to_image.mean_rank);
        println!("  Median Rank: {:.1}", retrieval.text_to_image.median_rank);

        println!("\nImage-to-Text Retrieval:");
        println!("  R@1:  {:.2}%", retrieval.image_to_text.r1 * 100.0);
        println!("  R@5:  {:.2}%", retrieval.image_to_text.r5 * 100.0);
        println!("  R@10: {:.2}%", retrieval.image_to_text.r10 * 100.0);
        println!("  Mean Rank: {:.1}", retrieval.image_to_text.mean_rank);
        println!("  Median Rank: {:.1}", retrieval.image_to_text.median_rank);

        println!("\nOverall Metrics:");
        println!("  Mean Reciprocal Rank: {:.3}", retrieval.mean_reciprocal_rank);
        println!("  Mean Average Precision: {:.3}", retrieval.mean_average_precision);
    }

    if let Some(ref zero_shot) = report.zero_shot {
        println!("\n🎯 ZERO-SHOT CLASSIFICATION");
        println!("---------------------------");
        println!("Top-1 Accuracy: {:.2}%", zero_shot.top1_accuracy * 100.0);
        println!("Top-5 Accuracy: {:.2}%", zero_shot.top5_accuracy * 100.0);

        println!("\nTop 5 Classes by Accuracy:");
        let mut class_accs: Vec<_> = zero_shot.class_accuracies.iter().collect();
        class_accs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (class, acc) in class_accs.iter().take(5) {
            println!("  {}: {:.1}%", class, acc * 100.0);
        }
    }

    if let Some(ref quality) = report.embedding_quality {
        println!("\n🧮 EMBEDDING SPACE QUALITY");
        println!("--------------------------");
        println!("Uniformity: {:.4}", quality.uniformity);
        println!("Alignment:  {:.4}", quality.alignment);
        println!("CKA Score:  {:.4}", quality.cka_score);
        println!("Intra-modal Variance: {:.6}", quality.intra_modal_variance);
        println!("Inter-modal Variance: {:.6}", quality.inter_modal_variance);

        // Quality assessment
        let quality_score = (quality.uniformity + quality.alignment + quality.cka_score) / 3.0;
        println!("Overall Quality Score: {:.3}", quality_score);

        if quality_score > 0.7 {
            println!("✅ High-quality embedding space");
        } else if quality_score > 0.5 {
            println!("⚠️  Moderate embedding quality - may need fine-tuning");
        } else {
            println!("❌ Poor embedding quality - significant training issues");
        }
    }

    println!("\n⏱️  EVALUATION SUMMARY");
    println!("---------------------");
    println!("Validation Time: {:.2}s", report.validation_time);

    if !report.summary.is_empty() {
        println!("\n📈 Key Metrics:");
        for (metric, value) in &report.summary {
            println!("  {}: {:.3}", metric, value);
        }
    }

    // Overall assessment
    let mut overall_score = 0.0;
    let mut components = 0;

    if let Some(ref retrieval) = report.retrieval {
        overall_score += retrieval.text_to_image.r1;
        components += 1;
    }

    if let Some(ref zero_shot) = report.zero_shot {
        overall_score += zero_shot.top1_accuracy;
        components += 1;
    }

    if let Some(ref quality) = report.embedding_quality {
        overall_score += (quality.uniformity + quality.alignment) / 2.0;
        components += 1;
    }

    if components > 0 {
        overall_score /= components as f64;
        println!("\n🏆 OVERALL ASSESSMENT");
        println!("=====================");
        println!("Composite Score: {:.3}", overall_score);

        if overall_score > 0.7 {
            println!("🎉 Excellent CLIP model performance!");
        } else if overall_score > 0.5 {
            println!("👍 Good performance - ready for deployment");
        } else if overall_score > 0.3 {
            println!("⚠️  Moderate performance - may need fine-tuning");
        } else {
            println!("❌ Poor performance - significant training issues");
        }
    }
}

/// Estimate memory usage for evaluation
fn estimate_evaluation_memory(num_samples: usize) -> f64 {
    // Rough estimation:
    // - Images: 224x224x3 bytes per image = ~150KB
    // - Embeddings: 512 floats per embedding = ~2KB
    // - Overhead: ~10KB per sample

    let image_memory = num_samples as f64 * 150.0 * 1024.0; // KB to bytes
    let embedding_memory = num_samples as f64 * 2.0 * 1024.0; // KB
    let overhead_memory = num_samples as f64 * 10.0 * 1024.0; // KB

    (image_memory + embedding_memory + overhead_memory) / (1024.0 * 1024.0) // MB
}

/// Benchmark CLIP inference performance
async fn benchmark_clip_performance(model: &Model, batch_sizes: &[usize]) -> Result<()> {
    println!("\n⚡ CLIP Performance Benchmark");
    println!("=============================");

    for &batch_size in batch_sizes {
        println!("\n📏 Batch Size: {}", batch_size);

        // Create dummy batch
        let dummy_images: Vec<Vec<u8>> = (0..batch_size)
            .map(|_| vec![128u8; 224 * 224 * 3])
            .collect();

        let dummy_texts: Vec<String> = (0..batch_size)
            .map(|i| format!("This is a test caption number {}", i))
            .collect();

        // Benchmark image encoding
        let image_start = Instant::now();
        let _image_embeddings = model.encode_images(&dummy_images)?;
        let image_time = image_start.elapsed();

        // Benchmark text encoding
        let text_start = Instant::now();
        let _text_embeddings = model.encode_texts(&dummy_texts)?;
        let text_time = text_start.elapsed();

        // Benchmark similarity computation (simplified)
        let similarity_start = Instant::now();
        let image_emb = &_image_embeddings[0];
        for text_emb in &_text_embeddings {
            let _similarity = compute_cosine_similarity(image_emb, text_emb)?;
        }
        let similarity_time = similarity_start.elapsed();

        println!("  Image Encoding: {:.2}ms ({:.1} samples/sec)",
                image_time.as_millis(),
                batch_size as f64 * 1000.0 / image_time.as_millis() as f64);

        println!("  Text Encoding:  {:.2}ms ({:.1} samples/sec)",
                text_time.as_millis(),
                batch_size as f64 * 1000.0 / text_time.as_millis() as f64);

        println!("  Similarity:     {:.2}ms ({:.1} pairs/sec)",
                similarity_time.as_millis(),
                batch_size as f64 * 1000.0 / similarity_time.as_millis() as f64);
    }

    Ok(())
}

/// Compute cosine similarity (helper function)
fn compute_cosine_similarity<B, S, T>(
    emb1: &Tensor<B, S, T>,
    emb2: &Tensor<B, S, T>,
) -> Result<f64>
where
    B: crate::backend::Backend<Data = T>,
    S: crate::storage::Storage<T>,
    T: crate::dtype::DataType,
{
    let emb1_data = emb1.as_slice();
    let emb2_data = emb2.as_slice();

    let dot_product: f64 = emb1_data.iter()
        .zip(emb2_data.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();

    let norm1: f64 = emb1_data.iter().map(|&x| (x as f64).powi(2)).sum().sqrt();
    let norm2: f64 = emb2_data.iter().map(|&x| (x as f64).powi(2)).sum().sqrt();

    if norm1 > 0.0 && norm2 > 0.0 {
        Ok(dot_product / (norm1 * norm2))
    } else {
        Ok(0.0)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Run main evaluation
    if let Err(e) = run_clip_evaluation().await {
        eprintln!("❌ Evaluation failed: {}", e);
        std::process::exit(1);
    }

    // Optional: Run performance benchmarks
    println!("\n🚀 Running Performance Benchmarks...");
    let model = Model::new(ClipConfig {
        vision_config: nn::clip::VisionConfig {
            image_size: 224,
            patch_size: 16,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
        },
        text_config: nn::clip::TextConfig {
            vocab_size: 49408,
            max_position_embeddings: 77,
            hidden_size: 512,
            num_layers: 12,
            num_heads: 8,
            mlp_ratio: 4.0,
        },
        projection_dim: 512,
        temperature: 0.07,
    })?;

    let batch_sizes = [1, 4, 16, 32];
    if let Err(e) = benchmark_clip_performance(&model, &batch_sizes).await {
        eprintln!("⚠️  Benchmark failed: {}", e);
    }

    println!("\n✅ CLIP Evaluation Example Complete!");
    Ok(())
}

