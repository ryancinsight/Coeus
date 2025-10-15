//! Pretrained model implementations and configurations

use crate::registry::{ModelEntry, ModelMetadata, Task};
use std::collections::HashMap;

/// Register built-in models in the registry
pub fn register_builtin_models(registry: &mut crate::ModelRegistry) -> crate::Result<()> {
    // ResNet-50 for classification
    let resnet50 = create_resnet_entry("resnet50", 50);
    registry.register_model(resnet50)?;

    // ResNet-101 for classification
    let resnet101 = create_resnet_entry("resnet101", 101);
    registry.register_model(resnet101)?;

    // BERT Base for NLP tasks
    let bert_base = create_bert_entry("bert-base-uncased", false);
    registry.register_model(bert_base)?;

    // BERT Large for NLP tasks
    let bert_large = create_bert_entry("bert-large-uncased", true);
    registry.register_model(bert_large)?;

    // ViT Base for vision tasks
    let vit_base = create_vit_entry("vit-base-patch16-224", false);
    registry.register_model(vit_base)?;

    // GPT-2 Small for generation
    let gpt2_small = create_gpt_entry("gpt2", "small");
    registry.register_model(gpt2_small)?;

    Ok(())
}

/// Create ResNet model entry
fn create_resnet_entry(name: &str, layers: usize) -> ModelEntry {
    let parameters = match layers {
        50 => 25_557_032,
        101 => 44_549_160,
        _ => 25_557_032,
    };

    ModelEntry {
        id: name.to_string(),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        architecture: format!("ResNet-{}", layers),
        task: Task::Classification,
        metrics: HashMap::from([
            ("top1_accuracy".to_string(), 76.15), // Example ImageNet accuracy
            ("top5_accuracy".to_string(), 92.87),
        ]),
        metadata: ModelMetadata {
            description: format!("ResNet-{} model trained on ImageNet", layers),
            author: "Kaiming He et al.".to_string(),
            license: "MIT".to_string(),
            parameters,
            input_shape: vec![224, 224, 3],
            output_shape: vec![1000],
            dtype: "f32".to_string(),
            tags: vec![
                "vision".to_string(),
                "classification".to_string(),
                "imagenet".to_string(),
                "residual".to_string(),
            ],
            paper_url: Some("https://arxiv.org/abs/1512.03385".to_string()),
            code_url: Some("https://github.com/pytorch/vision".to_string()),
        },
        download_url: format!("https://example.com/models/{}.safetensors", name),
        checksum: format!("resnet{}_checksum", layers),
        file_size: parameters as u64 * 4, // Rough estimate: 4 bytes per f32 parameter
    }
}

/// Create BERT model entry
fn create_bert_entry(name: &str, is_large: bool) -> ModelEntry {
    let (parameters, hidden_size) = if is_large {
        (340_000_000, 1024)
    } else {
        (110_000_000, 768)
    };

    let task = if name.contains("classification") {
        Task::Classification
    } else {
        Task::Embedding
    };

    ModelEntry {
        id: name.to_string(),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        architecture: if is_large {
            "BERT-Large".to_string()
        } else {
            "BERT-Base".to_string()
        },
        task,
        metrics: HashMap::from([
            ("glue_score".to_string(), 82.0), // Example GLUE score
        ]),
        metadata: ModelMetadata {
            description: format!(
                "{} model for natural language understanding",
                if is_large { "BERT-Large" } else { "BERT-Base" }
            ),
            author: "Jacob Devlin et al.".to_string(),
            license: "Apache-2.0".to_string(),
            parameters,
            input_shape: vec![512], // Max sequence length
            output_shape: vec![hidden_size],
            dtype: "f32".to_string(),
            tags: vec![
                "nlp".to_string(),
                "transformer".to_string(),
                "bert".to_string(),
                "language".to_string(),
            ],
            paper_url: Some("https://arxiv.org/abs/1810.04805".to_string()),
            code_url: Some("https://github.com/google-research/bert".to_string()),
        },
        download_url: format!("https://example.com/models/{}.safetensors", name),
        checksum: format!("bert_{}_checksum", if is_large { "large" } else { "base" }),
        file_size: parameters as u64 * 4,
    }
}

/// Create Vision Transformer model entry
fn create_vit_entry(name: &str, _is_large: bool) -> ModelEntry {
    let parameters = 86_000_000; // ViT-Base

    ModelEntry {
        id: name.to_string(),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        architecture: "Vision Transformer".to_string(),
        task: Task::Classification,
        metrics: HashMap::from([
            ("top1_accuracy".to_string(), 81.07), // Example ImageNet accuracy
        ]),
        metadata: ModelMetadata {
            description: "Vision Transformer model for image classification".to_string(),
            author: "Alexey Dosovitskiy et al.".to_string(),
            license: "Apache-2.0".to_string(),
            parameters,
            input_shape: vec![224, 224, 3],
            output_shape: vec![1000],
            dtype: "f32".to_string(),
            tags: vec![
                "vision".to_string(),
                "transformer".to_string(),
                "vit".to_string(),
                "attention".to_string(),
            ],
            paper_url: Some("https://arxiv.org/abs/2010.11929".to_string()),
            code_url: Some("https://github.com/google-research/vision_transformer".to_string()),
        },
        download_url: format!("https://example.com/models/{}.safetensors", name),
        checksum: "vit_base_checksum".to_string(),
        file_size: parameters as u64 * 4,
    }
}

/// Create GPT model entry
fn create_gpt_entry(name: &str, size: &str) -> ModelEntry {
    let (parameters, vocab_size) = match size {
        "small" => (117_000_000, 50257),
        "medium" => (345_000_000, 50257),
        "large" => (774_000_000, 50257),
        _ => (117_000_000, 50257),
    };

    ModelEntry {
        id: name.to_string(),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        architecture: format!("GPT-2 {}", size),
        task: Task::Generation,
        metrics: HashMap::from([
            ("perplexity".to_string(), 18.5), // Example perplexity
        ]),
        metadata: ModelMetadata {
            description: format!("GPT-2 {} model for text generation", size),
            author: "OpenAI".to_string(),
            license: "MIT".to_string(),
            parameters,
            input_shape: vec![1024], // Max sequence length
            output_shape: vec![vocab_size],
            dtype: "f32".to_string(),
            tags: vec![
                "nlp".to_string(),
                "transformer".to_string(),
                "gpt".to_string(),
                "generation".to_string(),
            ],
            paper_url: Some(
                "https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf"
                    .to_string(),
            ),
            code_url: Some("https://github.com/openai/gpt-2".to_string()),
        },
        download_url: format!("https://example.com/models/{}.safetensors", name),
        checksum: format!("gpt2_{}_checksum", size),
        file_size: parameters as u64 * 4,
    }
}

/// Get a list of all built-in model names
pub fn builtin_model_names() -> Vec<&'static str> {
    vec![
        "resnet50",
        "resnet101",
        "bert-base-uncased",
        "bert-large-uncased",
        "vit-base-patch16-224",
        "gpt2",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_model_registration() {
        let mut registry = crate::ModelRegistry::new();
        register_builtin_models(&mut registry).unwrap();

        assert!(registry.get_model("resnet50").is_some());
        assert!(registry.get_model("bert-base-uncased").is_some());
        assert!(registry.get_model("vit-base-patch16-224").is_some());
        assert!(registry.get_model("gpt2").is_some());
    }

    #[test]
    fn test_resnet_metadata() {
        let entry = create_resnet_entry("resnet50", 50);
        assert_eq!(entry.metadata.parameters, 25_557_032);
        assert_eq!(entry.metadata.input_shape, vec![224, 224, 3]);
        assert_eq!(entry.metadata.output_shape, vec![1000]);
        assert!(entry.metadata.tags.contains(&"residual".to_string()));
    }

    #[test]
    fn test_bert_metadata() {
        let entry = create_bert_entry("bert-base-uncased", false);
        assert_eq!(entry.metadata.parameters, 110_000_000);
        assert_eq!(entry.metadata.input_shape, vec![512]);
        assert!(entry.metadata.tags.contains(&"transformer".to_string()));
    }

    #[test]
    fn test_builtin_model_names() {
        let names = builtin_model_names();
        assert!(names.contains(&"resnet50"));
        assert!(names.contains(&"gpt2"));
        assert_eq!(names.len(), 6);
    }
}
