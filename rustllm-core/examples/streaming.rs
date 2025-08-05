//! Streaming example for processing large files with RustLLM Core.

use rustllm_core::prelude::*;
use rustllm_core::foundation::error::internal_error;
use rustllm_core::foundation::iterator::IteratorExt;
use rustllm_tokenizer_basic::BasicTokenizer;
use rustllm_tokenizer_bpe::BpeTokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

/// Process a large file in streaming fashion
fn stream_process_file(path: &str) -> Result<()> {
    println!("Streaming File Processing Example");
    println!("================================\n");
    
    println!("Processing file: {}", path);
    
    let file = File::open(path)
        .map_err(|e| internal_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);
    
    let tokenizer = BasicTokenizer::new();
    
    // Statistics
    let mut total_lines = 0;
    let mut total_tokens = 0;
    let mut total_chars = 0;
    
    let start = Instant::now();
    
    // Process file line by line
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .map_err(|e| internal_error(format!("Failed to read line: {}", e)))?;
        
        total_lines += 1;
        total_chars += line.len();
        
        // Tokenize the line
        let tokens: Vec<_> = tokenizer.tokenize_str(&line).collect();
        total_tokens += tokens.len();
        
        // Process every 1000 lines
        if line_num % 1000 == 0 && line_num > 0 {
            println!("Processed {} lines, {} tokens so far...", line_num, total_tokens);
        }
    }
    
    let elapsed = start.elapsed();
    
    println!("\nProcessing complete!");
    println!("Total lines: {}", total_lines);
    println!("Total characters: {}", total_chars);
    println!("Total tokens: {}", total_tokens);
    println!("Time elapsed: {:?}", elapsed);
    println!("Throughput: {:.2} tokens/sec", total_tokens as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

/// Demonstrate sliding window processing
fn sliding_window_analysis() -> Result<()> {
    println!("\n\nSliding Window Analysis");
    println!("=======================\n");
    
    let text = "The quick brown fox jumps over the lazy dog. \
                The dog was really lazy. The fox was very quick and clever.";
    
    let tokenizer = BasicTokenizer::new();
    let tokens: Vec<_> = tokenizer.tokenize_str(text).collect();
    
    println!("Original text: {}", text);
    println!("Total tokens: {}\n", tokens.len());
    
    // Analyze with different window sizes
    for window_size in [3, 5, 7] {
        println!("Window size {}: ", window_size);
        
        let windows: Vec<_> = tokens.iter()
            .cloned()
            .windows(window_size)
            .take(5) // Just show first 5 windows
            .collect();
        
        for (i, window) in windows.iter().enumerate() {
            let window_text: Vec<_> = window.iter()
                .filter_map(|t| t.as_str())
                .collect();
            println!("  Window {}: {:?}", i, window_text);
        }
        println!();
    }
    
    Ok(())
}

/// Demonstrate parallel chunk processing
fn parallel_chunk_processing() -> Result<()> {
    println!("\nParallel Chunk Processing");
    println!("========================\n");
    
    // Simulate a large corpus
    let corpus: Vec<String> = (0..100)
        .map(|i| format!("This is sentence number {}. It contains some text for processing.", i))
        .collect();
    
    let tokenizer = BpeTokenizer::new();
    
    println!("Processing {} sentences in chunks...", corpus.len());
    
    let start = Instant::now();
    
    // Process in chunks
    let chunk_size = 10;
    let mut total_tokens = 0;
    
    for (chunk_idx, chunk) in corpus.chunks(chunk_size).enumerate() {
        // Process each chunk
        let chunk_tokens: Vec<Vec<_>> = chunk.iter()
            .map(|text| tokenizer.tokenize_str(text).collect::<Vec<_>>())
            .collect();
        
        let chunk_token_count: usize = chunk_tokens.iter()
            .map(|tokens| tokens.len())
            .sum();
        
        total_tokens += chunk_token_count;
        
        println!("  Chunk {}: {} sentences, {} tokens", 
                 chunk_idx, chunk.len(), chunk_token_count);
    }
    
    let elapsed = start.elapsed();
    
    println!("\nTotal tokens: {}", total_tokens);
    println!("Processing time: {:?}", elapsed);
    println!("Throughput: {:.2} tokens/sec", total_tokens as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

/// Demonstrate token frequency analysis
fn token_frequency_analysis() -> Result<()> {
    println!("\n\nToken Frequency Analysis");
    println!("=======================\n");
    
    let text = "The cat sat on the mat. The cat was fat. The mat was flat. \
                The fat cat sat on the flat mat.";
    
    let tokenizer = BasicTokenizer::new();
    let tokens: Vec<_> = tokenizer.tokenize_str(text).collect();
    
    // Count token frequencies
    use std::collections::HashMap;
    let mut frequencies: HashMap<String, usize> = HashMap::new();
    
    for token in &tokens {
        if let Some(s) = token.as_str() {
            *frequencies.entry(s.to_lowercase()).or_insert(0) += 1;
        }
    }
    
    // Sort by frequency
    let mut freq_vec: Vec<_> = frequencies.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
    
    println!("Token frequencies (top 10):");
    for (token, count) in freq_vec.iter().take(10) {
        println!("  '{}': {} times", token, count);
    }
    
    Ok(())
}

/// Create a sample file for testing
fn create_sample_file() -> Result<()> {
    let path = "sample_large_file.txt";
    let mut file = File::create(path)
        .map_err(|e| internal_error(format!("Failed to create file: {}", e)))?;
    
    // Write 10,000 lines
    for i in 0..10_000 {
        writeln!(file, "This is line number {}. It contains sample text for streaming processing. \
                        The quick brown fox jumps over the lazy dog.", i)
            .map_err(|e| internal_error(format!("Failed to write to file: {}", e)))?;
    }
    
    Ok(())
}

fn main() -> Result<()> {
    // Create a sample file if it doesn't exist
    let sample_file = "sample_large_file.txt";
    if !std::path::Path::new(sample_file).exists() {
        println!("Creating sample file...");
        create_sample_file()?;
    }
    
    // Run streaming file processing
    stream_process_file(sample_file)?;
    
    // Run sliding window analysis
    sliding_window_analysis()?;
    
    // Run parallel chunk processing
    parallel_chunk_processing()?;
    
    // Run token frequency analysis
    token_frequency_analysis()?;
    
    // Clean up
    if std::path::Path::new(sample_file).exists() {
        std::fs::remove_file(sample_file)
            .map_err(|e| internal_error(format!("Failed to remove file: {}", e)))?;
    }
    
    println!("\n\nAll streaming examples completed successfully!");
    
    Ok(())
}