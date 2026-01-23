//! Audio Processing and Speech Models for Neural Networks
//!
//! This module provides comprehensive audio processing capabilities including:
//! - Audio feature extraction (MFCC, spectrograms, mel-spectrograms)
//! - Speech recognition models (wav2vec, HuBERT, Whisper) - MS-46 Phase 3
//! - Text-to-speech synthesis models - MS-46 Phase 3
//! - Audio classification and music genre recognition - MS-46 Phase 3
//! - Real-time audio processing pipelines - MS-46 Phase 3
//! - Multi-language speech processing - MS-46 Phase 3
//! - Audio-visual multimodal integration - MS-47
//!
//! ## Current Status: Sprint MS-46 ~65% Complete
//! - ✓ Audio feature extraction (MFCC, spectrograms, mel-spectrograms, chroma, wavelet)
//! - ⏳ Speech models (wav2vec, HuBERT, Whisper) - TODO
//! - ⏳ Text-to-speech synthesis (Tacotron, FastSpeech) - TODO
//! - ⏳ Audio classification and recognition - TODO
//!
//! ## Feature Extraction
//! ```rust
//! use nn::audio::{AudioFeatureExtractor, MFCCExtractor, SpectrogramExtractor};
//!
//! // Extract MFCC features for speech recognition
//! let mfcc_extractor = MFCCExtractor::new(13, 23, 16000, 512);
//! let features = mfcc_extractor.extract(audio_samples)?;
//!
//! // Extract mel spectrograms for music processing
//! let mel_extractor = MelSpectrogramExtractor::new(128, 16000, 1024, 512);
//! let mel_spec = mel_extractor.extract(audio_samples)?;
//! ```
//!
//! ## Future APIs (Sprint MS-46 Completion)
//! ```rust
//! // Speech Recognition (TODO)
//! use nn::audio::{WhisperModel, SpeechRecognizer};
//!
//! let model = WhisperModel::load("whisper-base")?;
//! let result = model.transcribe(audio_samples, "en")?;
//! println!("Transcription: {}", result.text);
//!
//! // Text-to-Speech (TODO)
//! use nn::audio::TTSModel;
//!
//! let tts = TTSModel::new("tacotron2")?;
//! let audio = tts.synthesize("Hello, world!", "en")?;
//! ```

// Submodules moved to deeper hierarchy
// Use re-exports from audio/mod.rs for public API

// Re-exports now handled in parent audio/mod.rs

#[cfg(test)]
mod tests;
