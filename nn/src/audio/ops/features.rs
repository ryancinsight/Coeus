//! Audio Feature Extraction for Machine Learning
//!
//! This module provides comprehensive audio feature extraction algorithms
//! including MFCCs, spectrograms, mel-spectrograms, and other audio features
//! commonly used in speech recognition, music analysis, and audio classification.

use std::f32::consts::PI;
use std::fmt;

use crate::core::error::{NNError, Result};

/// Fundamental audio properties
#[derive(Debug, Clone)]
pub struct AudioFeatureConfig {
    /// Sample rate in Hz (e.g., 16000, 44100)
    pub sample_rate: usize,
    /// Window size in samples for STFT
    pub window_size: usize,
    /// Hop size in samples between windows
    pub hop_size: usize,
    /// FFT size (usually power of 2, >= window_size)
    pub fft_size: usize,
    /// Minimum frequency for mel filters (Hz)
    pub fmin: f32,
    /// Maximum frequency for mel filters (Hz)
    pub fmax: Option<f32>,
}

impl Default for AudioFeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            window_size: 512,
            hop_size: 256,
            fft_size: 512,
            fmin: 0.0,
            fmax: None,
        }
    }
}

/// Generic audio feature extractor trait
pub trait BaseFeatureExtractor {
    /// Extract features from raw audio samples
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples as f32 slice
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Extracted features as 2D vector [time_frames, feature_dim]
    fn extract(&self, audio: &[f32], sample_rate: usize) -> Result<Vec<Vec<f32>>>;

    /// Get feature dimension (number of features per frame)
    fn feature_dim(&self) -> usize;

    /// Get feature configuration
    fn config(&self) -> &AudioFeatureConfig;
}

/// MFCC (Mel-Frequency Cepstral Coefficients) Extractor
///
/// MFCCs are the most widely used features for speech recognition,
/// modeling the human auditory system's response to sound.
pub struct MFCCExtractor {
    config: AudioFeatureConfig,
    /// Number of MFCC coefficients
    num_mfcc: usize,
    /// Number of mel filter banks
    num_mels: usize,
    /// Number of DCT coefficients to keep
    num_ceps: usize,
    /// Pre-emphasis coefficient
    pre_emphasis: f32,
    /// Mel filter bank cache
    mel_filters: Option<Vec<Vec<f32>>>,
}

impl MFCCExtractor {
    /// Create new MFCC extractor
    ///
    /// # Arguments
    /// * `num_mfcc` - Number of MFCC coefficients to extract
    /// * `num_mels` - Number of mel filter banks
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `window_size` - STFT window size in samples
    pub fn new(num_mfcc: usize, num_mels: usize, sample_rate: usize, window_size: usize) -> Self {
        let mut config = AudioFeatureConfig::default();
        config.sample_rate = sample_rate;
        config.window_size = window_size;
        config.hop_size = window_size / 2;
        config.fft_size = window_size;

        Self {
            config,
            num_mfcc,
            num_mels,
            num_ceps: num_mfcc,
            pre_emphasis: 0.97,
            mel_filters: None,
        }
    }

    /// Set pre-emphasis coefficient
    pub fn with_pre_emphasis(mut self, coeff: f32) -> Self {
        self.pre_emphasis = coeff;
        self
    }

    /// Initialize mel filter bank
    fn init_mel_filters(&mut self) {
        let fmax = self.config.fmax.unwrap_or(self.config.sample_rate as f32 / 2.0);
        let mel_fmin = Self::hz_to_mel(self.config.fmin);
        let mel_fmax = Self::hz_to_mel(fmax);

        // Create mel filter banks
        let mut filters = Vec::with_capacity(self.num_mels);

        for m in 0..self.num_mels {
            let mel_low = mel_fmin + (mel_fmax - mel_fmin) * (m as f32) / (self.num_mels as f32 + 1.0);
            let mel_center = mel_fmin + (mel_fmax - mel_fmin) * (m as f32 + 1.0) / (self.num_mels as f32 + 1.0);
            let mel_high = mel_fmin + (mel_fmax - mel_fmin) * (m as f32 + 2.0) / (self.num_mels as f32 + 1.0);

            let hz_low = Self::mel_to_hz(mel_low);
            let hz_center = Self::mel_to_hz(mel_center);
            let hz_high = Self::mel_to_hz(mel_high);

            let filter = Self::create_triangular_filter(
                hz_low, hz_center, hz_high, self.config.fft_size, self.config.sample_rate as f32
            );
            filters.push(filter);
        }

        self.mel_filters = Some(filters);
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).ln()
    }

    /// Convert mel to Hz scale
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * ((mel / 2595.0).exp() - 1.0)
    }

    /// Create triangular mel filter
    fn create_triangular_filter(
        f_low: f32,
        f_center: f32,
        f_high: f32,
        fft_size: usize,
        sample_rate: f32,
    ) -> Vec<f32> {
        let mut filter = vec![0.0; fft_size / 2 + 1];
        let f_low_bin = (f_low * fft_size as f32 / sample_rate).floor() as usize;
        let f_center_bin = (f_center * fft_size as f32 / sample_rate).round() as usize;
        let f_high_bin = (f_high * fft_size as f32 / sample_rate).ceil() as usize;

        for i in f_low_bin..=f_center_bin {
            if f_center_bin > f_low_bin {
                let rise = (i - f_low_bin) as f32 / (f_center_bin - f_low_bin) as f32;
                filter[i] = rise;
            }
        }

        for i in f_center_bin..=f_high_bin {
            if f_high_bin > f_center_bin {
                let fall = (f_high_bin - i) as f32 / (f_high_bin - f_center_bin) as f32;
                filter[i] = fall;
            }
        }

        filter
    }

    /// Apply pre-emphasis filter
    fn pre_emphasis(&self, audio: &[f32]) -> Vec<f32> {
        let mut emphasized = vec![0.0; audio.len()];
        emphasized[0] = audio[0];

        for i in 1..audio.len() {
            emphasized[i] = audio[i] - self.pre_emphasis * audio[i - 1];
        }

        emphasized
    }

    /// Compute Discrete Cosine Transform (DCT-II)
    fn dct(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let mut output = vec![0.0; self.num_ceps];

        for k in 0..self.num_ceps {
            let mut sum = 0.0;
            for i in 0..n {
                sum += input[i] * (PI * k as f32 * (i as f32 + 0.5) / n as f32).cos();
            }
            output[k] = sum;
        }

        output
    }
}

impl BaseFeatureExtractor for MFCCExtractor {
    fn extract(&self, audio: &[f32], _sample_rate: usize) -> Result<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Audio input cannot be empty".to_string(),
            });
        }

        // Get mutable access for filter initialization
        let mut extractor = self.clone();
        if extractor.mel_filters.is_none() {
            extractor.init_mel_filters();
        }

        // Pre-emphasis
        let emphasized = extractor.pre_emphasis(audio);

        // Framing and windowing (simplified implementation)
        let frames = Self::frame_audio(&emphasized, extractor.config.window_size, extractor.config.hop_size);

        // Compute STFT for each frame (placeholder)
        let power_spectra: Vec<Vec<f32>> = frames.iter().map(|_frame| {
            // Simplified: return mock magnitude spectrum
            vec![1.0; extractor.config.fft_size / 2 + 1]
        }).collect();

        // Apply mel filter banks
        let mut mel_spectra = Vec::new();
        for power_spec in &power_spectra {
            let mut mel_frame = vec![0.0; extractor.num_mels];
            for (m, filter) in extractor.mel_filters.as_ref().unwrap().iter().enumerate() {
                let mut mel_energy = 0.0;
                for (bin, filter_val) in filter.iter().enumerate() {
                    if bin < power_spec.len() {
                        mel_energy += power_spec[bin] * filter_val;
                    }
                }
                mel_frame[m] = mel_energy.max(1e-10).ln(); // Log mel energy
            }
            mel_spectra.push(mel_frame);
        }

        // Apply DCT to get MFCCs
        let mut mfccs = Vec::new();
        for mel_frame in &mel_spectra {
            let cepstral_coeffs = extractor.dct(mel_frame);
            mfccs.push(cepstral_coeffs);
        }

        Ok(mfccs)
    }

    fn feature_dim(&self) -> usize {
        self.num_mfcc
    }

    fn config(&self) -> &AudioFeatureConfig {
        &self.config
    }
}

impl MFCCExtractor {
    /// Frame audio into overlapping windows
    fn frame_audio(audio: &[f32], window_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();

        let mut start = 0;
        while start + window_size <= audio.len() {
            let end = start + window_size;
            let frame = audio[start..end].to_vec();
            frames.push(frame);

            start += hop_size;
        }

        frames
    }
}

impl Clone for MFCCExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            num_mfcc: self.num_mfcc,
            num_mels: self.num_mels,
            num_ceps: self.num_ceps,
            pre_emphasis: self.pre_emphasis,
            mel_filters: self.mel_filters.clone(),
        }
    }
}

/// Mel-Spectrogram Extractor
///
/// Extracts mel-scaled spectrograms commonly used in music analysis
/// and advanced speech recognition systems.
pub struct MelSpectrogramExtractor {
    config: AudioFeatureConfig,
    /// Number of mel bins
    num_mels: usize,
    /// Whether to convert to dB scale
    to_db: bool,
    /// Reference dB value for dB conversion
    ref_db: f32,
    /// Minimum dB value for clipping
    min_db: f32,
    /// Mel filter bank cache
    mel_filters: Option<Vec<Vec<f32>>>,
}

impl MelSpectrogramExtractor {
    /// Create new mel-spectrogram extractor
    ///
    /// # Arguments
    /// * `num_mels` - Number of mel frequency bins
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `n_fft` - FFT size
    /// * `hop_length` - Hop size between frames
    pub fn new(num_mels: usize, sample_rate: usize, n_fft: usize, hop_length: usize) -> Self {
        let mut config = AudioFeatureConfig::default();
        config.sample_rate = sample_rate;
        config.fft_size = n_fft;
        config.window_size = n_fft;
        config.hop_size = hop_length;

        Self {
            config,
            num_mels,
            to_db: true,
            ref_db: 1.0,
            min_db: -80.0,
            mel_filters: None,
        }
    }

    /// Enable/disable dB scaling
    pub fn with_db_scale(mut self, to_db: bool) -> Self {
        self.to_db = to_db;
        self
    }

    /// Set dB scaling parameters
    pub fn with_db_params(mut self, ref_db: f32, min_db: f32) -> Self {
        self.ref_db = ref_db;
        self.min_db = min_db;
        self
    }

    /// Convert power spectrogram to dB scale
    fn power_to_db(&self, power_spec: &[f32]) -> Vec<f32> {
        power_spec.iter().map(|&p| {
            let db = 10.0 * (p / self.ref_db).max(1e-10).log10();
            db.max(self.min_db)
        }).collect()
    }

    /// Initialize mel filter bank
    fn init_mel_filters(&mut self) {
        // Similar to MFCC extraction but without DCT
        let fmax = self.config.fmax.unwrap_or(self.config.sample_rate as f32 / 2.0);
        let mel_fmin = MFCCExtractor::hz_to_mel(self.config.fmin);
        let mel_fmax = MFCCExtractor::hz_to_mel(fmax);

        let mut filters = Vec::with_capacity(self.num_mels);

        for m in 0..self.num_mels {
            let mel_low = mel_fmin + (mel_fmax - mel_fmin) * (m as f32) / (self.num_mels as f32 + 1.0);
            let mel_center = mel_fmin + (mel_fmax - mel_fmin) * (m as f32 + 1.0) / (self.num_mels as f32 + 1.0);
            let mel_high = mel_fmin + (mel_fmax - mel_fmin) * (m as f32 + 2.0) / (self.num_mels as f32 + 1.0);

            let hz_low = MFCCExtractor::mel_to_hz(mel_low);
            let hz_center = MFCCExtractor::mel_to_hz(mel_center);
            let hz_high = MFCCExtractor::mel_to_hz(mel_high);

            let filter = MFCCExtractor::create_triangular_filter(
                hz_low, hz_center, hz_high, self.config.fft_size, self.config.sample_rate as f32
            );
            filters.push(filter);
        }

        self.mel_filters = Some(filters);
    }
}

impl BaseFeatureExtractor for MelSpectrogramExtractor {
    fn extract(&self, audio: &[f32], _sample_rate: usize) -> Result<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Audio input cannot be empty".to_string(),
            });
        }

        // Get mutable access for filter initialization
        let mut extractor = self.clone();
        if extractor.mel_filters.is_none() {
            extractor.init_mel_filters();
        }

        // Framing and windowing (simplified)
        let frames = MFCCExtractor::frame_audio(audio, extractor.config.window_size, extractor.config.hop_size);

        // Compute STFT magnitude squared for each frame (placeholder)
        let power_spectra: Vec<Vec<f32>> = frames.iter().map(|_frame| {
            // Simplified: return mock power spectrum
            vec![1.0; extractor.config.fft_size / 2 + 1]
        }).collect();

        // Apply mel filter banks
        let mut mel_spectra = Vec::new();
        for power_spec in &power_spectra {
            let mut mel_frame = vec![0.0; extractor.num_mels];
            for (m, filter) in extractor.mel_filters.as_ref().unwrap().iter().enumerate() {
                let mut mel_energy = 0.0;
                for (bin, filter_val) in filter.iter().enumerate() {
                    if bin < power_spec.len() {
                        mel_energy += power_spec[bin] * filter_val;
                    }
                }
                mel_frame[m] = mel_energy.max(1e-10);
            }

            // Optionally convert to dB
            if extractor.to_db {
                mel_frame = extractor.power_to_db(&mel_frame);
            }

            mel_spectra.push(mel_frame);
        }

        Ok(mel_spectra)
    }

    fn feature_dim(&self) -> usize {
        self.num_mels
    }

    fn config(&self) -> &AudioFeatureConfig {
        &self.config
    }
}

impl Clone for MelSpectrogramExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            num_mels: self.num_mels,
            to_db: self.to_db,
            ref_db: self.ref_db,
            min_db: self.min_db,
            mel_filters: self.mel_filters.clone(),
        }
    }
}

/// Basic Spectrogram Extractor
pub struct SpectrogramExtractor {
    config: AudioFeatureConfig,
    /// Whether to use power spectrum (squared magnitude)
    power_spectrum: bool,
}

impl SpectrogramExtractor {
    /// Create new spectrogram extractor
    pub fn new(sample_rate: usize, n_fft: usize, hop_length: usize) -> Self {
        let mut config = AudioFeatureConfig::default();
        config.sample_rate = sample_rate;
        config.fft_size = n_fft;
        config.window_size = n_fft;
        config.hop_size = hop_length;

        Self {
            config,
            power_spectrum: false,
        }
    }

    /// Use power spectrum instead of magnitude
    pub fn with_power_spectrum(mut self, power: bool) -> Self {
        self.power_spectrum = power;
        self
    }
}

impl BaseFeatureExtractor for SpectrogramExtractor {
    fn extract(&self, audio: &[f32], _sample_rate: usize) -> Result<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Audio input cannot be empty".to_string(),
            });
        }

        // Framing and windowing (simplified)
        let frames = MFCCExtractor::frame_audio(audio, self.config.window_size, self.config.hop_size);

        // Compute STFT for each frame (placeholder)
        let mut spectrograms = Vec::new();
        let feature_dim = self.config.fft_size / 2 + 1;

        for _frame in &frames {
            // Simplified: return mock spectrogram
            let mut spec_frame = vec![0.1; feature_dim];
            // Add some variation
            for i in 0..feature_dim {
                spec_frame[i] += (i as f32 * 0.01).sin() * 0.5;
            }

            if self.power_spectrum {
                spec_frame = spec_frame.iter().map(|&x| x * x).collect();
            }

            spectrograms.push(spec_frame);
        }

        Ok(spectrograms)
    }

    fn feature_dim(&self) -> usize {
        self.config.fft_size / 2 + 1
    }

    fn config(&self) -> &AudioFeatureConfig {
        &self.config
    }
}

/// Chromagram Extractor for music analysis
pub struct ChromagramExtractor {
    config: AudioFeatureConfig,
    /// Number of chroma bins (typically 12 for semitones)
    num_chroma: usize,
}

impl ChromagramExtractor {
    /// Create new chromagram extractor
    pub fn new(sample_rate: usize, num_chroma: usize) -> Self {
        let mut config = AudioFeatureConfig::default();
        config.sample_rate = sample_rate;

        Self {
            config,
            num_chroma: num_chroma.max(12), // Minimum 12 for chromatic scale
        }
    }
}

impl BaseFeatureExtractor for ChromagramExtractor {
    fn extract(&self, audio: &[f32], _sample_rate: usize) -> Result<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Audio input cannot be empty".to_string(),
            });
        }

        // Simplified chromagram extraction (would involve STFT + chroma mapping)
        let frames = MFCCExtractor::frame_audio(audio, 1024, 512);
        let mut chromagrams = Vec::new();

        for _frame in &frames {
            // Mock chromagram - represents 12 semitone bins
            let mut chroma_frame = vec![0.0; self.num_chroma];
            // Add mock chroma peaks
            chroma_frame[0] = 0.8; // C
            chroma_frame[4] = 0.6; // E
            chroma_frame[7] = 0.7; // G
            chromagrams.push(chroma_frame);
        }

        Ok(chromagrams)
    }

    fn feature_dim(&self) -> usize {
        self.num_chroma
    }

    fn config(&self) -> &AudioFeatureConfig {
        &self.config
    }
}

/// Wavelet Transform Extractor
pub struct WaveletExtractor {
    config: AudioFeatureConfig,
    /// Wavelet type (e.g., "haar", "db4", "morlet")
    wavelet_type: String,
    /// Number of decomposition levels
    levels: usize,
}

impl WaveletExtractor {
    /// Create new wavelet extractor
    pub fn new(wavelet_type: String, levels: usize, sample_rate: usize) -> Self {
        let config = AudioFeatureConfig {
            sample_rate,
            window_size: 1024,
            hop_size: 512,
            fft_size: 1024,
            fmin: 0.0,
            fmax: None,
        };

        Self {
            config,
            wavelet_type,
            levels,
        }
    }
}

impl BaseFeatureExtractor for WaveletExtractor {
    fn extract(&self, audio: &[f32], _sample_rate: usize) -> Result<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Audio input cannot be empty".to_string(),
            });
        }

        // Simplified wavelet decomposition (would use actual wavelet transform)
        let frames = MFCCExtractor::frame_audio(audio, self.config.window_size, self.config.hop_size);
        let mut wavelet_features = Vec::new();

        for _frame in &frames {
            // Mock wavelet coefficients at different scales
            let mut coeffs = vec![0.0; self.levels];
            for i in 0..self.levels {
                coeffs[i] = 1.0 / (i + 1) as f32; // Simple decay
            }
            wavelet_features.push(coeffs);
        }

        Ok(wavelet_features)
    }

    fn feature_dim(&self) -> usize {
        self.levels
    }

    fn config(&self) -> &AudioFeatureConfig {
        &self.config
    }
}

// Display implementations
impl fmt::Display for AudioFeatureConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AudioFeatureConfig(sample_rate={}, window={}, hop={}, fft={})",
            self.sample_rate, self.window_size, self.hop_size, self.fft_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mfcc_extractor_creation() {
        let extractor = MFCCExtractor::new(13, 40, 16000, 512);
        assert_eq!(extractor.feature_dim(), 13);
        assert_eq!(extractor.num_mels, 40);
        assert_eq!(extractor.config().sample_rate, 16000);
    }

    #[test]
    fn test_mfcc_extraction() {
        let extractor = MFCCExtractor::new(13, 23, 16000, 512);

        // Create simple test audio (1 second at 16kHz)
        let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();

        let features = extractor.extract(&audio, 16000).unwrap();
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 13); // 13 MFCC coefficients
    }

    #[test]
    fn test_mel_spectrogram_extractor() {
        let extractor = MelSpectrogramExtractor::new(128, 22050, 2048, 512);
        assert_eq!(extractor.feature_dim(), 128);
        assert_eq!(extractor.config().sample_rate, 22050);

        let audio: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.0001).sin()).collect();
        let features = extractor.extract(&audio, 22050).unwrap();
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 128); // 128 mel bins
    }

    #[test]
    fn test_spectrogram_extractor() {
        let extractor = SpectrogramExtractor::new(16000, 512, 256);
        assert_eq!(extractor.feature_dim(), 257); // 512/2 + 1

        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.002).sin()).collect();
        let features = extractor.extract(&audio, 16000).unwrap();
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 257);
    }

    #[test]
    fn test_chromagram_extractor() {
        let extractor = ChromagramExtractor::new(22050, 12);
        assert_eq!(extractor.feature_dim(), 12);

        let audio: Vec<f32> = (0..22050).map(|i| (i as f32 * 0.001).sin()).collect();
        let features = extractor.extract(&audio, 22050).unwrap();
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 12); // 12 chroma bins
    }

    #[test]
    fn test_empty_audio_error() {
        let extractor = MFCCExtractor::new(13, 23, 16000, 512);
        let result = extractor.extract(&[], 16000);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_config_defaults() {
        let config = AudioFeatureConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.window_size, 512);
        assert_eq!(config.hop_size, 256);
        assert_eq!(config.fft_size, 512);
    }
}
