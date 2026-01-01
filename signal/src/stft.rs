use coeus_error::Result;
use dtype::complex::Complex32;
use dtype::float::Float32;
use storage::DenseStorage;
use storage::Storage;

pub struct StftConfig<'a> {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub window: Option<&'a [Float32]>,
    pub center: bool,
    pub normalized: bool,
    pub onesided: bool,
}

impl<'a> StftConfig<'a> {
    pub fn validate(&self) -> Result<()> {
        if self.n_fft == 0 {
            return Err(coeus_error::TensorError::InvalidIndex("n_fft must be > 0".to_string()).into());
        }
        if self.hop_length == 0 {
            return Err(
                coeus_error::TensorError::InvalidIndex("hop_length must be > 0".to_string()).into(),
            );
        }
        if self.win_length == 0 || self.win_length > self.n_fft {
            return Err(coeus_error::TensorError::ShapeMismatch(
                "win_length must be in 1..=n_fft".to_string(),
            )
            .into());
        }
        if let Some(w) = self.window {
            if w.len() != self.win_length {
                return Err(coeus_error::TensorError::ShapeMismatch(
                    "window length must equal win_length".to_string(),
                )
                .into());
            }
        }
        Ok(())
    }
}

pub fn stft_1d(input: &DenseStorage<Float32>, cfg: StftConfig<'_>) -> Result<DenseStorage<Complex32>> {
    cfg.validate()?;
    let x = input.as_slice();
    let pad = if cfg.center { cfg.n_fft / 2 } else { 0 };
    let padded_len = x.len() + 2 * pad;
    if padded_len < cfg.n_fft {
        return Err(coeus_error::TensorError::ShapeMismatch("input too short for n_fft".to_string()).into());
    }

    let n_frames = 1 + (padded_len - cfg.n_fft) / cfg.hop_length;
    let n_freq = if cfg.onesided {
        cfg.n_fft / 2 + 1
    } else {
        cfg.n_fft
    };

    let window_nfft = build_window_nfft(cfg.n_fft, cfg.win_length, cfg.window)?;

    let mut out = vec![Complex32::new(0.0, 0.0); n_freq * n_frames];
    let fft = fft::cpu::CpuFft::new(cfg.n_fft);
    let scale = if cfg.normalized {
        1.0 / (cfg.n_fft as f32).sqrt()
    } else {
        1.0
    };

    for frame in 0..n_frames {
        let start = frame * cfg.hop_length;
        let mut frame_vec = vec![Float32::new(0.0); cfg.n_fft];
        for i in 0..cfg.n_fft {
            let src = start + i;
            let sample = if pad == 0 {
                x[src].get()
            } else {
                let orig = (src as isize) - (pad as isize);
                let idx = reflect_index(orig, x.len());
                x[idx].get()
            };
            frame_vec[i] = Float32::new(sample * window_nfft[i]);
        }
        let frame_storage = DenseStorage::from_vec(frame_vec, &[cfg.n_fft])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()))?;
        let spec = fft.forward(&frame_storage)?;
        let spec_slice = spec.as_slice();
        for f in 0..n_freq {
            out[f * n_frames + frame] = Complex32::new(spec_slice[f].re * scale, spec_slice[f].im * scale);
        }
    }

    DenseStorage::from_vec(out, &[n_freq, n_frames])
        .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
}

pub struct IstftConfig<'a> {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub window: Option<&'a [Float32]>,
    pub center: bool,
    pub normalized: bool,
    pub onesided: bool,
    pub length: Option<usize>,
}

impl<'a> IstftConfig<'a> {
    pub fn validate(&self, spec: &DenseStorage<Complex32>) -> Result<()> {
        if self.n_fft == 0 {
            return Err(coeus_error::TensorError::InvalidIndex("n_fft must be > 0".to_string()).into());
        }
        if self.hop_length == 0 {
            return Err(
                coeus_error::TensorError::InvalidIndex("hop_length must be > 0".to_string()).into(),
            );
        }
        if self.win_length == 0 || self.win_length > self.n_fft {
            return Err(coeus_error::TensorError::ShapeMismatch(
                "win_length must be in 1..=n_fft".to_string(),
            )
            .into());
        }
        if let Some(w) = self.window {
            if w.len() != self.win_length {
                return Err(coeus_error::TensorError::ShapeMismatch(
                    "window length must equal win_length".to_string(),
                )
                .into());
            }
        }
        let dims = spec.shape().dims();
        if dims.len() != 2 {
            return Err(coeus_error::TensorError::ShapeMismatch("spec must be 2D".to_string()).into());
        }
        let expected_freq = if self.onesided { self.n_fft / 2 + 1 } else { self.n_fft };
        if dims[0] != expected_freq {
            return Err(coeus_error::TensorError::ShapeMismatch("spec frequency dimension mismatch".to_string()).into());
        }
        Ok(())
    }
}

pub fn istft_1d(spec: &DenseStorage<Complex32>, cfg: IstftConfig<'_>) -> Result<DenseStorage<Float32>> {
    cfg.validate(spec)?;
    let dims = spec.shape().dims();
    let n_freq = dims[0];
    let n_frames = dims[1];
    let pad = if cfg.center { cfg.n_fft / 2 } else { 0 };

    let out_len = cfg.n_fft + cfg.hop_length * (n_frames.saturating_sub(1));
    let mut y = vec![0.0f32; out_len];
    let mut wsum = vec![0.0f32; out_len];
    let window_nfft = build_window_nfft(cfg.n_fft, cfg.win_length, cfg.window)?;

    let fft = fft::cpu::CpuFft::new(cfg.n_fft);
    let inv_scale = if cfg.normalized {
        1.0 / (cfg.n_fft as f32).sqrt()
    } else {
        1.0
    };

    let spec_data = spec.as_slice();
    for frame in 0..n_frames {
        let mut full = vec![Complex32::new(0.0, 0.0); cfg.n_fft];
        if cfg.onesided {
            full[0] = spec_data[0 * n_frames + frame];
            if cfg.n_fft % 2 == 0 {
                full[cfg.n_fft / 2] = spec_data[(n_freq - 1) * n_frames + frame];
            }
            for k in 1..(n_freq - 1) {
                let v = spec_data[k * n_frames + frame];
                full[k] = v;
                full[cfg.n_fft - k] = v.conj();
            }
        } else {
            for k in 0..cfg.n_fft {
                full[k] = spec_data[k * n_frames + frame];
            }
        }

        if cfg.normalized {
            for v in &mut full {
                *v = Complex32::new(v.re * inv_scale, v.im * inv_scale);
            }
        }

        let full_storage = DenseStorage::from_vec(full, &[cfg.n_fft])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()))?;
        let time = fft.inverse(&full_storage)?;
        let t = time.as_slice();
        let base = frame * cfg.hop_length;
        for i in 0..cfg.n_fft {
            let pos = base + i;
            let w = window_nfft[i];
            y[pos] += t[i].get() * w;
            wsum[pos] += w * w;
        }
    }

    let eps = 1e-11f32;
    for i in 0..out_len {
        if wsum[i] > eps {
            y[i] /= wsum[i];
        }
    }

    let mut y = if pad > 0 && out_len >= 2 * pad {
        y[pad..(out_len - pad)].to_vec()
    } else {
        y
    };

    if let Some(len) = cfg.length {
        if y.len() > len {
            y.truncate(len);
        } else if y.len() < len {
            y.resize(len, 0.0);
        }
    }

    let out: Vec<Float32> = y.into_iter().map(Float32::new).collect();
    let out_len = out.len();
    DenseStorage::from_vec(out, &[out_len])
        .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
}

fn reflect_index(i: isize, len: usize) -> usize {
    let n = len as isize;
    if n <= 1 {
        return 0;
    }
    let mut idx = i;
    loop {
        if idx < 0 {
            idx = -idx;
        } else if idx >= n {
            idx = 2 * n - 2 - idx;
        } else {
            return idx as usize;
        }
    }
}

fn build_window_nfft(
    n_fft: usize,
    win_length: usize,
    window: Option<&[Float32]>,
) -> Result<Vec<f32>> {
    let mut w = vec![0.0f32; n_fft];
    let start = (n_fft - win_length) / 2;
    if let Some(src) = window {
        for (i, v) in src.iter().enumerate() {
            w[start + i] = v.get();
        }
    } else {
        for i in 0..win_length {
            w[start + i] = 1.0;
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[Float32], b: &[Float32], atol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            let dx = (x.get() - y.get()).abs();
            assert!(dx <= atol, "diff {dx} > {atol}");
        }
    }

    #[test]
    fn stft_istft_roundtrip_centered_rectangular() {
        let n = 512usize;
        let x: Vec<Float32> = (0..n)
            .map(|i| {
                let t = i as f32 / 32.0;
                Float32::new((2.0 * std::f32::consts::PI * t).sin())
            })
            .collect();
        let input = match DenseStorage::from_vec(x, &[n]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let cfg = StftConfig {
            n_fft: 64,
            hop_length: 16,
            win_length: 64,
            window: None,
            center: true,
            normalized: false,
            onesided: true,
        };
        let spec = match stft_1d(&input, cfg) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let icfg = IstftConfig {
            n_fft: 64,
            hop_length: 16,
            win_length: 64,
            window: None,
            center: true,
            normalized: false,
            onesided: true,
            length: Some(n),
        };
        let recon = match istft_1d(&spec, icfg) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx_eq(recon.as_slice(), input.as_slice(), 1e-3);
    }

    #[test]
    fn stft_shapes_match_expected() {
        let n = 256usize;
        let x: Vec<Float32> = (0..n).map(|i| Float32::new(i as f32)).collect();
        let input = match DenseStorage::from_vec(x, &[n]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let cfg = StftConfig {
            n_fft: 32,
            hop_length: 8,
            win_length: 16,
            window: None,
            center: true,
            normalized: false,
            onesided: true,
        };
        let spec = match stft_1d(&input, cfg) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let dims = spec.shape().dims();
        assert_eq!(dims[0], 17);
        let pad = 16usize;
        let padded_len = n + 2 * pad;
        let expected_frames = 1 + (padded_len - 32) / 8;
        assert_eq!(dims[1], expected_frames);
    }
}
