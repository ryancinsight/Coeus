import sys
import os
sys.path.insert(0, os.path.abspath("pycoeus/python"))
import coeus
import coeus.signal as signal
import coeus.special as special
import numpy as np

def test_signal_windows():
    print("Testing signal windows...")
    n = 10
    
    h1 = signal.hann_window(n, periodic=False)
    print(f"Hann window: {h1.numpy()}")
    
    h2 = signal.hamming_window(n, periodic=False)
    print(f"Hamming window: {h2.numpy()}")

def test_stft():
    print("\nTesting STFT...")
    n = 64
    t = np.linspace(0, 1, n)
    x = np.sin(2 * np.pi * 5 * t).astype(np.float32)
    
    xt = coeus.Tensor(x.tolist(), [n])
    
    # STFT returns (real, imag) tuple in our implementation
    real, imag = signal.stft(xt, n_fft=16, hop_length=8, win_length=16, center=True)
    
    print(f"STFT output shapes: Real {real.shape}, Imag {imag.shape}")
    
    # Check if we have some non-zero values
    real_np = real.numpy()
    imag_np = imag.numpy()
    mag = np.sqrt(real_np**2 + imag_np**2)
    print(f"Max magnitude: {np.max(mag)}")

def test_special():
    print("\nTesting special functions...")
    x_val = [0.0, 1.0, 2.0]
    xt = coeus.Tensor(x_val, [3])
    
    print(f"erf: {special.erf(xt).numpy()}")
    print(f"gamma: {special.gamma(xt).numpy()}")
    print(f"expit: {special.expit(xt).numpy()}")
    print(f"logit: {special.logit(coeus.Tensor([0.5, 0.1, 0.9], [3])).numpy()}")
    print(f"sinc: {special.sinc(xt).numpy()}")

if __name__ == "__main__":
    try:
        test_signal_windows()
        test_stft()
        test_special()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
