import torch
import numpy as np
import coeus
import coeus.nn as nn
import pytest

def test_fft_forward_parity():
    size = 1024
    input_np = np.random.randn(size).astype(np.float32)
    
    # torch
    input_torch = torch.from_numpy(input_np)
    fft_torch = torch.fft.fft(input_torch)
    
    # coeus functional API
    fft_coeus_result = coeus.fft.fft(input_np)
    
    # Convert coeus result to numpy
    fft_coeus_np = np.array([complex(re, im) for re, im in fft_coeus_result], dtype=np.complex64)
    
    # Compare with higher tolerance for FFT numerical variations
    np.testing.assert_allclose(fft_coeus_np, fft_torch.numpy(), atol=1e-4)
    print("FFT forward parity passed!")

def test_fft_inverse_parity():
    size = 1024
    # Create original signal
    original_np = np.random.randn(size).astype(np.float32)
    # Get its FFT to get complex input for IFFT
    input_complex_np = np.fft.fft(original_np).astype(np.complex64)
    
    # torch
    input_torch = torch.from_numpy(input_complex_np)
    ifft_torch = torch.fft.ifft(input_torch)
    
    # coeus functional API
    ifft_coeus_result = coeus.fft.ifft(input_complex_np)
    
    # Convert coeus result to numpy
    ifft_coeus_np = np.array(ifft_coeus_result, dtype=np.float32)
    
    # Compare
    np.testing.assert_allclose(ifft_coeus_np, ifft_torch.real.numpy(), atol=1e-4)
    print("FFT inverse parity passed!")

if __name__ == "__main__":
    test_fft_forward_parity()
    test_fft_inverse_parity()
