from .. import _coeus
from ..tensor import Tensor
import numpy as np

def fft(input, n=None):
    """
    Compute the 1 dimensional discrete Fourier Transform.
    
    Args:
        input (Tensor): The input tensor.
        n (int, optional): Signal length.
        
    Returns:
        tuple (Tensor, Tensor): (real, imag) parts of the FFT.
    """
    if isinstance(input, Tensor):
        return _coeus.fft(input, n)
    
    # Legacy support for lists/numpy
    if isinstance(input, np.ndarray):
        input_list = input.astype(np.float32).tolist()
    else:
        input_list = [float(x) for x in input]
    
    size = n if n is not None else len(input_list)
    f = _coeus.FFT(size)
    return f.forward(input_list)

def ifft(input_real, input_imag=None, n=None):
    """
    Compute the 1 dimensional inverse discrete Fourier Transform.
    
    Args:
        input_real (Tensor or list): Real part of spectrum (or full complex list/numpy).
        input_imag (Tensor, optional): Imaginary part of spectrum.
        n (int, optional): Signal length.
        
    Returns:
        tuple (Tensor, Tensor): (real, imag) parts of the IFFT.
    """
    if isinstance(input_real, Tensor):
        if input_imag is None:
            raise ValueError("input_imag is required when input_real is a Tensor")
        return _coeus.ifft(input_real, input_imag, n)
    
    # Legacy support
    input = input_real
    if isinstance(input, np.ndarray):
        if np.iscomplexobj(input):
            input_list = [(float(c.real), float(c.imag)) for c in input.flatten()]
        else:
            input_list = [(float(x), 0.0) for x in input.flatten()]
    elif isinstance(input, list):
        if len(input) > 0 and isinstance(input[0], complex):
            input_list = [(float(c.real), float(c.imag)) for c in input]
        elif len(input) > 0 and isinstance(input[0], (tuple, list)):
            input_list = [(float(x[0]), float(x[1])) for x in input]
        else:
            input_list = [(float(x), 0.0) for x in input]
    else:
        raise ValueError("Input must be a list or numpy array")

    size = n if n is not None else len(input_list)
    f = _coeus.IFFT(size)
    return f.inverse(input_list)

def rfft(input, n=None):
    """
    Compute the 1 dimensional discrete Fourier Transform of a real-valued input.
    Returns the one-sided spectrum.
    """
    if not isinstance(input, Tensor):
        input = Tensor(input)
    return _coeus.rfft(input, n)

def irfft(input_real, input_imag, n=None):
    """
    Compute the inverse of rfft.
    Takes a one-sided spectrum and returns a real-valued signal.
    """
    if not isinstance(input_real, Tensor):
        input_real = Tensor(input_real)
    if not isinstance(input_imag, Tensor):
        input_imag = Tensor(input_imag)
    return _coeus.irfft(input_real, input_imag, n)
