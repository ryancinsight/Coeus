from .. import _coeus
import numpy as np

def fft(input):
    """Compute the one-dimensional discrete Fourier Transform."""
    if isinstance(input, np.ndarray):
        input_list = input.astype(np.float32).tolist()
    else:
        input_list = [float(x) for x in input]
    
    size = len(input_list)
    f = _coeus.FFT(size)
    return f.forward(input_list)

def ifft(input):
    """Compute the one-dimensional inverse discrete Fourier Transform."""
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

    size = len(input_list)
    f = _coeus.IFFT(size)
    return f.inverse(input_list)
