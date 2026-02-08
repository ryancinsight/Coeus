# Import Tensor from parent module
from .._coeus import Tensor

# The NN classes are registered directly in the PyO3 module
# We'll access them through the module namespace

class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            return super().__new__(cls, [], [0])
        if isinstance(data, Tensor):
            # Special case for PyO3: we can't easily change the class of an existing object
            # but since we enabled subclassing, we can return it if it's already a Tensor
            # or try to wrap it.
            return data
        return super().__new__(cls, data)

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"

class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        self.training = True
        self._modules = {}
        self._parameters = {}

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def children(self):
        for name, module in self._modules.items():
            yield module

    def modules(self):
        yield self
        for name, module in self._modules.items():
            for m in module.modules():
                yield m

    def parameters(self, recurse=True):
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for module in self.children():
                yield from module.parameters(recurse)

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module)} is not a Module subclass")
        self._modules[name] = module

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

# Define basic Module classes - these will be replaced by PyO3 classes when available
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        Module.__init__(self)
        # Placeholder - actual implementation in PyO3
        raise NotImplementedError("Linear layer not available - Pycoeus needs rebuild")

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None):
        Module.__init__(self)
        raise NotImplementedError("Conv1d not available - Pycoeus needs rebuild")

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None):
        Module.__init__(self)
        raise NotImplementedError("Conv2d not available - Pycoeus needs rebuild")

class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None):
        Module.__init__(self)
        raise NotImplementedError("Conv3d not available - Pycoeus needs rebuild")

class ConvTranspose1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, output_padding=None, bias=None):
        Module.__init__(self)
        raise NotImplementedError("ConvTranspose1d not available - Pycoeus needs rebuild")

class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, output_padding=None, bias=None):
        Module.__init__(self)
        raise NotImplementedError("ConvTranspose2d not available - Pycoeus needs rebuild")

class ConvTranspose3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, output_padding=None, bias=None):
        Module.__init__(self)
        raise NotImplementedError("ConvTranspose3d not available - Pycoeus needs rebuild")

class Sequential(Module):
    def __init__(self, *args):
        Module.__init__(self)
        # Note: Sequential.__new__ already initialized the Rust side with *args
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def add_module(self, name, module):
        # Override to avoid calling Sequential.add_module (which is NotImplemented)
        Module.add_module(self, name, module)

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        Module.__init__(self)
        raise NotImplementedError("BatchNorm1d not available - Pycoeus needs rebuild")

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        Module.__init__(self)
        raise NotImplementedError("BatchNorm2d not available - Pycoeus needs rebuild")

class BatchNorm3d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        Module.__init__(self)
        raise NotImplementedError("BatchNorm3d not available - Pycoeus needs rebuild")

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        Module.__init__(self)
        raise NotImplementedError("LayerNorm not available - Pycoeus needs rebuild")

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        Module.__init__(self)
        raise NotImplementedError("GroupNorm not available - Pycoeus needs rebuild")

class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        Module.__init__(self)
        raise NotImplementedError("RMSNorm not available - Pycoeus needs rebuild")

class Dropout(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        raise NotImplementedError("Dropout not available - Pycoeus needs rebuild")

class Dropout2d(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        raise NotImplementedError("Dropout2d not available - Pycoeus needs rebuild")

class Dropout3d(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        raise NotImplementedError("Dropout3d not available - Pycoeus needs rebuild")

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        Module.__init__(self)
        raise NotImplementedError("Embedding not available - Pycoeus needs rebuild")

class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        Module.__init__(self)
        raise NotImplementedError("RNN not available - Pycoeus needs rebuild")

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        Module.__init__(self)
        raise NotImplementedError("LSTM not available - Pycoeus needs rebuild")

class GRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        Module.__init__(self)
        raise NotImplementedError("GRU not available - Pycoeus needs rebuild")

class Bilinear(Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        Module.__init__(self)
        raise NotImplementedError("Bilinear not available - Pycoeus needs rebuild")

# Activations
class ReLU(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("ReLU not available - Pycoeus needs rebuild")

class GELU(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("GELU not available - Pycoeus needs rebuild")

class SiLU(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("SiLU not available - Pycoeus needs rebuild")

class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        Module.__init__(self)
        raise NotImplementedError("PReLU not available - Pycoeus needs rebuild")

class Sigmoid(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("Sigmoid not available - Pycoeus needs rebuild")

class Tanh(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("Tanh not available - Pycoeus needs rebuild")

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        Module.__init__(self)
        raise NotImplementedError("LeakyReLU not available - Pycoeus needs rebuild")

class ELU(Module):
    def __init__(self, alpha=1.0):
        Module.__init__(self)
        raise NotImplementedError("ELU not available - Pycoeus needs rebuild")

class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        Module.__init__(self)
        raise NotImplementedError("Hardtanh not available - Pycoeus needs rebuild")

class Softplus(Module):
    def __init__(self, beta=1, threshold=20):
        Module.__init__(self)
        raise NotImplementedError("Softplus not available - Pycoeus needs rebuild")

class Mish(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("Mish not available - Pycoeus needs rebuild")

class ReLU6(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("ReLU6 not available - Pycoeus needs rebuild")

class SELU(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("SELU not available - Pycoeus needs rebuild")

class Hardsigmoid(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("Hardsigmoid not available - Pycoeus needs rebuild")

class Hardswish(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("Hardswish not available - Pycoeus needs rebuild")

class LogSigmoid(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("LogSigmoid not available - Pycoeus needs rebuild")

# Pooling
class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        Module.__init__(self)
        raise NotImplementedError("MaxPool1d not available - Pycoeus needs rebuild")

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        Module.__init__(self)
        raise NotImplementedError("MaxPool2d not available - Pycoeus needs rebuild")

class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        Module.__init__(self)
        raise NotImplementedError("MaxPool3d not available - Pycoeus needs rebuild")

class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        Module.__init__(self)
        raise NotImplementedError("AvgPool1d not available - Pycoeus needs rebuild")

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        Module.__init__(self)
        raise NotImplementedError("AvgPool2d not available - Pycoeus needs rebuild")

class AvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        Module.__init__(self)
        raise NotImplementedError("AvgPool3d not available - Pycoeus needs rebuild")

class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        Module.__init__(self)
        raise NotImplementedError("AdaptiveAvgPool1d not available - Pycoeus needs rebuild")

class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        Module.__init__(self)
        raise NotImplementedError("AdaptiveAvgPool2d not available - Pycoeus needs rebuild")

class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        Module.__init__(self)
        raise NotImplementedError("AdaptiveAvgPool3d not available - Pycoeus needs rebuild")

class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size):
        Module.__init__(self)
        raise NotImplementedError("AdaptiveMaxPool1d not available - Pycoeus needs rebuild")

class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        Module.__init__(self)
        raise NotImplementedError("AdaptiveMaxPool2d not available - Pycoeus needs rebuild")

class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size):
        Module.__init__(self)
        raise NotImplementedError("AdaptiveMaxPool3d not available - Pycoeus needs rebuild")

# Utility
class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        Module.__init__(self)
        raise NotImplementedError("Flatten not available - Pycoeus needs rebuild")

class Identity(Module):
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("Identity not available - Pycoeus needs rebuild")

class Softmax(Module):
    def __init__(self, dim=-1):
        Module.__init__(self)
        raise NotImplementedError("Softmax not available - Pycoeus needs rebuild")

class LogSoftmax(Module):
    def __init__(self, dim=-1):
        Module.__init__(self)
        raise NotImplementedError("LogSoftmax not available - Pycoeus needs rebuild")

# Loss Functions
class BCEWithLogitsLoss(Module):
    """Binary Cross Entropy with Logits Loss."""
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("BCEWithLogitsLoss not available - Pycoeus needs rebuild")

class BCELoss(Module):
    """Binary Cross Entropy Loss."""
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("BCELoss not available - Pycoeus needs rebuild")

class CrossEntropyLoss(Module):
    """Cross Entropy Loss for classification."""
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("CrossEntropyLoss not available - Pycoeus needs rebuild")

class L1Loss(Module):
    """L1 Loss (Mean Absolute Error)."""
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("L1Loss not available - Pycoeus needs rebuild")

class MSELoss(Module):
    """Mean Squared Error Loss."""
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("MSELoss not available - Pycoeus needs rebuild")

class NLLLoss(Module):
    """Negative Log Likelihood Loss."""
    def __init__(self):
        Module.__init__(self)
        raise NotImplementedError("NLLLoss not available - Pycoeus needs rebuild")

class SmoothL1Loss(Module):
    """Smooth L1 Loss (Huber Loss)."""
    def __init__(self, beta=1.0):
        Module.__init__(self)
        raise NotImplementedError("SmoothL1Loss not available - Pycoeus needs rebuild")

try:
    from .._coeus import Linear as _Linear, LazyLinear as _LazyLinear, Bilinear as _Bilinear
    from .._coeus import ReLU as _ReLU, Sequential as _Sequential
    # Normalization layers
    from .._coeus import BatchNorm1d as _BatchNorm1d, BatchNorm2d as _BatchNorm2d, BatchNorm3d as _BatchNorm3d
    from .._coeus import LayerNorm as _LayerNorm, GroupNorm as _GroupNorm, RMSNorm as _RMSNorm
    from .._coeus import InstanceNorm1d as _InstanceNorm1d, InstanceNorm2d as _InstanceNorm2d, InstanceNorm3d as _InstanceNorm3d
    # Dropout layers  
    from .._coeus import Dropout as _Dropout, Dropout1d as _Dropout1d, Dropout2d as _Dropout2d, Dropout3d as _Dropout3d
    from .._coeus import AlphaDropout as _AlphaDropout
    # Loss functions
    from .._coeus import MSELoss as _MSELoss, CrossEntropyLoss as _CrossEntropyLoss
    from .._coeus import NLLLoss as _NLLLoss, BCEWithLogitsLoss as _BCEWithLogitsLoss
    from .._coeus import L1Loss as _L1Loss, SmoothL1Loss as _SmoothL1Loss, KLDivLoss as _KLDivLoss
    # Embedding
    from .._coeus import Embedding as _Embedding
    # Conv layers
    from .._coeus import Conv1d as _Conv1d, Conv2d as _Conv2d, Conv3d as _Conv3d
    from .._coeus import ConvTranspose1d as _ConvTranspose1d, ConvTranspose2d as _ConvTranspose2d, ConvTranspose3d as _ConvTranspose3d
    from .._coeus import LazyConv1d as _LazyConv1d, LazyConv2d as _LazyConv2d, LazyConv3d as _LazyConv3d
    
    # Utilities
    from .._coeus import Flatten as _Flatten, Identity as _Identity, Bilinear as _Bilinear

    # Activations 
    from .._coeus import Softmax as _Softmax, LogSoftmax as _LogSoftmax
    from .._coeus import GELU as _GELU, SiLU as _SiLU, LeakyReLU as _LeakyReLU, ELU as _ELU
    from .._coeus import PReLU as _PReLU, Hardtanh as _Hardtanh, Softplus as _Softplus, Mish as _Mish
    from .._coeus import ReLU6 as _ReLU6, SELU as _SELU, Hardsigmoid as _Hardsigmoid, Hardswish as _Hardswish
    from .._coeus import LogSigmoid as _LogSigmoid, Softsign as _Softsign, Tanhshrink as _Tanhshrink
    from .._coeus import Threshold as _Threshold, CELU as _CELU, Softmin as _Softmin, Softshrink as _Softshrink
    from .._coeus import Hardshrink as _Hardshrink, GLU as _GLU, RReLU as _RReLU
    
    # Apply to module namespace
    Linear = _Linear
    LazyLinear = _LazyLinear
    Bilinear = _Bilinear
    ReLU = _ReLU
    Sequential = _Sequential
    BatchNorm1d = _BatchNorm1d
    BatchNorm2d = _BatchNorm2d
    BatchNorm3d = _BatchNorm3d
    LayerNorm = _LayerNorm
    GroupNorm = _GroupNorm
    RMSNorm = _RMSNorm
    InstanceNorm1d = _InstanceNorm1d
    InstanceNorm2d = _InstanceNorm2d
    InstanceNorm3d = _InstanceNorm3d
    Dropout = _Dropout
    Dropout1d = _Dropout1d
    Dropout2d = _Dropout2d
    Dropout3d = _Dropout3d
    AlphaDropout = _AlphaDropout
    MSELoss = _MSELoss
    CrossEntropyLoss = _CrossEntropyLoss
    NLLLoss = _NLLLoss
    BCEWithLogitsLoss = _BCEWithLogitsLoss
    L1Loss = _L1Loss
    SmoothL1Loss = _SmoothL1Loss
    KLDivLoss = _KLDivLoss
    Embedding = _Embedding
    Conv1d = _Conv1d
    Conv2d = _Conv2d
    Conv3d = _Conv3d
    ConvTranspose1d = _ConvTranspose1d
    ConvTranspose2d = _ConvTranspose2d
    ConvTranspose3d = _ConvTranspose3d
    LazyConv1d = _LazyConv1d
    LazyConv2d = _LazyConv2d
    LazyConv3d = _LazyConv3d
    Flatten = _Flatten
    Identity = _Identity
    Softmax = _Softmax
    LogSoftmax = _LogSoftmax
    GELU = _GELU
    SiLU = _SiLU
    LeakyReLU = _LeakyReLU
    ELU = _ELU
    PReLU = _PReLU
    Hardtanh = _Hardtanh
    Softplus = _Softplus
    Mish = _Mish
    ReLU6 = _ReLU6
    SELU = _SELU
    Hardsigmoid = _Hardsigmoid
    Hardswish = _Hardswish
    LogSigmoid = _LogSigmoid
    Softsign = _Softsign
    Tanhshrink = _Tanhshrink
    Threshold = _Threshold
    CELU = _CELU
    Softmin = _Softmin
    Softshrink = _Softshrink
    Hardshrink = _Hardshrink
    GLU = _GLU
    RReLU = _RReLU
except ImportError:
    pass  # Keep placeholder classes


# Expose functional submodule
from . import functional


__all__ = [
    "Module",
    "Parameter",
    "Sequential",
    "Linear",
    "LazyLinear",
    "Bilinear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "RMSNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "Embedding",
    "RNN",
    "LSTM",
    "GRU",
    "ReLU",
    "GELU",
    "SiLU",
    "PReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "ELU",
    "Hardtanh",
    "Softplus",
    "Mish",
    "ReLU6",
    "SELU",
    "Hardsigmoid",
    "Hardswish",
    "LogSigmoid",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    # Utility
    "Flatten",
    "Identity",
    "Softmax",
    "LogSoftmax",
    # Loss
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "L1Loss",
    "MSELoss",
    "NLLLoss",
    "SmoothL1Loss",
    "functional",
]


