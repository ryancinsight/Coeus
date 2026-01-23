"""
Categorize PyTorch API parity gaps by priority and implementability.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Priority categories
CRITICAL = "Critical"  # Core functionality needed for basic usage
IMPORTANT = "Important"  # Commonly used features
OPTIONAL = "Optional"  # Rarely used or specialized features

# Implementability categories
IMPLEMENTABLE = "Implementable"  # Can be implemented with current architecture
ARCHITECTURAL = "Architectural"  # Requires architectural changes (e.g., JIT, distributed)
INTERNAL = "Internal"  # Internal/private APIs not needed for users

def categorize_item(item: str) -> Tuple[str, str, str]:
    """
    Categorize a missing item by priority and implementability.
    Returns: (priority, implementability, reason)
    """
    
    # Internal/Private APIs
    if any(x in item for x in [
        "PRIVATE_OPS", "TYPE_CHECKING", "USE_GLOBAL_DEPS", "USE_RTLD_GLOBAL",
        "CallStack", "Code", "CompilationUnit", "Graph", "Node", "Value",
        "DeepCopyMemoTable", "SerializationStorageContext", "DeserializationStorageContext",
        "DispatchKey", "DispatchKeySet", "ExcludeDispatchKeyGuard",
        "TracingState", "ErrorReport", "FatalError", "FileCheck",
        "LoggerBase", "LockingLogger", "NoopLogger",
        "PythonFileReader", "PythonFileWriter",
        "classproperty", "builtins", "functools", "importlib", "inspect",
        "glob", "os", "platform", "sys", "threading", "textwrap", "types",
        "ctypes", "math", "warnings", "library", "overrides",
    ]):
        return (OPTIONAL, INTERNAL, "Internal API not needed for user-facing functionality")
    
    # JIT Compilation (Architectural)
    if any(x in item for x in [
        "jit", "JIT", "Script", "Trace", "compile", "compiler",
        "TorchScript", "Graph", "Node", "Block", "FunctionSchema",
    ]):
        return (IMPORTANT, ARCHITECTURAL, "Requires JIT compilation infrastructure")
    
    # Distributed Training (Architectural)
    if any(x in item for x in [
        "distributed", "RRef", "futures", "multiprocessing",
        "DataParallel", "parallel",
    ]):
        return (IMPORTANT, ARCHITECTURAL, "Requires distributed training infrastructure")
    
    # Quantization (Architectural - complex)
    if any(x in item for x in [
        "quantiz", "QInt", "QUInt", "qint", "quint", "qscheme",
        "fake_quantize", "q_per_channel", "q_scale", "q_zero_point",
        "int_repr", "fbgemm", "qat", "quantizable",
    ]):
        return (IMPORTANT, ARCHITECTURAL, "Requires quantization infrastructure")
    
    # Storage types (Architectural - already have storage abstraction)
    if any(x in item for x in [
        "Storage", "TypedStorage", "UntypedStorage",
        "BFloat16Storage", "BoolStorage", "ByteStorage", "CharStorage",
        "ComplexDoubleStorage", "ComplexFloatStorage", "DoubleStorage",
        "FloatStorage", "HalfStorage", "IntStorage", "LongStorage",
        "ShortStorage", "QInt32Storage", "QInt8Storage",
    ]):
        return (OPTIONAL, ARCHITECTURAL, "Storage abstraction already exists in Coeus")
    
    # Type system (Architectural)
    if any(x in item for x in [
        "Type", "AnyType", "BoolType", "ComplexType", "FloatType",
        "IntType", "NumberType", "StringType", "TensorType", "TupleType",
        "ListType", "DictType", "OptionalType", "UnionType", "FutureType",
        "ClassType", "InterfaceType", "EnumType", "DeviceObjType",
        "StreamObjType", "PyObjectType", "SymBoolType", "SymIntType",
        "InferredType", "ConcreteModuleType",
    ]):
        return (OPTIONAL, ARCHITECTURAL, "Type system for JIT/tracing")
    
    # Symbolic shapes (Architectural)
    if any(x in item for x in [
        "Sym", "sym_", "SymBool", "SymFloat", "SymInt",
    ]):
        return (OPTIONAL, ARCHITECTURAL, "Symbolic shape inference for compilation")
    
    # Backend-specific (Architectural - some backends)
    if any(x in item for x in [
        "cuda", "cudnn", "mps", "xpu", "xla", "ipu", "mtia", "maia",
        "vulkan", "mkldnn", "miopen", "accelerator",
    ]):
        # CUDA is critical, others are optional
        if "cuda" in item.lower():
            return (CRITICAL, IMPLEMENTABLE, "CUDA backend support needed")
        return (OPTIONAL, ARCHITECTURAL, "Backend-specific functionality")
    
    # Sparse tensors (Implementable with current storage abstraction)
    if any(x in item for x in [
        "sparse", "Sparse", "csr", "csc", "coo", "bsr", "bsc",
        "sparse_dim", "dense_dim", "coalesce", "is_coalesced",
        "ccol_indices", "col_indices", "crow_indices", "row_indices",
        "values", "indices",
    ]):
        return (IMPORTANT, IMPLEMENTABLE, "Sparse tensor operations - storage abstraction exists")
    
    # Core tensor operations (Critical)
    core_ops = [
        "abs", "add", "sub", "mul", "div", "matmul", "mm", "bmm",
        "transpose", "reshape", "view", "squeeze", "unsqueeze",
        "cat", "stack", "split", "chunk",
        "sum", "mean", "std", "var", "max", "min",
        "exp", "log", "sqrt", "pow", "sin", "cos", "tan",
        "relu", "sigmoid", "tanh", "softmax", "log_softmax",
    ]
    if any(op in item.lower() for op in core_ops):
        return (CRITICAL, IMPLEMENTABLE, "Core tensor operation")
    
    # NN layers (Critical/Important)
    if item.startswith("nn."):
        layer_name = item.replace("nn.", "")
        
        # Critical layers
        critical_layers = [
            "Linear", "Conv1d", "Conv2d", "Conv3d",
            "BatchNorm1d", "BatchNorm2d", "LayerNorm",
            "ReLU", "Sigmoid", "Tanh", "Softmax",
            "Dropout", "Embedding",
            "LSTM", "GRU", "RNN",
            "MSELoss", "CrossEntropyLoss", "BCELoss",
            "Module", "Parameter", "Sequential",
        ]
        if any(layer in layer_name for layer in critical_layers):
            return (CRITICAL, IMPLEMENTABLE, "Core neural network layer")
        
        # Important layers
        important_layers = [
            "MaxPool", "AvgPool", "AdaptiveAvgPool", "AdaptiveMaxPool",
            "ConvTranspose", "Upsample",
            "GroupNorm", "InstanceNorm",
            "LeakyReLU", "ELU", "GELU", "SiLU",
            "Attention", "Transformer",
            "L1Loss", "NLLLoss", "KLDivLoss",
        ]
        if any(layer in layer_name for layer in important_layers):
            return (IMPORTANT, IMPLEMENTABLE, "Commonly used neural network layer")
        
        # Functional operations
        if "functional" in item:
            return (IMPORTANT, IMPLEMENTABLE, "Functional API for neural network operations")
        
        # Utils and helpers
        if any(x in item for x in ["utils", "init", "parameter", "modules"]):
            return (IMPORTANT, IMPLEMENTABLE, "Neural network utilities")
        
        # Specialized layers
        return (OPTIONAL, IMPLEMENTABLE, "Specialized neural network layer")
    
    # Optimizers (Important)
    if item.startswith("optim."):
        opt_name = item.replace("optim.", "")
        
        # Critical optimizers
        if any(x in opt_name for x in ["Adam", "SGD", "Optimizer"]):
            return (CRITICAL, IMPLEMENTABLE, "Core optimizer")
        
        # Learning rate schedulers
        if "lr_scheduler" in opt_name:
            return (IMPORTANT, IMPLEMENTABLE, "Learning rate scheduler")
        
        # Other optimizers
        return (IMPORTANT, IMPLEMENTABLE, "Additional optimizer")
    
    # Autograd (Critical)
    if any(x in item for x in ["autograd", "grad", "backward", "Gradient"]):
        return (CRITICAL, IMPLEMENTABLE, "Automatic differentiation functionality")
    
    # Tensor creation (Critical)
    if any(x in item for x in [
        "zeros", "ones", "empty", "full", "arange", "linspace",
        "rand", "randn", "randint", "randperm",
        "eye", "tensor", "as_tensor", "from_numpy",
    ]):
        return (CRITICAL, IMPLEMENTABLE, "Tensor creation function")
    
    # Linear algebra (Important)
    if any(x in item for x in [
        "linalg", "svd", "eig", "qr", "cholesky", "lu",
        "det", "logdet", "slogdet", "inverse", "pinverse",
        "norm", "matrix_power", "matrix_exp",
    ]):
        return (IMPORTANT, IMPLEMENTABLE, "Linear algebra operation")
    
    # FFT (Important)
    if "fft" in item.lower():
        return (IMPORTANT, IMPLEMENTABLE, "Fast Fourier Transform operation")
    
    # Signal processing (Optional)
    if "signal" in item.lower() or "stft" in item.lower() or "istft" in item.lower():
        return (OPTIONAL, IMPLEMENTABLE, "Signal processing operation")
    
    # Special functions (Optional)
    if any(x in item for x in [
        "special", "gamma", "digamma", "polygamma", "erf", "erfc",
        "lgamma", "mvlgamma", "i0", "igamma", "igammac",
    ]):
        return (OPTIONAL, IMPLEMENTABLE, "Special mathematical function")
    
    # Distributions (Optional)
    if "distributions" in item:
        return (OPTIONAL, IMPLEMENTABLE, "Probability distribution")
    
    # Hub (Optional)
    if "hub" in item:
        return (OPTIONAL, ARCHITECTURAL, "Model hub functionality")
    
    # Profiling (Optional)
    if "profil" in item.lower() or "benchmark" in item.lower():
        return (OPTIONAL, IMPLEMENTABLE, "Profiling and benchmarking tools")
    
    # Testing utilities (Optional)
    if "testing" in item:
        return (OPTIONAL, IMPLEMENTABLE, "Testing utilities")
    
    # AMP (Automatic Mixed Precision) (Important)
    if "amp" in item.lower() or "autocast" in item.lower() or "GradScaler" in item:
        return (IMPORTANT, IMPLEMENTABLE, "Automatic mixed precision training")
    
    # Utils (Optional)
    if "utils" in item:
        return (OPTIONAL, IMPLEMENTABLE, "Utility functions")
    
    # Default: Optional and Implementable
    return (OPTIONAL, IMPLEMENTABLE, "Additional functionality")


def parse_missing_file(filepath: str) -> List[str]:
    """Parse the missing items file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Handle escaped newlines
    content = content.replace('\\n', '\n')
    
    # Split by newlines and filter empty
    items = [line.strip() for line in content.split('\n') if line.strip()]
    return items


def categorize_all_items(items: List[str]) -> Dict:
    """Categorize all missing items."""
    categorized = {
        CRITICAL: {IMPLEMENTABLE: [], ARCHITECTURAL: [], INTERNAL: []},
        IMPORTANT: {IMPLEMENTABLE: [], ARCHITECTURAL: [], INTERNAL: []},
        OPTIONAL: {IMPLEMENTABLE: [], ARCHITECTURAL: [], INTERNAL: []},
    }
    
    for item in items:
        priority, implementability, reason = categorize_item(item)
        categorized[priority][implementability].append((item, reason))
    
    return categorized


def generate_report(categorized: Dict, output_file: str):
    """Generate a categorized report."""
    with open(output_file, 'w') as f:
        f.write("# PyTorch API Parity Gap Analysis\n\n")
        f.write("This report categorizes missing PyTorch functionality by priority and implementability.\n\n")
        
        # Summary statistics
        total = sum(
            len(items)
            for priority_dict in categorized.values()
            for items in priority_dict.values()
        )
        
        f.write("## Summary Statistics\n\n")
        f.write(f"**Total Missing Items:** {total}\n\n")
        
        for priority in [CRITICAL, IMPORTANT, OPTIONAL]:
            priority_total = sum(len(items) for items in categorized[priority].values())
            f.write(f"**{priority}:** {priority_total}\n")
            for impl in [IMPLEMENTABLE, ARCHITECTURAL, INTERNAL]:
                count = len(categorized[priority][impl])
                f.write(f"  - {impl}: {count}\n")
        
        f.write("\n---\n\n")
        
        # Detailed breakdown
        for priority in [CRITICAL, IMPORTANT, OPTIONAL]:
            f.write(f"## {priority} Priority\n\n")
            
            for impl in [IMPLEMENTABLE, ARCHITECTURAL, INTERNAL]:
                items = categorized[priority][impl]
                if not items:
                    continue
                
                f.write(f"### {impl} ({len(items)} items)\n\n")
                
                # Group by reason
                by_reason = defaultdict(list)
                for item, reason in items:
                    by_reason[reason].append(item)
                
                for reason, item_list in sorted(by_reason.items()):
                    f.write(f"#### {reason}\n\n")
                    for item in sorted(item_list)[:50]:  # Limit to 50 per reason
                        f.write(f"- `{item}`\n")
                    if len(item_list) > 50:
                        f.write(f"- ... and {len(item_list) - 50} more\n")
                    f.write("\n")
            
            f.write("---\n\n")


def main():
    print("Parsing missing items...")
    items = parse_missing_file("comparison_missing.txt")
    print(f"Found {len(items)} missing items")
    
    print("Categorizing items...")
    categorized = categorize_all_items(items)
    
    print("Generating report...")
    generate_report(categorized, ".kiro/specs/coeus-architecture-enhancement/PARITY_CATEGORIZATION.md")
    
    print("\nCategorization Summary:")
    for priority in [CRITICAL, IMPORTANT, OPTIONAL]:
        priority_total = sum(len(items) for items in categorized[priority].values())
        print(f"\n{priority}: {priority_total}")
        for impl in [IMPLEMENTABLE, ARCHITECTURAL, INTERNAL]:
            count = len(categorized[priority][impl])
            print(f"  {impl}: {count}")
    
    print("\nReport saved to: .kiro/specs/coeus-architecture-enhancement/PARITY_CATEGORIZATION.md")


if __name__ == "__main__":
    main()
