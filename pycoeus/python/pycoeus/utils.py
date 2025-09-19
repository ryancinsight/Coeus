"""
Utility functions for PyCoeus

All utilities delegate to Rust implementations.
"""

from typing import Any, Dict, List, Optional, Union
from pycoeus import PyTensor

def save(obj: Any, f: Union[str, Any]) -> None:
    """Save object to file - delegates to Rust."""
    from pycoeus._core import save as rust_save
    rust_save(obj, f)

def load(f: Union[str, Any], map_location: Optional[str] = None) -> Any:
    """Load object from file - delegates to Rust."""
    from pycoeus._core import load as rust_load
    return rust_load(f, map_location)

def get_device() -> str:
    """Get current device - delegates to Rust."""
    from pycoeus._core import get_device
    return get_device()

def set_device(device: str) -> None:
    """Set current device - delegates to Rust."""
    from pycoeus._core import set_device
    set_device(device)

def memory_summary(device: Optional[str] = None) -> str:
    """Get memory summary - delegates to Rust."""
    from pycoeus._core import memory_summary as rust_memory_summary
    return rust_memory_summary(device)

def empty_cache() -> None:
    """Empty GPU cache - delegates to Rust."""
    from pycoeus._core import empty_cache
    empty_cache()

def synchronize(device: Optional[str] = None) -> None:
    """Synchronize device - delegates to Rust."""
    from pycoeus._core import synchronize as rust_synchronize
    rust_synchronize(device)

class TensorDataset:
    """Tensor dataset - delegates to Rust."""
    
    def __init__(self, *tensors: PyTensor):
        from pycoeus._core import TensorDataset as RustTensorDataset
        self._dataset = RustTensorDataset(list(tensors))
    
    def __len__(self) -> int:
        return self._dataset.len()
    
    def __getitem__(self, index: int) -> List[PyTensor]:
        return self._dataset.get_item(index)

class DataLoader:
    """Data loader - delegates to Rust."""
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, 
                 num_workers: int = 0, drop_last: bool = False):
        from pycoeus._core import DataLoader as RustDataLoader
        self._loader = RustDataLoader(dataset, batch_size, shuffle, num_workers, drop_last)
    
    def __iter__(self):
        return self._loader.iter()
    
    def __len__(self) -> int:
        return self._loader.len()

# Export all utilities
__all__ = [
    "save",
    "load", 
    "get_device",
    "set_device",
    "memory_summary",
    "empty_cache",
    "synchronize",
    "TensorDataset",
    "DataLoader",
]