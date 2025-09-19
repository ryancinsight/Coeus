"""
Learning rate schedulers for PyCoeus

All schedulers delegate to Rust implementations.
"""

from typing import List, Union, Callable, Optional

class StepLR:
    """Step learning rate scheduler - delegates to Rust."""
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        from pycoeus._core import StepLRScheduler
        self._scheduler = StepLRScheduler(optimizer, step_size, gamma, last_epoch)
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler - delegates to Rust."""
        self._scheduler.step(epoch)
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates - delegates to Rust."""
        return self._scheduler.get_last_lr()

class MultiStepLR:
    """Multi-step learning rate scheduler - delegates to Rust."""
    
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        from pycoeus._core import MultiStepLRScheduler
        self._scheduler = MultiStepLRScheduler(optimizer, milestones, gamma, last_epoch)
    
    def step(self, epoch: Optional[int] = None):
        self._scheduler.step(epoch)
    
    def get_last_lr(self) -> List[float]:
        return self._scheduler.get_last_lr()

class ExponentialLR:
    """Exponential learning rate scheduler - delegates to Rust."""
    
    def __init__(self, optimizer, gamma: float, last_epoch: int = -1):
        from pycoeus._core import ExponentialLRScheduler
        self._scheduler = ExponentialLRScheduler(optimizer, gamma, last_epoch)
    
    def step(self, epoch: Optional[int] = None):
        self._scheduler.step(epoch)
    
    def get_last_lr(self) -> List[float]:
        return self._scheduler.get_last_lr()

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler - delegates to Rust."""
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        from pycoeus._core import CosineAnnealingLRScheduler
        self._scheduler = CosineAnnealingLRScheduler(optimizer, T_max, eta_min, last_epoch)
    
    def step(self, epoch: Optional[int] = None):
        self._scheduler.step(epoch)
    
    def get_last_lr(self) -> List[float]:
        return self._scheduler.get_last_lr()

class ReduceLROnPlateau:
    """Reduce learning rate on plateau scheduler - delegates to Rust."""
    
    def __init__(self, optimizer, mode: str = 'min', factor: float = 0.1, 
                 patience: int = 10, threshold: float = 1e-4, 
                 threshold_mode: str = 'rel', cooldown: int = 0, 
                 min_lr: Union[float, List[float]] = 0, eps: float = 1e-8):
        from pycoeus._core import ReduceLROnPlateauScheduler
        self._scheduler = ReduceLROnPlateauScheduler(
            optimizer, mode, factor, patience, threshold, 
            threshold_mode, cooldown, min_lr, eps
        )
    
    def step(self, metrics: float):
        self._scheduler.step(metrics)
    
    def get_last_lr(self) -> List[float]:
        return self._scheduler.get_last_lr()

class LambdaLR:
    """Lambda learning rate scheduler - delegates to Rust."""
    
    def __init__(self, optimizer, lr_lambda: Union[Callable, List[Callable]], last_epoch: int = -1):
        from pycoeus._core import LambdaLRScheduler
        self._scheduler = LambdaLRScheduler(optimizer, lr_lambda, last_epoch)
    
    def step(self, epoch: Optional[int] = None):
        self._scheduler.step(epoch)
    
    def get_last_lr(self) -> List[float]:
        return self._scheduler.get_last_lr()

# Export all schedulers
__all__ = [
    "StepLR",
    "MultiStepLR", 
    "ExponentialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "LambdaLR",
]