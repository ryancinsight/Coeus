from .._coeus import (
    SGD as _SGD,
    Adam as _Adam,
    AdamW as _AdamW,
    RMSprop as _RMSprop,
    Adagrad as _Adagrad,
    Adadelta as _Adadelta,
    Adamax as _Adamax,
    NAdam as _NAdam,
    RAdam as _RAdam,
)
from . import lr_scheduler

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = {}
        self.param_groups = []
        
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
            
        for param_group in param_groups:
            Optimizer.add_param_group(self, param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param_group must be a dictionary"
        params = param_group['params']
        if isinstance(params, (list, tuple)):
            param_group['params'] = list(params)
        else:
            param_group['params'] = [params]
            
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
            
        self.param_groups.append(param_group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self):
        raise NotImplementedError

# Wrap Rust optimizers
class SGD(_SGD, Optimizer):
    def __new__(cls, params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        params_list = list(params)
        obj = _SGD.__new__(cls, params_list, lr, momentum, dampening, weight_decay, nesterov)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
            ),
        )
        Optimizer.__init__(self, params_list, defaults)

class Adam(_Adam, Optimizer):
    def __new__(cls, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        params_list = list(params)
        obj = _Adam.__new__(cls, params_list, lr, betas[0], betas[1], eps, weight_decay)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad),
        )
        Optimizer.__init__(self, params_list, defaults)

class AdamW(_AdamW, Optimizer):
    def __new__(cls, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        params_list = list(params)
        obj = _AdamW.__new__(cls, params_list, lr, betas[0], betas[1], eps, weight_decay)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad),
        )
        Optimizer.__init__(self, params_list, defaults)

class RMSprop(_RMSprop, Optimizer):
    def __new__(cls, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        params_list = list(params)
        obj = _RMSprop.__new__(cls, params_list, lr, alpha, eps, weight_decay, momentum, centered)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
            ),
        )
        Optimizer.__init__(self, params_list, defaults)

class Adagrad(_Adagrad, Optimizer):
    def __new__(cls, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, 
                        initial_accumulator_value=initial_accumulator_value, eps=eps)
        params_list = list(params)
        obj = _Adagrad.__new__(cls, params_list, lr, lr_decay, weight_decay, initial_accumulator_value, eps)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
            ),
        )
        Optimizer.__init__(self, params_list, defaults)

# Wrap additional Rust optimizers
class Adadelta(_Adadelta, Optimizer):
    def __new__(cls, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        params_list = list(params)
        obj = _Adadelta.__new__(cls, params_list, lr, rho, eps, weight_decay)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay),
        )
        Optimizer.__init__(self, params_list, defaults)

class Adamax(_Adamax, Optimizer):
    def __new__(cls, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params_list = list(params)
        obj = _Adamax.__new__(cls, params_list, lr, betas[0], betas[1], eps, weight_decay)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        Optimizer.__init__(self, params_list, defaults)

class NAdam(_NAdam, Optimizer):
    def __new__(
        cls,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        momentum_decay=0.004,
        decoupled_weight_decay=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
        )
        params_list = list(params)
        obj = _NAdam.__new__(
            cls,
            params_list,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            momentum_decay,
            decoupled_weight_decay,
        )
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        momentum_decay=0.004,
        decoupled_weight_decay=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
    ):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                momentum_decay=momentum_decay,
                decoupled_weight_decay=decoupled_weight_decay,
                foreach=foreach,
                maximize=maximize,
                capturable=capturable,
                differentiable=differentiable,
            ),
        )
        Optimizer.__init__(self, params_list, defaults)

class RAdam(_RAdam, Optimizer):
    def __new__(cls, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params_list = list(params)
        obj = _RAdam.__new__(cls, params_list, lr, betas[0], betas[1], eps, weight_decay)
        obj._py_params_list = params_list
        obj._py_defaults = defaults
        return obj

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        params_list = getattr(self, "_py_params_list", list(params))
        defaults = getattr(
            self,
            "_py_defaults",
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        Optimizer.__init__(self, params_list, defaults)

# Expose PyTorch-compatible API
__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adamax",
    "NAdam",
    "RAdam",
    "lr_scheduler",
]
