"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            nonlinearity="relu", 
            device=device, 
            dtype=dtype,
            requires_grad=True
        ))

        if bias == True:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(
                fan_in=out_features,
                fan_out=1,
                nonlinearity="relu",
                device=device,
                dtype=dtype,
                requires_grad=True
            )))
        else:
            self.bias = None

        

    def forward(self, X: Tensor) -> Tensor:
        if self.bias == None:
            return ops.matmul(X, self.weight)
        else:
            return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        


class Flatten(Module):
    def forward(self, X):
        B = X.shape[0]
        dims = 1
        for i in range(1, len(X.shape)):
            dims *= X.shape[i]
        return ops.reshape(X, (B, dims))



class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x



class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = init.one_hot(logits.shape[-1], y, device=logits.device)
        zy = ops.summation(ops.multiply(y_one_hot, logits), axes=(-1,))

        log_sum_exp = ops.logsumexp(logits, axes=(-1,))
        return ops.divide_scalar(ops.summation(ops.add(log_sum_exp, ops.negate(zy))), logits.shape[0])
        


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias   = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        B = 1
        for i in range(len(x.shape) - 1):
            B *= x.shape[i]

        shape_bc = lambda t: ops.broadcast_to(ops.reshape(t, (1, x.shape[-1])), x.shape)

        if self.training:
            batch_avg = ops.summation(x, axes=tuple(range(len(x.shape) - 1))) / B
            batch_var = ops.summation(ops.power_scalar(ops.add(x, shape_bc(ops.negate(batch_avg))), 2), axes=tuple(range(len(x.shape) - 1))) / B

            self.running_mean = self.running_mean * (1 - self.momentum) + batch_avg * self.momentum
            self.running_var  = self.running_var  * (1 - self.momentum) + batch_var * self.momentum

            return ops.add(
                ops.multiply(
                    shape_bc(self.weight),
                    ops.divide(
                        ops.add(x, shape_bc(ops.negate(batch_avg))),
                        shape_bc(ops.power_scalar(ops.add_scalar(batch_var, self.eps), 0.5))
                    )
                ),
                shape_bc(self.bias)
            )

        else:
            return ops.add(
                ops.multiply(
                    shape_bc(self.weight),
                    ops.divide(
                        ops.add(x, shape_bc(ops.negate(self.running_mean))), 
                        shape_bc(ops.power_scalar(ops.add_scalar(self.running_var, self.eps), 0.5))
                    )
                ),
                shape_bc(self.bias)
            )
        



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias   = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))  

    def forward(self, x: Tensor) -> Tensor:
        avg = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(-1,))/ x.shape[-1], (*x.shape[:-1], 1)), x.shape)

        var = ops.broadcast_to(ops.reshape(
            ops.summation(ops.power_scalar(ops.add(x, ops.negate(avg)), 2), axes=(-1,)) / x.shape[-1], 
            (*x.shape[:-1], 1)
        ), x.shape)

        # weight * (x - avg) / sqrt(var + eps) + bias
        return ops.add(
            ops.multiply(
                ops.broadcast_to(self.weight, x.shape), 
                ops.divide(
                    ops.add(x, ops.negate(avg)), 
                    ops.power_scalar(ops.add_scalar(var, self.eps), 0.5)
                )
            ), 
            ops.broadcast_to(self.bias, x.shape)
        )



class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
