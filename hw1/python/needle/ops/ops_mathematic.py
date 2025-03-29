"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return (multiply(out_grad, b), multiply(out_grad, a))


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (mul_scalar(out_grad, self.scalar),) 


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b
        
    def gradient(self, out_grad, node):
        a, b = node.inputs
        a_grad = multiply(multiply(out_grad, b), power(a, add_scalar(b, - 1)))
        b_grad = multiply(multiply(out_grad, log(a)), power(a, b))
        return (a_grad, b_grad)

        

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return multiply(mul_scalar(out_grad, self.scalar), power_scalar(a, self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""
    EPSILON = 1e-8

    def compute(self, a, b):
        return a / (b + self.EPSILON)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return (
            divide(out_grad, add_scalar(b, self.EPSILON)),
            divide(
                negate(multiply(out_grad, a)),
                add_scalar(power_scalar(b, 2), self.EPSILON)                
            )            
        )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (divide_scalar(out_grad, self.scalar),)



def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if axes is not None else (-2, -1)

    def compute(self, a):
        return array_api.swapaxes(a, *(self.axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes) 


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return reshape(out_grad, a.shape)



def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        expanded_axes = []
        for i in range(-1, -len(self.shape) - 1, -1):
            if i < -len(a.shape):
                expanded_axes.append(i + len(self.shape))
            elif self.shape[i] != a.shape[i]:
                expanded_axes.append(i + len(self.shape))
        
        return reshape(summation(out_grad, tuple(expanded_axes)), a.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes 

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        shape = []
        if self.axes is None:
            shape = [1] * len(a.shape)
        else:
            shape = [1 if i in self.axes else a.shape[i] for i in range(len(a.shape))]
        return broadcast_to(reshape(out_grad, tuple(shape)), a.shape)



def summation(a, axes=None):
    return Summation(axes)(a)



class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        a_grad = matmul(out_grad, transpose(b))
        b_grad = matmul(transpose(a), out_grad)
        if len(a_grad.shape) != len(a.shape):
            a_grad = summation(a_grad, tuple(range(len(a_grad.shape) - len(a.shape))))
        if len(b_grad.shape) != len(b.shape):
            b_grad = summation(b_grad, tuple(range(len(b_grad.shape) - len(b.shape))))
        return a_grad, b_grad



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (negate(out_grad),)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    EPSILON = 1e-8
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (divide(out_grad, add_scalar(a, self.EPSILON)),)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (multiply(out_grad, exp(a)),)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        # relu_mask = a > 0
        relu_mask = Tensor(
            array_api.greater(a.realize_cached_data(), 0).astype(a.realize_cached_data().dtype),
        )
        return (multiply(out_grad, relu_mask),)
        


def relu(a):
    return ReLU()(a)

