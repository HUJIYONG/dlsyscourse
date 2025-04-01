from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def __init__(self):
        self.axes = (1,)
    def compute(self, Z):
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_exp = array_api.exp(Z - Z_max)
        Z_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True) 
        return Z - Z_max - array_api.log(Z_sum)

    def gradient(self, out_grad, node):
        '''
        Y = log(softmax(Z))
        dZ = dY - sum(dY) * softmax(Z)
        '''
        Z = node.inputs[0]

        shape = []
        if self.axes is None:
            shape = [1] * len(Z.shape)
        else:
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            pos_axes = tuple([i + len(Z.shape) if i < 0 else i for i in self.axes])

            shape = [1 if i in pos_axes else Z.shape[i] for i in range(len(Z.shape))]

        log_sumexp = logsumexp(Z, tuple(self.axes))
        log_softmax = add(Z, negate(broadcast_to(reshape(log_sumexp, tuple(shape)), Z.shape)))
        softmax = exp(log_softmax)

        sum_grad = broadcast_to(reshape(summation(out_grad, tuple(self.axes)), tuple(shape)), Z.shape)

        return (add(out_grad, negate(multiply(softmax, sum_grad))),)


 


def logsoftmax(a):
    return LogSoftmax()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_exp = array_api.exp(Z - Z_max)
        Z_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True)
        return array_api.squeeze(Z_max + array_api.log(Z_sum))

    def gradient(self, out_grad, node):
        Z = node.inputs[0]

        shape = []
        if self.axes is None:
            shape = [1] * len(Z.shape)
        else:
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            pos_axes = tuple([i + len(Z.shape) if i < 0 else i for i in self.axes])

            shape = [1 if i in pos_axes else Z.shape[i] for i in range(len(Z.shape))]

        log_softmax = add(Z, negate(broadcast_to(reshape(node, tuple(shape)), Z.shape)))
        softmax = exp(log_softmax)
        broadcast_out_grad = broadcast_to(reshape(out_grad, tuple(shape)), Z.shape)

        return (multiply(softmax, broadcast_out_grad),)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

