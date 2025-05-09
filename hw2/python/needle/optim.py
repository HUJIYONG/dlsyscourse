"""Optimization module"""
import needle as ndl
import numpy as np
import needle.init as init


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.requires_grad:
                if self.u.get(id(p)) is None:
                    self.u[id(p)] = init.zeros(*p.shape, device=p.device)

                grad_with_weight_decay = p.grad.data + self.weight_decay * p.data

                self.u[id(p)].data = self.u[id(p)].data * self.momentum + grad_with_weight_decay.data * (1 - self.momentum)

                p.data = p.data - self.lr * self.u[id(p)].data



    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.requires_grad:
                if self.m.get(id(p)) is None:
                    self.m[id(p)] = init.zeros(*p.shape, device=p.device)
                if self.v.get(id(p)) is None:
                    self.v[id(p)] = init.zeros(*p.shape, device=p.device)

                grad_with_weight_decay = p.grad.data + self.weight_decay * p.data

                self.m[id(p)].data = self.beta1 * self.m[id(p)].data + (1 - self.beta1) * grad_with_weight_decay.data
                self.v[id(p)].data = self.beta2 * self.v[id(p)].data + (1 - self.beta2) * grad_with_weight_decay.data ** 2

                m_hat = self.m[id(p)].data / (1 - self.beta1 ** self.t)
                v_hat = self.v[id(p)].data / (1 - self.beta2 ** self.t)

                p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)


 


