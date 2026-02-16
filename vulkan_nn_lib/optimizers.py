import numpy as np
from . import kernels as K

class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.zero_grad()

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def step(self):
        for p in self.params:
            if p.grad is not None:
                K.k_scale_backward(p.grad.arr, -self.lr, p.arr, p.total_size)

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros(p.shape) for p in self.params]
        self.v = [np.zeros(p.shape) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is not None:
                g = p.grad.to_numpy()
                self.m[i] = b1 * self.m[i] + (1 - b1) * g
                self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
                m_hat = self.m[i] / (1 - b1**self.t)
                v_hat = self.v[i] / (1 - b2**self.t)
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                p.arr.from_numpy((p.to_numpy() - update).astype(np.float32).flatten())
