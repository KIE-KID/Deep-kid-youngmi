import numpy as np

# 확률적 경사 하강법 (Stochastic Gradient Descent)
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
