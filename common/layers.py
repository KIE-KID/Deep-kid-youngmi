import numpy as np
# ReLU 함수
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0) # x가 0 이하인 인덱스는 True, 0보다 큰 원소는 False
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

#sigmoid
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out # y, 순전파의 출력을 보관해뒀다가 역전파 계산 때 사용

        return out

    def backward(self, dout):
        dx = dout * self.out * (1-self.out)

        return dx