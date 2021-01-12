import numpy as np
# 4장 신경망 학습 ReLU
from common.functions import *


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

# 4장 신경망 학습 sigmoid
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


# 5장 오차역전파법 어파인 계층
# 텐서 대응을 위한 함수로 교재에 있는 예제와 차이가 있음.
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.sKhape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) # 열방햡으로 합계

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

# softmax with loss
class SoftmaxWithLoss:
    def __init__(self): # 학습 파라미터 없음
        self.loss = None    # 손실함수
        self.y = None       # softmax의 출력
        self.t = None       # 정답레이블(원-핫 인코딩 형태)

    #순전파 - 데이터, 정답레이블
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) # 확률분포로 변환
        self.loss = cross_entropy_error(self.y, self.t) # 손실함수 값

        return self.loss

    def backward(self, dout=1): # 미분값 1
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태라면
            dx = (self.y - self.t) / batch_size  # 배치처리, 평균
        else: # 원-핫 인코딩이 아니라면
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx/batch_size

        return dx