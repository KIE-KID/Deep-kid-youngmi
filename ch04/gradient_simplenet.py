'''
신경망에서 기울기 구하기 구현 예
'''
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):  # contructor
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화, 가중치를 [2,3] 사이즈로 랜덤으로 초기화함.

    def predict(self, x):  # 예측을 수행
        return np.dot(x, self.W)

    def loss(self, x, t):  # 손실함수의 값 구함
        z = self.predict(x)
        y = softmax(z)  # 예측 결과의 비중을 확인
        loss = cross_entropy_error(y, t)  # 예측과 정답과의 손실함수

        return loss


x = np.array([0.6, 0.9]) # 입력 데이터
t = np.array([0, 0, 1]) # 정답 레이블

net = simpleNet()
print(net.W) # 가중치 매개변수, 랜덤으로 초기화하므로 매번 다름.
# [[-0.80506062  0.41034176  0.16365316]
#  [ 0.43194153  1.52554446 -1.22960732]]

p = net.predict(x) # 예측수행
print(p)  # [-0.09428899  1.61919508 -1.00845469]

print(np.argmax(p))  # 최대값의 인덱스, 1

l = net.loss(x, t) # 손실함수 계산
print(l)  # 2.852777454165025

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)  # 손실함수의 기울기 - 편미분값

print(dW)
# [[ 0.44386014  0.00680997 -0.45067011]
# [ 0.66579021  0.01021496 -0.67600516]]
