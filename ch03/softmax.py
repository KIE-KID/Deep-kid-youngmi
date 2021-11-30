# softmax 함수
# softmax 출력의 총합은 항상 1 = 확률로 해석가능하다 = 가장 높은 값(확률)을 선택
# softmax 함수 사용해도 원소의 대소관계는 그대로이다. (생략해도 괜찮다.)
# 보통 train은 softmax 사용, predit는 사용X

import numpy as np
import matplotlib.pyplot as plt

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
print(y)

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c) #오버플로우 방지
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

def softmax(x):
    if x.ndim == 2:
        x = x.T # x의 전치행렬
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T # y의  전치행렬

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(np.sum(y))

X = np.arange(-5.0, 5.0, 0.1)
#-5.0 에서 5.0전까지 0.1간격의 넘파이 배열 생성 -> [-0.5, -4.9 ... 4.9]
Y = softmax(X)
plt.plot(X, Y)      #X, Y를 인수로 그래프 그리기
plt.ylim(-0.009, 0.1) #y축의 범위를 지정
plt.yticks([0,0.02,0.04,0.06,0.08,0.1])
plt.show()          #그래프를 화면에 출력