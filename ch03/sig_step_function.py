#시그모이드 함수와 계단 함수 비교
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def step_function(x):
    # 참=1, 거짓=0
    return np.array(x > 0, dtype=np.int32) #boolean을 정수(32bit짜리 정수)로 처리

x = np.arange(-5.0, 5.0, 0.1) # [-5.0, -4.9, ..., 4.9] 배열을 생성
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, '--')
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()