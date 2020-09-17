#시그모이드 함수
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)      #X, Y를 인수로 그래프 그리기
plt.ylim(-0.1, 1.1) #y축 범위를 지정
plt.show()          #그래프를 화면에 출력
