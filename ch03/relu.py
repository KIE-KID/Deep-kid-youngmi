#ReLU 함수
import numpy as np
import matplotlib.pylab as plt

#입력이 0을 넘으면 그 입력을 그대로 출력
#0이하이면 0을 출력
def relu(x):
    return np.maximum(0,x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1, 5.5)
plt.show()
