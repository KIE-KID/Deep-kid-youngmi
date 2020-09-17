#3.2.3 계단함수 구현하기
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)  #bool타입을 int로 변환

X = np.arange(-5.0, 5.0, 0.1)
#-5.0 에서 5.0전까지 0.1간격의 넘파이 배열 생성 -> [-0.5, -4.9 ... 4.9]
Y = step_function(X)
plt.plot(X, Y)      #X, Y를 인수로 그래프 그리기
plt.ylim(-0.1, 1.1) #y축의 범위를 지정
plt.show()          #그래프를 화면에 출력