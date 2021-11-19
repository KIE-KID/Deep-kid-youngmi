# 수치미분으로 기울기 구하기 예시
import numpy as np
import matplotlib.pylab as plt

# 수치미분, 중앙차분
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 함수
def function_1(x):
    return 0.01*x**2 + 0.1*x 

#접선 그리는 함수
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)  # 미분결과 0.1999999999990898
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1) # 0에서 20까지 0.1간격으로 배열 생성
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
