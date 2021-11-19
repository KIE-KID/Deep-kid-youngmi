import numpy as np
import matplotlib.pyplot as plt
'''
수치미분: 아주 작은 차분(근사치)으로 미분하는 것
해석적미분: 수식을 전개해 미분하는 것
'''
# 수치미분 - 나쁜 구현의 예
def numerical_diff1(f, x):
    h = 10e-50 # h에 작은 값을 대입하기 위해 0에 무한히 가깝게 하기위한 값, float 32형으로 나타내면 0.0이됨
    return (f(x+h) - f(x)) /h # 차분의 문제가 있음

# 수치미분 - 개선된 예
def numerical_diff2(f, x):
    h = 1e-4 # 0.0001, 좋은 결과를 얻는 다고 알려진 값
    return (f(x+h) - f(x-h)) / (2*h) # 중심 차분(중앙 차분)계산 한 것

# 함수1 - 수치미분
def function_1(x):
    return 0.01*x**2 + 0.1*x

# 함수2 - 편미분, 식에 변수가 여러 개인 경우, 변수를 선택해서 부분적으로 미분
def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)

# 수식1-x0에 대한 편미분
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

# 수식2-x1에 대한 편미분
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

x = np.arange(0.0, 20.0, 0.1) #0에서 20까지 0.1간격으로 배열 생성
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
#plt.show()

# 함수의 미분계산
# 수치미분
print(numerical_diff2(function_1, 5)) # 0.1999999999990898  해석적미분: 0.2
print(numerical_diff2(function_1, 10)) # 0.2999999999986347  해석적미분: 0.3

# 편미분
print(numerical_diff2(function_tmp1, 3.0)) # 6.00000000000378  해석적미분: 6
print(numerical_diff2(function_tmp2, 4.0)) # 7.999999999999119  해석적미분: 8