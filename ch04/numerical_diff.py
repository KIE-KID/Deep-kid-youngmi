import numpy as np
import matplotlib.pyplot as plt
# 수치미분 - 나쁜 구현의 예

def numerical_diff1(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) /h

# 수치미분 - 개선된 예
def numerical_diff2(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 함수1 - 수치미분
def function_1(x):
    return 0.01*x**2 + 0.1*x

# 함수2 - 편미분
def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

# 수치미분
print(numerical_diff2(function_1, 5)) # 0.1999999999990898  해석적미분: 0.2
print(numerical_diff2(function_1, 10)) # 0.2999999999986347  해석적미분: 0.3

# 편미분
print(numerical_diff2(function_tmp1, 3.0)) # 6.00000000000378  해석적미분: 6
print(numerical_diff2(function_tmp2, 4.0)) # 7.999999999999119  해석적미분: 8