'''
학습률의 크기에 따른 결과비교
'''
import os, sys
sys.path.append(os.pardir)
import numpy as np
from gradient_2d import function_2, gradient_descent

# lr=10, 학습률이 너무 클 때
init_x = np.array([-3.0, 4.0])

gradient_result1, history1 = gradient_descent(function_2, init_x = init_x, lr = 10.0, step_num=100)
print(gradient_result1) # [-2.58983747e+13 -1.29524862e+12]
# 너무 큰 값으로 발산

# lr=1e-10, 학습률이 너무 작을 때
init_x = np.array([-3.0, 4.0])

gradient_result2, history2 = gradient_descent(function_2, init_x = init_x, lr = 1e-10, step_num=100)
print(gradient_result2) # [-2.99999994  3.99999992]
# 거의 갱신되지 않고 끝남.