import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 정답은 '2'
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예1: '2'일 확률이 가장 높다고 추정함(0.6)
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
loss1 = mean_squared_error(np.array(y1), np.array(t))
print(loss1)    # 0.09750000000000003

# 예2: '7'일 확률이 가장 높다고 추정함(0.6)
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
loss2 = mean_squared_error(np.array(y2), np.array(t))
print(loss2)    # 0.5975

# 첫 번째 결과(loss function)가 정답 레이블과의 오차가 더 작으니
# 정답에 더 가깝다고 판단 할 수 있다.
