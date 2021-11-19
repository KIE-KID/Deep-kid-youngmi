'''미니 배치 구현'''
# 훈련 데이터가 6만개로 모든 데이터를 대상으로 손실 함수의 합을 구하려면 시간이 오래 걸린다.
# 그래서 데이터의 일부를 추려서 근사치로 학습에 이용할 수 있다.
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0] # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 60000개 중에서 10개를 랜덤으로 가져오는데, 인덱스를 가져옴.
print(batch_mask) # [28616 40437 20310 35362 39939 57401  1441  7263 55281 33077], 매번 바뀜

x_batch = x_train[batch_mask] # 해당 인덱스를 가지고 있는걸 반환
t_batch = t_train[batch_mask]