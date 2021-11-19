'''
미니배치 학습 구현
미니배치 학습: 훈련 데이터 중 일부를 무작위로 꺼내고(미니배치)-100개에 대해서 경사법으로 매개변수를 갱신.
여기서 TwoLayerNet함수를 사용한다. MNIST 데이터 셋 사용하여 학습 수행
'''
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 600000, 훈련 데이터 개수
batch_size = 100  # 미니배치 크기, 임의로 100개의 데이터(이미지와 레이블)을 추출한다.
learning_rate = 0.1 # 학습률

train_loss_list = [] # 학습 에러값
train_acc_list = [] # 학습데이터 정확도
test_acc_list = [] # 테스트데이터 정확도

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1) # 600번, 에폭 한번 학습데이터를 전부 다 한것

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size) # 랜덤 인덱스 100개를 얻음
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 너무 느려서 오차역 전파법으로 대체
    grad = network.gradient(x_batch, t_batch) #5장 오차역 전파법
    # 100개의 미니배치를 대상으로 확률적 경사 하강법 수행

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        # 학습률만큼 곱해서 params에 부호를 바꿔서 넣음. 기울기가 작아지는 방향으로 갱신된다.

    # 학습 경과 기록, 갱신할 때 마다 훈련데이터에 대한 손실함수를 계산하고 배열에 추가
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산, 600번마다 한번씩
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(str(i) + " train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기- 훈련데이터와 시험 데이터에 대한 정확도 추이
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show() # 손실 함수 값의 추이 그래프를 얻을 수 있다.
# 학습 횟수가 늘어날 수록 accuracy가 증가하고 있음을 알 수 있다.
# 실선: 훈련데이터에 대한 정확도, 점선: 시험 데이터에 대한 정확도, 오버피팅이 일어나지 않음

'''
수치미분 직접 구하는 것은 오래걸림
오차역 전파법 결과
0 train acc, test acc | 0.10441666666666667, 0.1028
600 train acc, test acc | 0.7802333333333333, 0.7823
1200 train acc, test acc | 0.87715, 0.8808
1800 train acc, test acc | 0.8981, 0.9021
2400 train acc, test acc | 0.9067666666666667, 0.9101
3000 train acc, test acc | 0.9138, 0.9167
3600 train acc, test acc | 0.9189333333333334, 0.9209
4200 train acc, test acc | 0.92345, 0.9246
4800 train acc, test acc | 0.9274166666666667, 0.9282
5400 train acc, test acc | 0.93095, 0.9332
6000 train acc, test acc | 0.93395, 0.9349
6600 train acc, test acc | 0.937, 0.9378
7200 train acc, test acc | 0.9388166666666666, 0.9391
7800 train acc, test acc | 0.9409833333333333, 0.9415
8400 train acc, test acc | 0.9429666666666666, 0.9421
9000 train acc, test acc | 0.9453666666666667, 0.9442
9600 train acc, test acc | 0.9471166666666667, 0.9463
'''