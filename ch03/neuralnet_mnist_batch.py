# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test # 학습은 하지 않기떄문에 리턴하지 않음


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f) # 학습된 가중치 매개변수를 읽음, 가중치와 편향 매개변수
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3'] # 가중치
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # 편향

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size): # 100씩 건너뛰면서 루프를 돈다.
    x_batch = x[i:i+batch_size] #100개 데이터를 한번에 가져옴
    y_batch = predict(network, x_batch) # 100개를 한번에 추론(예측)
    p = np.argmax(y_batch, axis=1) # 100개의 추론 결과 배열에서 가장 예측치가 높은 항의 인덱스를  행방향으로 가져옴.
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) # 추론 결과와 test셋의 실제 경과가 같은 것의 횟수를 센다.

print("Accuracy:" + str(float(accuracy_cnt) / len(x     ))) # Accuracy:0.9352

