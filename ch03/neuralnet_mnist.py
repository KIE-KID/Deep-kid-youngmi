#신경망으로 손글씨 숫자 그림을 추론 
#입력층, 은닉층1, 은닉층2, 출력층의 뉴런 수는 각각 784, 50, 100, 10입니다.
#784=28*28
import sys, os
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten=True, one_hot_label=False) #normalization: 정규화, flatten: 평탄화(1차원배열로 변환), one_hot_label: 원-핫 인코딩 형태 저장여부 false일때는 숫자 형태로 저장, true일때는 원-핫 인코딩
  return x_test,t_test

def init_network():
  with open("sample_weight.pkl",'rb') as f:
    network = pickle.load(f)
  return network

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3'] # 가중치
  b1, b2, b3 = network['b1'], network['b2'], network['b3'] # 편향

  a1 = np.dot(x,W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1,W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2,W3) + b3
  y = softmax(a3)

  return y

x, t = get_data()
print(x.shape) # (10000, 784)
print(x[0].shape) # (784,)
network = init_network()
accuracy_cnt = 0

for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
  if p == t[i]: # 정답이 맞다면 1 증가
    accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) # Accuracy:0.9352
# nomalize를 하지 않으면 accuracy는 0.9207이며 overflow가 발생한다.