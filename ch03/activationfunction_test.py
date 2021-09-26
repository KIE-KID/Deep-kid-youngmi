import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from common.functions import identity_function,step_function, sigmoid, softmax, relu
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict_identity(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = identity_function(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = identity_function(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


def predict_step(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = step_function(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = step_function(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

def predict_sigmoid(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

def predict_relu(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
identity_cnt = 0
step_cnt = 0
sigmoid_cnt = 0
relu_cnt = 0

for i in range(len(x)):
  y = predict_identity(network, x[i])
  p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
  if p == t[i]: # 정답이 맞다면 1 증가
    identity_cnt += 1

for i in range(len(x)):
  y = predict_step(network, x[i])
  p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
  if p == t[i]: # 정답이 맞다면 1 증가
    step_cnt += 1

for i in range(len(x)):
  y = predict_sigmoid(network, x[i])
  p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
  if p == t[i]: # 정답이 맞다면 1 증가
    sigmoid_cnt += 1

for i in range(len(x)):
  y = predict_relu(network, x[i])
  p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
  if p == t[i]: # 정답이 맞다면 1 증가
    relu_cnt += 1

identity_accuracy = float(identity_cnt) / len(x)
step_accuracy = float(step_cnt) / len(x)
sigmoid_accuracy = float(sigmoid_cnt) / len(x)
relu_accuracy = float(relu_cnt) / len(x)

print("Accuracy:" + str(identity_accuracy)) # Accuracy:0.7889
print("Accuracy:" + str(step_accuracy)) # Accuracy:0.9182
print("Accuracy:" + str(sigmoid_accuracy)) # Accuracy:0.9352
print("Accuracy:" + str(relu_accuracy)) # Accuracy:0.8415

# make graph
acc_value = [identity_accuracy, step_accuracy, sigmoid_accuracy, relu_accuracy]
functions = ['Identity function', 'Step function', 'Sigmoid function', 'ReLU function']

plt.figure(figsize=(7,6))
plt.bar(range(len(acc_value)), acc_value)
plt.xticks(range(len(functions)), functions, rotation=10)
plt.ylim(0, 1) #y축의 범위를 지정
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()