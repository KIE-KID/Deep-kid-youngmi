#신경망의 기원이 되는 알고리즘
#다수의 신호를 입력으로 받아 하나의 신호를 출력
#뉴런신호(x₁, x₂), 가중치(w₁, w₂), 임계값(θ)
import numpy as np

#AND
def AND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

#편향을 도입힌 AND, θ = -b
def AND2(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7  # 편향
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#NAND
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


#multi-layer perceptron
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND2(s1,s2)
    return y

print('AND')
print(AND1(0,0))     #0출력
print(AND1(1,0))     #0출력
print(AND1(0,1))     #0출력
print(AND1(1,1),'\n')#1출력

print('XOR')
print(XOR(0,0))     #0출력
print(XOR(1,0))     #1출력
print(XOR(0,1))     #1출력
print(XOR(1,1))     #0출력
