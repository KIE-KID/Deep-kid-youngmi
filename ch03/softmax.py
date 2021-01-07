# softmax 함수
# softmax 출력의 총합은 항상 1 = 확률로 해석가능하다 = 가장 높은 값(확률)을 선택
# softmax 함수 사용해도 원소의 대소관계는 그대로이다. (생략해도 괜찮다.)
# 보통 train은 softmax 사용, predit는 사용X

import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
print(y)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #오버플로우 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(np.sum(y))
