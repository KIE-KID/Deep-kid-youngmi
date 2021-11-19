'''
신경망 학습 알고리즘 구현
1. 2층 신경망 클래스 구현
손글씨 숫자 학습을 위한 모델, 은닉층이 1개인 네트워크, MNIST 데이터셋을 사용
'''
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet: # 클래스 정의

    # 초기화를 수행하는 함수, 인수:입력층의 뉴런 수, 은닉층의 뉴런수, 출력층의 뉴런수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {} # 딕셔너리 - 신경망 매개변수를 보관하는 변수
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 1번째 층 가중치, 정규분포를 갖는 랜덤 weight
        self.params['b1'] = np.zeros(hidden_size) # 1번쨰 층 편항 - 은닉층 개수만큼
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 2번째 층 가중치,
        self.params['b2'] = np.zeros(output_size) # 2번째 층 편향 - 출력층 개수만큼

    def predict(self, x): # 순전파, 예측을 수행, 인수 x: 이미지 데이터
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1) # 활성화 함수
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2) # 예측, 활성화 함수, 0~9까지 예측 비중
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t): # 손실함수를 계산해주는 함수, 교차 엔트로피 에러 계산
        y = self.predict(x)
        
        return cross_entropy_error(y, t) # 예측 값과 정답간의 cee 계산
    
    def accuracy(self, x, t): # 정확도를 계산해주는 함수
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 행 기준 가장 큰 값의 인덱스, 정답이라고 예측한 값의 인덱스
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0]) # 같은 개수 / 배치 행의 개수
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):  # 수치 미분 함수, 가중치 매개변수의 기울기를 구한다.
        loss_W = lambda W: self.loss(x, t)
        
        grads = {} # 기울기를 저장하는 딕셔너리
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 1번째 층의 가중치의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 1번째 층의 편향의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 2번째 층의 가중치의 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) # 2번째 층의 편향의 기울기
        
        return grads # params 변수에 대응하는 각 매개변수의 기울기가 저장됨.
        
    def gradient(self, x, t): # 기울기 갱신, 위의 numerical gradient 대신 오차역 전파법 이용.
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0] # 배치 개수
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2) # 추론값
        
        # backward 5장 내용, 역전파-미분
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy) # 기울기 갱신
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
