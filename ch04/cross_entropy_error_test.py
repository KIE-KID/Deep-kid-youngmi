'''교차 엔트로피 오차'''
import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7  # 0과 비슷한 작은 값
    return -np.sum(t * np.log(y + delta))  # np.log: 자연로그, y에 0이 들어가면 -inf가 되기 때문에 아주 작은 값을 더해서 방지해줌.


# 배치용 교차 엔트로피 오차
def cross_entropy_error_batch(y, t):
    if y.ndim == 1:  # 1차원 입력이라면, reshape로 형상을 바꿔준다.
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)

    batch_size = y.shape[0]  # 행의 개수, 배치 크기
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # batch크기로 나눠 정규화하여 이미지 1장당 평균의 교차 엔트로피 오차를 계산함.


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 정답 레이블
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # 모델 예측값1
cross1 = cross_entropy_error(np.array(y1), np.array(t))
print(cross1)  # 0.510825457099338

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]  # 모델 예측값1
cross2 = cross_entropy_error(np.array(y2), np.array(t))
print(cross2)  # 2.302584092994546

# 정답 출력값이 1에 가까워질수록 오차는 0에 가까워지고, 정답 출력이 작아질수록 오차는 커진다.
# 첫 번째 결과(loss function)가 더 작기 때문에 정답에 더 가깝다고 판단 할 수 있다.
