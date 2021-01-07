## 3장 신경망
* 신경망에서는 활설화 함수로 시그모이드 함수와 ReLU 함수 같은 매끄럽게 변화하는 함수를 이용한다.
* 넘파이 다차원 배열을 잘 사용하면 신경망을 효율적으로 구현할 수 있다.
* 기계학습 문제는 크게 회귀와 분류로 나눌 수 있다.
* 출력층의 활성화 함수로는 회귀에서는 주로 항등 함수를, 분류에서는 주로 소프트맥스 함수를 이용한다.
* 분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수와 같게 설정한다.
* 입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 이 배치 단위로 진행하면 결과를 훨씬 빠르게 얻을 수 있다.

## 실행 결과
### 1. 계단 함수의 그래프 - step_function.py
<img src='https://user-images.githubusercontent.com/53163222/93405954-70a1fa80-f8c9-11ea-9f3a-c38db7594e4f.png' width='50%'>

### 2. 시그모이드 함수의 그래프 - sigmoid.py
<img src='https://user-images.githubusercontent.com/53163222/93405957-726bbe00-f8c9-11ea-840f-8941296c18e7.png' width='50%'>

### 3. 시그모이드 함수와 계단 함수 비교 그래프 - sig_step_function.py
<img src='https://user-images.githubusercontent.com/53163222/93642227-ac0f0700-fa38-11ea-8cd4-734fadc25536.png' width='50%'>

### 4. 렐루 함수의 그래프 - relu.py
<img src='https://user-images.githubusercontent.com/53163222/93851573-56ae5080-fceb-11ea-92d5-9d5be95aca0b.png' width='50%'>
