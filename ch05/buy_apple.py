#곱셈 계층
#사과 2개를 구입하는 예제의 순전파와 역전파 구현
from ch05.layer_naive import *

apple = 100 # 사과 가격
apple_num = 2 # 사과 개수
tax = 1.1 # 소비세

# 계층들 초기화
mul_apple_layer = MulLayer() # 사과 x 사과개수
mul_tax_layer = MulLayer() # 사과가격 x 소비세

# 순전파
apple_price = mul_apple_layer.forward(apple,apple_num) # 사과가격 = 사과, 사과개수
price = mul_tax_layer.forward(apple_price, tax) # 전체가격 = 사과가격, 소비세

print(price) # 220

#역전파
dprice = 1 #입력신호
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax) # 2.2, 110, 200