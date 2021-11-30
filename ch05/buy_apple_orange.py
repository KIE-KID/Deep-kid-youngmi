#3. 사과와 오랜지를 구입하는 예제의 순전파와 역전파 구현
from ch05.layer_naive import *

apple = 100 # 사과 가격
apple_num = 2 # 사과 개수
orange = 150 # 오렌지 가격
orange_num = 3 # 오렌지 개수
tax = 1.1 # 소비세

# 계층들 초기화
mul_apple_layer = MulLayer() # 사과 가격 x 사과 개수
mul_orange_layer = MulLayer() # 오렌지 가격 x 오렌지 개수
add_apple_orange_layer = AddLayer() # 덧셈계층
mul_tax_layer = MulLayer() # 가격 x 소비세

#순전파
apple_price = mul_apple_layer.forward(apple,apple_num) # (1)
orange_prcie = mul_orange_layer.forward(orange, orange_num) # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_prcie) # (3)
price = mul_tax_layer.forward(all_price, tax) # (4)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) # (3)
dapple, dapple_num = mul_apple_layer.backward(dapple_price) # (2)
dorange, dorange_num = mul_orange_layer.backward(dorange_price) # (1)

print("price:", int(price)) # price: 715
print("dApple:", dapple) # dApple: 2.2
print("dApple_num:", int(dapple_num)) # dApple_num: 110
print("dOrange:", dorange) # dOrange: 3.3000000000000003
print("dOrange_num:", int(dorange_num)) # dOrange_num: 165
print("dTax:", dtax) # dTax: 650
