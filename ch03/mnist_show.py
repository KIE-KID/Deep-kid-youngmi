import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint(img))
    pil_img.show()

(x_train, t_train), (x_test, x_test) = load_mnist(flatten=True, normalize=False)  #1차원 배열로 저장, 정규화X

img = x_train[0]          #0번째 훈련 이미지를 가져옴
label = t_train[0]        #0번째 훈련 레이블을 가져옴
print(label)

print(img.shape)          #(784, ) - 1차원 배열 형태
img = img.reshape(28,28)  #원래 이미지의 모양으로 변형
print(img.shape)          #(28, 28)

img_show(img)
