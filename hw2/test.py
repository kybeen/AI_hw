import sys, os
sys.path.append(os.pardir) # 부모 디렉토리에서 import

import numpy as np
from dataset.mnist import load_mnist    # mnist 데이터 load 함수 import

from PIL import Image   # 파이썬 이미지 프로세싱 라이브러리 (pillow package)

# 학습용, 테스트용 데이터 분리
# flatten : 이미지를 1차원 배열로 읽기
# normalize : 0~1 실수로 (false -> 0~255)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

image = x_train[0]
label = t_train[0]

print(label)
print(image.shape)


#------------------[ Data Visualization ]--------------------
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
# image를 unsigned int로

image = image.reshape(28,28)    # 1차원 -> 2차원 (28x28)

print(image.shape)
img_show(image)