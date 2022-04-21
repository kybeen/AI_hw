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


# # --------------[ Train데이터와 Test데이터 input feature 축소시키기 (28x28 -> 20x20) ]---------------
# # 축소된 input feature 데이터를 받을 넘파이 배열
# xTrain2 = np.empty((0,400), dtype=np.int64)
# xTest2 = np.empty((0,400), dtype=np.int64)

# for i in range(0,len(xTrain)):
#     a = np.reshape(xTrain[i],(28,28))   # input feature 28x28형대로 변환
#     for j in range(4):  # 배열 요소 삭제 np.delete(배열, 인덱스, axis)
#         a = np.delete(a,0,0)
#         a = np.delete(a,-1,0)
#         a = np.delete(a,0,1)
#         a = np.delete(a,-1,1)
#     a = np.reshape(a,-1)
#     xTrain2 = np.append(xTrain2, np.array([a]), axis=0)
# print(xTrain2[0])
# print(len(xTrain2[0]))
# print(xTrain2[0].shape)

# for i in range(0,len(x_sample)):
#     a = np.reshape(x_sample[i],(28,28))
#     for j in range(4):  # 배열 요소 삭제 np.delete(배열, 인덱스, axis)
#         a = np.delete(a,0,0)
#         a = np.delete(a,-1,0)
#         a = np.delete(a,0,1)
#         a = np.delete(a,-1,1)
#     a = np.reshape(a,-1)
#     xTest2 = np.append(xTest2, np.array([a]), axis=0)
# print(xTest2[0])
# print(len(xTest2[0]))
# print(xTest2[0].shape)
# # --------------------------------------------------------------------------