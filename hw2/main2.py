# 2. input feature 를 자신만의 방식으로 가공해서 차수 줄이기
# MNIST 데이터 10개를 임의로 하나씩 출력해 보았을 때, 상하좌우로 3~4줄 정도는 대부분 공백(0)을 나타내는 무의미한 데이터이기 때문에, 계산에 필요 없다고 생각되어 input feature의 상하좌우 4줄씩을 제거하여 input feature의 차수를 28x28 -> 20x20으로 줄였다. (1차원 : 784 -> 400)

import sys, os
sys.path.append(os.pardir) # 부모 디렉토리에서 import
import numpy as np
from dataset.mnist import load_mnist    # mnist 데이터 load 함수 import
from PIL import Image   # 파이썬 이미지 프로세싱 라이브러리 (pillow package)
from knn import KNN             # knn.py의 KNN 클래스 import

# 학습용, 테스트용 데이터 분리
# flatten : 이미지를 1차원 배열로 읽기
# normalize : 0~1 실수로 (false -> 0~255)
(xTrain, yTrain), (xTest, yTest) = load_mnist(flatten=True, normalize=False)
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # 레이블 이름

# --------[ MNIST 데이터 하나 골라서 모양 출력해보기 ]-------------------------------------------------
# a = np.reshape(xTrain[0],(28,28))
# print(a.shape)
# i_len, j_len = a.shape
# print(i_len)
# print(j_len)
# for i in range(0,i_len):
#     for j in range(0,j_len):
#         print("{:3}".format(a[i][j]), end='  ')
#     print('\n')
# ---------------------------------------------------------------------------------------------

# test데이터 랜덤으로 뽑아서 쓰기
size = 100   # 랜덤으로 뽑을 데이터 개수
sample = np.random.randint(0, yTest.shape[0], size)    # 0부터 10000까지의 정수 중에 랜덤으로 size개 만큼 뽑기

# 랜덤으로 뽑은 데이터를 담을 넘파이 배열
x_sample = np.empty((0,784), dtype=np.int64)
y_sample = np.array([], dtype=np.int64)
for i in sample:
    x_sample = np.append(x_sample, np.array([xTest[i]]), axis=0)
    y_sample = np.append(y_sample, yTest[i])

# --------------[ Train데이터와 Test데이터 input feature 축소시키기 (28x28 -> 20x20) ]---------------
xTrain2 = np.empty((0,400), dtype=np.int64) # 20x20 크기로 변환한 Train데이터를 저장할 리스트
xTest2 = np.empty((0,400), dtype=np.int64)  # 20x20 크기로 변환한 Test데이터를 저장할 리스트

# [ Train 데이터의 input feature 가공 ]
for i in range(0,len(xTrain)):
    a = np.reshape(xTrain[i],(28,28))   # 784x1 형태로 받아왔던 input feature를 다시 28x28형대로 변환
    for j in range(4):  # input feature 리스트의 상하좌우 4줄씩을 삭제한다. np.delete(배열, 인덱스, axis)
        a = np.delete(a,0,0)    # 위 4줄 삭제
        a = np.delete(a,-1,0)   # 아래 4줄 삭제
        a = np.delete(a,0,1)    # 왼쪽 4줄 삭제
        a = np.delete(a,-1,1)   # 오른쪽 4줄 삭제
    a = np.reshape(a,-1)    # 20x20로 가공된 input feature를 다시 1차원으로 변환한다. (400x1 크기)
    xTrain2 = np.append(xTrain2, np.array([a]), axis=0) # 가공된 데이터를 xTrain2 리스트에 새로 담아준다.
# # 데이터 가공 확인용 코드
# print(xTrain2[0])
# print(len(xTrain2[0]))
# print(xTrain2[0].shape)

# [ Test 데이터의 input feature 가공 ]
for i in range(0,len(x_sample)):
    a = np.reshape(x_sample[i],(28,28))
    for j in range(4):
        a = np.delete(a,0,0)
        a = np.delete(a,-1,0)
        a = np.delete(a,0,1)
        a = np.delete(a,-1,1)
    a = np.reshape(a,-1)
    xTest2 = np.append(xTest2, np.array([a]), axis=0)
# print(xTest2[0])
# print(len(xTest2[0]))
# print(xTest2[0].shape)
# -------------------------------------------------------------------------------------------

K = 9   # K 값 설정

knn = KNN(K, xTrain2, yTrain, xTest2, y_sample)    # knn 객체 생성 (input feature 데이터는 위에서 가공된 xTrain2와 xTest2로 받아온다.)

result = knn.weighted_majority_vote()   # weighted_majority_vote를 사용한 knn 알고리즘으로 MNIST classification 수행

c = 0   # 정답 맞춘 횟수
for i in range(size):
    print(f'{sample[i]} th data result {result[i]}   label {yTest[sample[i]]}')
    if(result[i] == yTest[sample[i]]):
        c += 1
print(f'Accuracy = {c / size}')
print(f'10000개 test data 중 {size}개 사용 (K = {K})')
