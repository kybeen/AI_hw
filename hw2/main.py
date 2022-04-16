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

# test데이터 랜덤으로 뽑아서 쓰기
size = 100   # 100
sample = np.random.randint(0, yTest.shape[0], size)    # 0부터 10000까지의 정수 중에 랜덤으로 100개 뽑기

# 랜덤으로 뽑은 데이터를 담을 넘파이 배열
x_sample = np.empty((0,784), dtype=np.int64)
y_sample = np.array([], dtype=np.int64)
for i in sample:
    x_sample = np.append(x_sample, np.array([xTest[i]]), axis=0)
    y_sample = np.append(y_sample, yTest[i])

K = 9   # K 값 설정

knn = KNN(K, xTrain, yTrain, x_sample, y_sample)    # knn 객체 생성

result = knn.weighted_majority_vote()   # weighted_majority_vote를 사용한 knn 알고리즘으로 MNIST classification 수행

c = 0   # 정답 맞춘 횟수
for i in range(size):
    print(f'{sample[i]} th data result {result[i]}   label {yTest[sample[i]]}')
    if(result[i] == yTest[sample[i]]):
        c += 1
print(f'Accuracy = {c / size}')
print(f'10000개 test data 중 {size}개 사용 (K = {K})')
