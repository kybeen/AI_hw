import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from knn import KNN             # knn.py의 KNN 클래스 import

# iris 데이터를 받아옴
iris = load_iris()
X = iris.data                   # input feature ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
y = iris.target                 # output label [0, 1, 2]
y_name = iris.target_names      # label name ['Setosa', 'Versicolor', 'Virginica']

# KNN 객체 생성
knn = KNN()

# KNN 객체에 test와 train용으로 데이터,레이블을 분배해준다.
for i in range(0, 150):
    if( (i+1) % 15 == 0 ):    # 15번째 인덱스마다 나오는 데이터들을 test 데이터로 넣어주고(10개) 나머지 데이터들은(140개) train 데이터로 넣어준다.
        knn.X_test = np.append(knn.X_test, np.array([X[i]]), axis=0)
        knn.y_test = np.append(knn.y_test, y[i])
    else:
        knn.X_train = np.append(knn.X_train, np.array([X[i]]), axis=0)
        knn.y_train = np.append(knn.y_train, y[i])

#result = knn.majority_vote()    # majority vote로 classification
result = knn.weighted_majority_vote()  # weighted majority vote로 classification

# 테스트 데이터를 classification한 결과로 받아온 값을 실제 테스트 데이터의 레이블 값과 비교한 뒤 결과와 정확도를 출력한다.
c = 0   # 정답을 맞춘 횟수 count
for i in range(0, len(result)):
    test_result = y_name[result[i]]
    real_label = y_name[int(knn.y_test[i])] # 해당 테스트 데이터의 실제 결과
    print(f"Test Data Index: {i}  Computed class: {test_result}, True class: {real_label}")
    # classification 결과값과 실제 레이블 값을 비교한 뒤 일치하면 c를 1씩 카운트
    if(test_result == real_label):
        c += 1
print(f"Accuracy : { c / 10 * 100 }%")