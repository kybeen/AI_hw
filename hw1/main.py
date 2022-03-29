import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from knn import KNN             # KNN 클래스파일 import

# iris 데이터를 받아옴
iris = load_iris()
#print(iris)
X = iris.data                   # input feature ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
y = iris.target                 # output label [0, 1, 2]
y_name = iris.target_names      # label name ['Setosa', 'Versicolor', 'Virginica']

# KNN 객체 생성
knn = KNN()

# KNN 객체에 test 데이터와 train 데이터를 분배해준다.
for i in range(0, 150):
    if(i > 0 and i % 14 == 0):    # 15의 배수번째 데이터들을 test 데이터로 넣어주고 나머지 데이터들은 train 데이터로 넣어준다.
        knn.X_test = np.append(knn.X_test, np.array([X[i]]), axis=0)
        knn.y_test = np.append(knn.y_test, y[i])
    else:
        knn.X_train = np.append(knn.X_train, np.array([X[i]]), axis=0)
        knn.y_train = np.append(knn.y_train, y[i])

# print(knn.X_test)
# print(knn.y_test)
# print(knn.X_train)
# print(knn.y_train)

#result = knn.majority_vote()
result = knn.weighted_majority_vote()
# print(result)

c = 0
for i in range(0, len(result)):
    test_result = y_name[result[i]]
    test_label = y_name[int(knn.y_test[i])]
    print(f"Test Data Index: {i}  Computed class: {test_result}, True class: {test_label}")
    if(test_result == test_label):
        c += 1
print(f"Accuracy : { c / 10 * 100 }")
