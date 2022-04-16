from turtle import distance
import numpy as np

class KNN:
    def __init__(self, K, xTrain, yTrain, xTest, yTest):
        self.K = K                    # 가까운 neighbor로 사용할 k개의 개수 설정
        self.X_train = xTrain     # 손글씨 이미지 train 데이터
        self.y_train = yTrain     # 손글씨 이미지 train 데이터 라벨
        self.X_test = xTest      # 손글씨 이미지 test 데이터
        self.y_test = yTest      # 손글씨 이미지 test 데이터 라벨
        self.distance = []                  # 두 점 사이의 거리 저장할 리스트
        self.vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 테스트 결과값을 카운트 하기 위한 리스트
        self.result = []                # 테스트 데이터의 결과를 저장할 리스트

    # 두 점 사이의 최단거리 계산 메서드
    def calculate_distance(self, a, b):
        #return np.sqrt(np.sum(np.power(a-b,2)))
        return np.linalg.norm(a-b)

    # 거리가 가장 짧은 k개의 neighbor를 구하는 메서드
    def obtain_knn(self, K, arr):
        max_default = max(arr) + 1
        temp = [max_default for i in range(K)]  # k개의 neighbor들과의 최단거리를 저장할 리스트 (초기값은 가장 먼 거리 + 1로 해서 어느 거리와 비교해도 크도록 함)
        pair = [0 for i in range(K)]    # k개의 neighbor들의 레이블을 저장할 리스트 (0으로 초기화)
        for i in arr:
            max_t = max(temp)   # temp에서의 최대값과 비교한 뒤 더 작은 값이면 neighbor값을 갱신한다.
            if(i < max_t):
                pair[temp.index(max_t)] = int(self.y_train[arr.index(i)])    # 더 짧은 neighbor의 레이블로 갱신한다.
                temp[temp.index(max_t)] = i                             # 더 짧은 neighbor와의 거리로 갱신한다.
        return temp, pair

    # Weighted Majority Vote
    def weighted_majority_vote(self):
        for i in range( 0, len(self.X_test) ):
            for j in range( 0, len(self.X_train) ):
                self.distance.append( self.calculate_distance( self.X_test[i],self.X_train[j] ) ) # 두 점 사이의 거리 구한 뒤 distance에 저장
            temp, pair = self.obtain_knn(self.K, self.distance) # 거리가 가장 짧은 k개의 neighbor와 해당하는 레이블을 받아옴
            weight = sum(temp)
            # 단순 수를 세는 것이 아니라 [KNN들의 최단거리 합 - 해당 neighbor와의 최단거리 / KNN들의 최단거리 합]을 가중치로 하여 1에 가중치를 곱한 값들을 해당 vote값에 더해준다.
            for i in range(0, len(temp)):  # neighbor들의 레이블 종류별로 
                self.vote[pair[i]] += 1 * ( (weight-temp[i]) / weight )
            self.result.append( self.vote.index(max(self.vote)) )   # 가중치까지 고려한 vote값이 가장 큰 인덱스 값을 result에 추가헤준다. (테스트결과)
            # 다음 테스트 데이터로 넘어가기 전에 vote와 distance 초기화
            # print(self.vote)
            # print(self.distance[0])
            self.vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.distance = []
        return self.result