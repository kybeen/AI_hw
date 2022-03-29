import numpy as np

class KNN:
    def __init__(self):
        self.K = 9                    # 가까운 neighbor로 사용할 k개의 개수 설정
        self.X_train = np.empty((0,4))     # 붓꽃 특징 train 데이터
        self.y_train = np.array([])     # 붓꽃 특징 train 데이터들의 label
        self.X_test = np.empty((0,4))      # 붓꽃 특징 test 데이터
        self.y_test = np.array([])      # 붓꽃 특징 test 데이터들의 label
        self.distance = []                  # 두 점 사이의 거리 저장할 리스트
        self.vote = [0, 0, 0]   # 테스트 결과값을 카운트 하기 위한 리스트
        self.result = []                # 테스트 데이터의 결과를 저장할 리스트

    # 두 점 사이의 최단거리 계산 메서드
    def calculate_distance(self, a, b):
        # return np.sqrt(np.sum(np.power(a-b,2)))
        return np.linalg.norm(a-b)

    # 거리가 가장 짧은 k개의 neighbor를 구하는 메서드
    def obtain_knn(self, K, arr):
        temp = [100 for i in range(K)]
        for i in arr:
            max_t = max(temp)
            if(i < max_t):
                temp[temp.index(max_t)] = i
        return temp
    
    # Majority Vote
    def majority_vote(self):
        for i in range( 0, len(self.X_test) ):
            for j in range( 0, len(self.X_train) ):
                self.distance.append( self.calculate_distance( self.X_test[i],self.X_train[j] ) ) # 두 점 사이의 거리 구한 뒤 distance에 저장
            temp = self.obtain_knn(self.K, self.distance) # 거리가 가장 짧은 k개의 neighbor 구하기
            for i in temp:  # neighbor들의 레이블 종류별로 수를 카운트한다.
                if( self.y_train[self.distance.index(i)] == 0 ):
                    self.vote[0] += 1
                elif( self.y_train[self.distance.index(i)] == 1 ):
                    self.vote[1] += 1
                else:
                    self.vote[2] += 1
            self.result.append( self.vote.index(max(self.vote)) )   # 해당 테스트 데이터에 대한 레이블을 result에 저장
            # 다음 테스트 데이터로 넘어가기 전에 vote와 distance 초기화
            self.vote = [0, 0, 0]
            self.distance = []
        return self.result
    
    # def weighted_majority_vote(self):
    #     sadasdasdad