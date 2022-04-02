import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

iris = load_iris()
#print(iris)

X = iris.data                   # input feature ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
y = iris.target                 # output label [0, 1, 2]
#y_name = iris.target_names      # label name ['Setosa', 'Versicolor', 'Virginica']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=30)
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print("clf.score")
print("(pred")






# # 2D로 plotting
# x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k')
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)

# # 3D로 plotting
# fig = plt.figure(figsize=(8,6))
# ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])

# plt.show()