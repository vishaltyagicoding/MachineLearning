import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv("train.csv")
x = df.iloc[:,1:]
y = df.iloc[:,0]


def knn_n(x_, y_):
    start = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=0)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    predictions = knn.predict(x_test)
    print(accuracy_score(y_test, predictions)*100)

    end = time.time()
    print(end-start)

# knn_n(x, y)

scaler = StandardScaler()

scale_data = scaler.fit_transform(x)
# for i in range(1, 200):
#     print(i)
#     pca = PCA(n_components=i)
#     x_trans = pca.fit_transform(scale_data)
#     knn_n(x_trans, y)

# 95.47 % accuracy loop time 41


pca = PCA(n_components=229)
x_trans = pca.fit_transform(scale_data)
knn_n(x_trans, y)

print(np.cumsum(pca.explained_variance_ratio_))

# sum_ = 0
# count = 0
# for data in pca.explained_variance_ratio_:
#     sum_ += data*100
#     print(sum_)
#     count += 1
#     if sum_ >= 90:
#         break
#
# print(sum_)
# print(count)

