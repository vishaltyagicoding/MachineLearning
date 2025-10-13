import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# import breast cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target


# print the first 5 rows of the data
# print(X[:5])
# print(y[:5])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# score = []
# for i in range(1, 20):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred), "for k =", i)
#     score.append(accuracy_score(y_test, y_pred))


# plt.plot(range(1, 20), score)
# plt.xlabel("K")
# plt.ylabel("Accuracy")
# plt.show()


knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred), "for k =", 11)

