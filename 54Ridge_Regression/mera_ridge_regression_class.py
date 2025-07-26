
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)
# plt.scatter(X,y)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()


class MeraRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.m = None
        self.b = None

    def fit(self, x_train_, y_train_):
        x_train_ = np.insert(x_train_, 0, 1, axis=1)
        I = np.identity(x_train_.shape[1])
        I[0][0] = 0
        result = np.linalg.inv(np.dot(x_train_.T,x_train_) + self.alpha * I).dot(x_train_.T).dot(y_train_)
        self.b = result[0]
        self.m = result[1:]
        print("Intercept:", self.b)
        print("Coefficients:", self.m)

    def predict(self, x_test_):
        y_pred = np.dot(x_test_, self.m) + self.b
        return y_pred
    
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = MeraRidgeRegression(alpha=0)
lr.fit(x_train, y_train)
# y_pred = lr.predict(x_test)
# print("R2 score:", r2_score(y_test, y_pred))

rr = MeraRidgeRegression(alpha=10)
rr.fit(x_train, y_train)

rr1 = MeraRidgeRegression(alpha=100)
rr1.fit(x_train, y_train)


plt.scatter(x_train, y_train, color='blue', label='Training points')
plt.plot(X,lr.predict(X),color='red',label='alpha=0')
plt.plot(X,rr.predict(X),color='green',label='alpha=10')
plt.plot(X,rr1.predict(X),color='orange',label='alpha=100')
plt.legend()
plt.show()


    