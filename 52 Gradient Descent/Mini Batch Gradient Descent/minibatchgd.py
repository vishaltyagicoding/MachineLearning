# MeraMiniBatchGD.py
# Mini-Batch Gradient Descent Implementation
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random


class MeraMiniBatchGD:
    def __init__(self, batch_size):
        self.learning_rate = 0.1
        self.iterations = 100
        self.batch_size = batch_size
        self.m = None
        self.b = None

    def fit(self, X, y):
        rows = X.shape[0]
        cols = X.shape[1]
        self.b = 0
        self.m = np.ones(cols)
        for i in range(self.iterations):
            for j in range(int(rows / self.batch_size)):
              idx = random.sample(range(rows), self.batch_size)
            # print(self.m)
              y_hat = np.dot(X[idx], self.m) + self.b
              # print(y_hat.shape)
              loss_slope_b = -2 * np.mean(y_train[idx] - y_hat)
              # print(loss_slope_b)
              self.b = self.b - (self.learning_rate * loss_slope_b)
              loss_slope_m = -2 * np.dot((y[idx] - y_hat), X[idx])
              # print(loss_slope_m.shape)
              self.m = self.m - (self.learning_rate * loss_slope_m)
        print(self.m)
        print(self.b)

    def predict(self, X):
        y_pred = np.dot(X, self.m) + self.b
        return y_pred

X,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(x_train.shape)
gd = MeraMiniBatchGD(batch_size=32)
gd.fit(x_train, y_train)
prediction = gd.predict(x_test)
print("r2 score")
print(r2_score(y_test, prediction))

# Note: The learning rate and batch size can be adjusted based on the dataset and convergence requirements.
# This implementation uses random sampling for mini-batch selection.
# how to implment mini-batch gradient descent in python sklearn

from sklearn.linear_model import SGDRegressor
def linear(X, y):
    model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.1)
    epoch = 100
    for _ in range(epoch):
        idx = random.sample(range(x_train.shape[0]), 32)
        # Partial fit for mini-batch gradient descent
        model.partial_fit(x_train[idx], y_train[idx])
    
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    predictions = model.predict(x_test)
    print("R2 Score:", r2_score(y_test, predictions))

linear(X, y)