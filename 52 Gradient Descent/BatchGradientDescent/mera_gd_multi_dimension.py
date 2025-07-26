import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class GD:
    def __init__(self):
        self.learning_rate = 0.5
        self.iterations = 1000
        self.m = None
        self.b = None

    def fit(self, X, y):
        rows = X.shape[0]
        cols = X.shape[1]
        self.b = 0
        self.m = [1 for _ in range(cols)]
        for i in range(self.iterations):
            # print(self.m)
            y_hat = np.dot(X, self.m) + self.b
            # print(y_hat.shape)
            loss_slope_b = (-2 * np.sum(y - y_hat)) / rows
            # print(loss_slope_b)
            self.b = self.b - (self.learning_rate * loss_slope_b)
            loss_slope_m = -2 * np.dot((y - y_hat), X) / rows
            # print(loss_slope_m.shape)
            self.m = self.m - (self.learning_rate * loss_slope_m)
        print(self.m)
        print(self.b)

    def predict(self, X):
        y_pred = np.dot(X, self.m) + self.b
        return y_pred

X,y = load_diabetes(return_X_y=True)
# print(X.shape)
# print(y.shape)
import linear_algo
linear_algo.linear(X,  y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(x_train.shape)






gd = GD()
gd.fit(x_train, y_train)

prediction = gd.predict(x_test)
print("r2 score")
print(r2_score(y_test, prediction))


class GDRegressor:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        # init your coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # update all the coef and the intercept
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            # print("Shape of y_hat",y_hat.shape)
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)

            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)

        print(self.intercept_, self.coef_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_


gdr = GDRegressor(epochs=1000, learning_rate=0.5)
gdr.fit(x_train, y_train)

y_pred = gdr.predict(x_test)
print(r2_score(y_test,y_pred))



