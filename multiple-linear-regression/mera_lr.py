import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class MeraLR:
    def __init__(self):
        self.intercept = None
        self.coef_ = None

    def _fit_(self, x_train_, y_train_):
        x_train_ = np.insert(x_train_, 0, 1, axis=1)
        betas = np.linalg.inv(np.dot(x_train_.T, x_train_)).dot(x_train_.T).dot(y_train_)

        self.intercept = betas[0]
        self.coef_ = betas[1:]


    def predict(self, x_test_):
        return np.dot(x_test_, self.coef_) + self.intercept







X,y = load_diabetes(return_X_y=True)
# print(X)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MeraLR()
model._fit_(x_train, y_train)

y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))


lgr = LinearRegression()
lgr.fit(x_train, y_train)
y_pred = lgr.predict(x_test)
print(r2_score(y_test, y_pred))

