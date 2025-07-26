import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
class MeraLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self,x_train_,y_train_):
        looptime = x_train_.shape[0]
        x_train_mean = x_train_.mean()
        y_train_mean = y_train_.mean()
        numerator = 0
        denominator = 0
        for i in range(looptime):
            numerator += (x_train_[i] - x_train_mean) * (y_train_[i] - y_train_mean)
            denominator += (x_train_[i] - x_train_mean) ** 2

        self.m = numerator / denominator
        self.b = y_train_mean - (self.m * x_train_mean)

        return self.m, self.b



    def predict(self,x_test_):
        prediction = self.m * x_test_ + self.b
        return prediction

df = pd.read_csv("placement.csv")

x = df.iloc[:,0].values
y = df.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lg = MeraLR()

lg.fit(x_train, y_train)
prediction = lg.predict(x_test)
print(prediction)

print(lg.m)
print(lg.b)