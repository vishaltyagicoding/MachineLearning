import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])

# print(df.isnull().sum())

x = df.drop(columns=['Survived'])
y = df['Survived']





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


imputer = SimpleImputer(add_indicator=True)
imputer.fit(x_train)
x_train_ = imputer.transform(x_train)
x_test_ = imputer.transform(x_test)
print(x_train_)
lg = LogisticRegression()
lg.fit(x_train_, y_train)
pred = lg.predict(x_test_)

acc = accuracy_score(y_test, pred)
print(acc)