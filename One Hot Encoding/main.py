import pandas as pd
import numpy as np


df = pd.read_csv("cars.csv")

# print(df.sample(5))
# print(df["brand"].unique())
# print(df["brand"].nunique())


# one hot encoding by the pandas module
# df = pd.get_dummies(df, columns=["brand", "fuel", "owner"], drop_first=True)

# print(df.sample(5))

# one hot encoding by the sklearn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,0:4], df.iloc[:,-1], test_size=0.2, random_state=0)

# print(x_train)
# print(y_train)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe.fit(x_train)

x_train_trans = ohe.fit_transform(x_train[["fuel", "owner"]])
x_test_trans = ohe.transform(x_test[["fuel", "owner"]])

# print(np.hstack((x_train[["brand", "km_driven"]].values, x_train_trans)))

counts = x_train["brand"].value_counts()
# print(counts)

threshold = 100

repl = counts[counts <= 100].index

print(pd.get_dummies(x_train["brand"].replace(repl, "uncommon")))













