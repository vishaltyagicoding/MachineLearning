import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df  = pd.read_csv('titanic_toy.csv')
# print(df.sample(5))

# find percentage value of missing data

# print(df.isnull().mean())

# plot age column kde

# df["Age"].plot(kind="kde")
# plt.show()
#
# df["Fare"].plot(kind="kde")
# plt.show()

# print(df.describe())
# print(df.var())
# print(df.cov())
# print(df.corr())

df["Age"] = df["Age"].fillna(99)
df["Fare"] = df["Fare"].fillna(-1)

# df["Age"].plot(kind="kde")
# plt.show()

# df["Fare"].plot(kind="kde")
# plt.show()



# print(df.var())
# print(df.cov())
# print(df.corr())