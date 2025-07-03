import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("cars.csv")
# print(df.isnull().sum())

# print(df.sample(5))
# print(df.head())
# print(df.info())

# print(df.nunique())

# print(df["brand"].value_counts())
# counts = df["brand"].value_counts()

# dd = counts[counts >= 100].index
# print(dd.value_counts().sum())


dependent = df["selling_price"]




from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False,drop="first", dtype=np.int32)
ohe.fit(df[["brand"]])

brand = ohe.transform(df[["brand"]])
# print(brand)
# print(brand.shape)


std = StandardScaler()

std.fit(df[["km_driven"]])
km_driven = std.transform(df[["km_driven"]])

# print(km_driven)

ohe.fit(df[["fuel"]])

fuel = ohe.transform(df[["fuel"]])

# print(fuel)

ohe.fit(df[["owner"]])

owner = ohe.transform(df[["owner"]])

# print(owner)

cars = np.hstack((brand, km_driven, fuel, owner))

# print(cars.shape)
independents = cars

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(independents, dependent, test_size=0.2)


# print(x_train)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

predictions = logreg.predict(x_test)
print(predictions)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(accuracy*100)