import pandas as pd

df = pd.read_csv("customer.csv")

# print(df.sample(5))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,2:4], df.iloc[:,-1], test_size=0.2, random_state=0)

# print(x_train)
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(categories=[["Poor", "Average", "Good"], ["School", "UG", "PG"]])
ordinal_encoder.fit(x_train)

transformed_x_train = ordinal_encoder.transform(x_train)
transformed_x_test = ordinal_encoder.transform(x_test)

# print(x_train)
# print(transformed_x_train)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)
tr = le.transform(y_train)
tt = le.transform(y_test)

print(tr)
print(y_train)