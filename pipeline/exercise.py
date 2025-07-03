import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

# TODO 1. Read csv file from relative path
df = pd.read_csv("train.csv")

# TODO 2. get some idea about data frame(data set) columns
# print(df.sample(5))
# print(df.head())
# print(df.describe())

# TODO 3. Find useful columns
df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"], inplace=True)

# TODO 4. split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=["Survived"]), df["Survived"], test_size=0.2)

# TODO 5. chack there is any missing values in columns

print(x_train.isnull().sum())


# TODO 6. if there is null values then fill missing values

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

tf1 = ColumnTransformer([("imputer", SimpleImputer(), [2]),
                         ("imputer_", SimpleImputer(strategy="most_frequent"), [6])], remainder="passthrough")




# TODO 7. convert string columns
from sklearn.preprocessing import OneHotEncoder
tf2 = ColumnTransformer([("ohe", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1,6])], remainder="passthrough")

# TODO 8. scale the age and fare columns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

tf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])
# TODO 0. Import model training algorithm

from sklearn.tree import DecisionTreeClassifier
tf5 = DecisionTreeClassifier()

# TODO 10. create pipe line

from sklearn.pipeline import  Pipeline
pipeline_ = Pipeline([("tf1", tf1),
                      ("tf2", tf2),
                      ("tf3", tf3),
                      ("tf5", tf5)
                      ])

# print(pipeline_.steps)
pipeline_.fit(x_train, y_train)
y_pred = pipeline_.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))



