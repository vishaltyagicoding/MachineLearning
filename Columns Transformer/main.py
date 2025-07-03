import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# df = pd.read_csv("cars_.csv")

# print(df.sample(5))
# print(df.shape)
# print(df["brand"].nunique())
# print(df["brand"].unique())
# counts = df.brand.value_counts()
# threshold = 100
# repl = counts[counts >= threshold].index
# df = df[df.brand.isin(repl)]
# print(df["brand"].nunique())
# print(df["brand"].unique())


# from sklearn.model_selection import train_test_split
#
# x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=["selling_price"]), df["selling_price"], test_size=0.2, random_state=0)

# print(x_train.shape)
# print(x_train.sample(5))

from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer(
#     transformers=[
#         ("tnf1", OneHotEncoder(sparse_output=False, drop="first"), ["brand", "fuel", "owner"]),
#     ],
#     remainder="passthrough"
# )

# print(ct.fit_transform(x_train))
# print(ct.transform(x_test))
# print(ct.fit_transform(x_train).shape)
# print(ct.transform(x_test).shape)


# second file

dff = pd.read_csv("covid_toy.csv")

# print(dff.sample(5))

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
x_train, x_test, y_train, y_test = train_test_split(dff.drop(columns=["has_covid"]), dff["has_covid"], test_size=0.2, random_state=0)

# print(x_train.shape)
# print(x_test.shape)

from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(remainder='passthrough', transformers=[("tnf1", SimpleImputer(), ["fever"]),
                                                                       ("tnf2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
                                                                       ("tnf3", OneHotEncoder(sparse_output=False, drop="first"), ["gender", "city"])
                                                                       ])


# print(transformer.fit_transform(x_train).shape)
# print(transformer.fit_transform(x_test).shape)


