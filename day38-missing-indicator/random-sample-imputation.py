import pandas as pd

df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])

# print(df.sample(5))

# fill random value for numeric data
# df["Age"][df["Age"].isnull()] = df["Age"].dropna().sample(df['Age'].isnull().sum()).values
# df["Fare"][df["Fare"].isnull()] = df["Fare"].dropna().sample(df['Fare'].isnull().sum()).values

# fill random value for categorical data

ca = pd.read_csv("../day37-handling-missing-categorical-data/train.csv",usecols=['GarageQual','FireplaceQu', 'SalePrice'])
# print(ca.head())

# print(ca.isnull().sum())
ca["FireplaceQu"][ca["FireplaceQu"].isnull()] = ca["FireplaceQu"].dropna().sample(ca['FireplaceQu'].isnull().sum()).values
ca["GarageQual"][ca["GarageQual"].isnull()] = ca["GarageQual"].dropna().sample(ca['GarageQual'].isnull().sum()).values

print(ca.isnull().sum())