import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# first_col = 153441.51, 142107.34 index 2, 4
# sec_col = 151377.59 index 1
# third_col = 471784.1 index 0
# df = pd.read_csv("50_Startups.csv", usecols=['R&D Spend','Administration','Marketing Spend'])

df = pd.read_csv("train.csv", usecols=['Pclass','Age', 'Fare', 'Survived'])
print(df)
# length = df.isnull().index.step
missing_cols_indices = np.where(df.isna().any())[0].tolist()
print(missing_cols_indices)

missing_indices = [df.index[df.iloc[:,x].isna()].tolist() for x in missing_cols_indices]
# print(df.isnull().sum())
# print(missing_indices)


df.fillna(df.mean(), inplace=True)



def iteration():
    xi = 0
    for index_ in missing_cols_indices:
        # Fill NaN at specific indices with a value
        fill_value = np.nan
        df.loc[missing_indices[xi], df.columns[index_]] = fill_value
        xi += 1
        x_train = df.dropna(subset=[df.columns[index_]]).drop(columns=[df.columns[index_]])
        y_train = df.dropna(subset=[df.columns[index_]])[df.columns[index_]]
        x_test = df[df.iloc[:, index_].isna()].drop(columns=[df.columns[index_]])

        lr = LinearRegression()
        lr.fit(x_train, y_train)
        predictions = lr.predict(x_test)

        replacement_values = predictions

        # fill iteration column missing values by
        # the predicted values
        nan_indices = df[df.iloc[:, index_].isna()].index
        for ids, idx in enumerate(nan_indices):
            df.iloc[idx, index_] = replacement_values[ids]

        # print("Predicted values:")
        # print(f"{1+index_} columns")
        # print(predictions)
        # print(df.head())




def predict():
    x = df.drop(columns=["Survived"])
    y = df['Survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lg = LogisticRegression()
    lg.fit(x_train, y_train)

    predictions = lg.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    return accuracy

#
print(df.head(18))
print("Before Method")
print(predict())

for i in range(5):
    iteration()

print("Final data")
print(df.head(18))

print("After Method")
print(predict())















# all code copy

    # df_copy = df.copy()

    # x_train = df.dropna(subset=[df.columns[0]]).drop(columns=[df.columns[0]])
    # y_train = df.dropna(subset=[df.columns[0]])[df.columns[0]]

    # x_train = df_copy.dropna(subset=[df_copy.columns[0]]).iloc[:,1:3]
    # y_train = df_copy.dropna(subset=[df_copy.columns[0]]).iloc[:,0:1]
    # print(y_train)

    # x_train.fillna(x_train.mean(), inplace=True)
    # y_train.fillna(y_train.mean(), inplace=True)

    # print(x_train)
    # print(y_train)

    # x_test = df[df.iloc[:, 0].isna()].drop(columns=[df.columns[0]])

    # print(x_test)

    # lr = LinearRegression()
    # lr.fit(x_train, y_train)
    # predictions = lr.predict(x_test)
    # print(predictions)
    # print("Before")
    # print(df.head())

    # replacement_values = predictions
    #
    # nan_indices = df[df.iloc[:, 0].isna()].index
    # for i, idx in enumerate(nan_indices):
    #     df.iloc[idx, 0] = replacement_values[i]
    #
    # print("After")
    # print(df.head())

    # df_copy = df.copy()
