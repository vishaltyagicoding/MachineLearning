import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
# from PCA.exercise import remove_outliers

df = pd.read_csv("ALL_DATA.csv", usecols= ["Open", "High", "Low", "Close", "Volume"])
df["High"] = df["High"].fillna(df["High"].mean())
df["Low"] = df["Low"].fillna(df["Low"].mean())
df["Open"] = df["Open"].fillna(df["Open"].mean())
df["Close"] = df["Close"].fillna(df["Close"].mean())

# print(df.info())
#
# print(df.sample(5))
# print(df.isnull().sum())
# print(df.isnull().mean())
# print(df.shape)


def box_plot(data):
    sns.boxplot(data)
    plt.show()


def kde_plot(data):
    sns.kdeplot(data)
    plt.show()

def prediction(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    lg = LinearRegression()
    lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print("R2: ", r2*100)


def remove_outliers_iqr(df, column):
    """
    Remove outliers from a pandas DataFrame using the IQR method

    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to process

    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    # Calculate Q1, Q3 and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_filtered


power_transformer = PowerTransformer()

trans = power_transformer.fit_transform(df)

df = pd.DataFrame(trans, columns=["Open", "High", "Low", "Close", "Volume"])

kde_plot(df["Volume"])
box_plot(df["Volume"])
for col in list(df.columns):
    df = remove_outliers_iqr(df, col)

kde_plot(df["Volume"])
box_plot(df["Volume"])

x_ = df.drop(columns=["Volume"])
y_ = df["Volume"]
# print(x_)
# print(y_)
prediction(x_, y_)









