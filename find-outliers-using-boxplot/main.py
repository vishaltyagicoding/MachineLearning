import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 8.808933625397168
# 5.113546374602832
df = pd.read_csv("placement.csv")

# print(df.head())
# print(df.isnull().sum())

def outlier_detection(column_values):
    cgpa_25_percentile = np.percentile(column_values, 25)
    cgpa_75_percentile = np.percentile(column_values, 75)

    # print(cgpa_25_percentile)
    # print(cgpa_75_percentile)
    #
    # print(df["cgpa"].describe())

    cgpa_iqr = cgpa_75_percentile - cgpa_25_percentile
    # print(cgpa_iqr)

    higher_value = cgpa_75_percentile + (cgpa_iqr * 1.5)
    lower_value = cgpa_25_percentile - (cgpa_iqr * 1.5)

    return higher_value, lower_value




return_value1 = outlier_detection(df["cgpa"])
return_value2 = outlier_detection(df["placement_exam_marks"])


# triming value from dataframe

# df[df["cgpa"] <= return_value1[0] & df["cgpa"] >= return_value1[1]]
# df = df[(df["cgpa"] <= return_value1[0]) & (df["cgpa"] >= return_value1[1])]
# df = df[(df["placement_exam_marks"] <= return_value2[0]) & (df["placement_exam_marks"] >= return_value2[1])]
# print(df)

# caping

# df["cgpa"] = np.where(df["cgpa"] > return_value1[0],return_value1[0], np.where(df["cgpa"] < return_value1[1], return_value1[1], df["cgpa"]))


# using percentile

upper_limit = df["cgpa"].quantile(0.99)
lower_limit = df["cgpa"].quantile(0.01)
# print(upper_limit, lower_limit)

# df = df[(df["cgpa"] <= upper_limit) & (df["cgpa"] >= lower_limit)]

# df["cgpa"] = np.where(df["cgpa"] > upper_limit,upper_limit, np.where(df["cgpa"] < lower_limit, lower_limit, df["cgpa"]))

# print(df.shape)

def lg():
    x_ = df.drop(columns=["placed"])
    y = df["placed"]
    x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, random_state=42)

    leg = LogisticRegression().fit(x_train, y_train)

    predictions = leg.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(accuracy*100)

lg()