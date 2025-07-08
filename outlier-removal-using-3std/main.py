import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv("placement.csv")
# print(df.head())
# print(df.isnull().sum())


def lg(df):
    x_ = df.drop(columns=["placed"])
    y = df["placed"]
    x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, random_state=42)

    leg = LogisticRegression().fit(x_train, y_train)

    predictions = leg.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(accuracy*100)

# lg(df)

# df['cgpa'].plot(kind="kde")
# df['placement_exam_marks'].plot(kind="kde")
# plt.show()


# print(df['placement_exam_marks'].skew())
# print(df['cgpa'].skew())

# print(x.describe())

highest = df["cgpa"].mean() + 3*df["cgpa"].std()
lowest = df["cgpa"].mean() - 3*df["cgpa"].std()


# print(highest)
# print(lowest)


df = df[(df["cgpa"] <= highest) & (df["cgpa"] >= lowest)]
# print(df["cgpa"].shape)
# print(df["placement_exam_marks"].shape)
# print(df.shape)

# df["cgpa"] = numpy.where(df["cgpa"] > highest, highest, numpy.where(df["cgpa"] < lowest, lowest, df["cgpa"]))

# print(x.describe())

lg(df)



