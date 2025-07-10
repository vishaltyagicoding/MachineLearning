import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# df = pd.read_csv("train.csv")[['Age','Pclass','SibSp','Parch','Survived']]
# # print(df.isnull().sum())
# # print(df.shape)
# df["Age"] = df["Age"].fillna(df["Age"].mean())
#
# print(df.sample(10))
# print(df.isnull().sum())
# def lg():
#     x_ = df.drop(columns=["Survived"])
#     y = df["Survived"]
#     x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, random_state=42)
#
#     leg = LogisticRegression().fit(x_train, y_train)
#
#     predictions = leg.predict(x_test)
#
#     accuracy = accuracy_score(y_test, predictions)
#     print(accuracy*100)
#
# # lg()
#
# df["Family"] = df["SibSp"] + df["Parch"] + 1
# df = df.drop(columns=["SibSp", "Parch"])
# print(df.head())
#
# def myfunc(num):
#     if num == 1:
#         #alone
#         return 0
#     elif 1 < num <= 4:
#         # small family
#         return 1
#     else:
#         # large family
#         return 2
#
# df["Family"] = df["Family"].apply(myfunc)

# lg()
# Feature splitting
df = pd.read_csv("train.csv")
df["Title"] = df["Name"].str.split(",", expand=True)[1].str.split(".", expand=True)[0]
# print(s.value_counts())
# Calculate mean only for numeric columns
print(df.groupby('Title').mean(numeric_only=True)['Survived'].sort_values(ascending=False))