import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../understanding_data/Titanic-Dataset.csv")
# print(df.shape)

# print(df["Survived"].value_counts())

# TODO 1 count plot
# df["Survived"].value_counts().plot(kind="bar")
# sns.countplot(df, x="Survived")
# sns.countplot(df, x="Sex")
# sns.countplot(df, x="Embarked")
# sns.countplot(df, x="Pclass")
# plt.show()


# TODO 2 pie plot
# df["Survived"].value_counts().plot(kind="pie", autopct="%1.2f%%")
# plt.show()

# TODO 3 Numerical data
# histogram

# plt.hist(df["Age"])
# plt.show()

# TODO 4 displot
# sns.displot(df["Age"], kind="hist", kde=True)  # Can also use kind="kde" for just density
# sns.histplot(df["Age"], kde=True)  # Can also use kind="kde" for just density
# plt.show()

# TODO 5 Boxplot

# sns.boxplot(df, x="Fare")
# plt.show()

# TODO 6 skew

# print(df["Age"].skew())


# d = df[df["Fare"].max() == df["Fare"]]

# for data in sorted(df["Name"]):
#     print(data)


# print(d["Name"])