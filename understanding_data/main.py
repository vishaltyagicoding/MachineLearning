import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv", usecols=["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", ])

# TODO 1. How big is data?
# print(df.shape)
# print(df.sample(5))

# TODO. 2. How does data look like?
# print(df.sample(5))

# TODO. 3. What is the data type of col?
# print(df.info())

# TODO. 4.Are there any missing values?
# print(df.isnull().sum())

# TODO. 5.How does the data look mathematically?
"""
The describe() function in pandas provides a quick statistical summary of your DataFrame or Series.
 It's one of the most useful functions for exploratory data analysis.
 like count, mean, min, std, max etc.
"""
# print(df.describe())

# TODO. 6.Are there any duplicate values?
# print(df.duplicated().sum())

# TODO. 7.How is the correlation between columns?
print(df.corr()['Survived'])

