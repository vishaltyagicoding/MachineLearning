import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score


titanic = sns.load_dataset("titanic")[['survived', 'age', 'fare']]
# print(titanic.head())
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())
print(titanic.isnull().sum())


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
#
# sns.distplot(titanic["age"], ax=ax1, kde=True)
#
# # For your age column
# sns.distplot(titanic["fare"], ax=ax2, kde=True)
#
# plt.show()




x_train, x_test, y_train, y_test = train_test_split(titanic.drop(columns=['survived']), titanic['survived'], test_size=0.2)
# print(x_train)

# Before function transformer
print("Before function transformer")
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

dl = DecisionTreeClassifier()
dl.fit(x_train, y_train)

pred = dl.predict(x_test)
print(accuracy_score(y_test,pred))

# After function transformer

print("After function transformer")
# ft = FunctionTransformer(func=np.log1p)
ft = PowerTransformer()

ft.fit(x_train)

x_train_transform = ft.transform(x_train)
x_test_transform = ft.transform(x_test)


lr = LogisticRegression()
lr.fit(x_train_transform, y_train)
y_pred = lr.predict(x_test_transform)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

dl = DecisionTreeClassifier()
dl.fit(x_train_transform, y_train)

pred = dl.predict(x_test_transform)
print(accuracy_score(y_test,pred))

df = pd.DataFrame(x_train_transform, columns=x_train.columns)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

sns.distplot(df["age"], ax=ax1, kde=True)

# For your age column
sns.distplot(df["fare"], ax=ax2, kde=True)

plt.show()






