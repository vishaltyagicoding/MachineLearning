import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score


concrete = pd.read_csv("concrete_data.csv")
# print(concrete.isnull().sum())
# print(concrete.shape)
# print(concrete.sample(5))
# print(concrete.head())
x = concrete.drop(columns=['Strength'])
y = concrete['Strength']
# print(x)


x_train, x_test, y_train, y_test = train_test_split(concrete.drop(columns=['Strength']), concrete['Strength'], test_size=0.1)

print("Before function transformer")
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

accuracy = r2_score(y_test,y_pred)
print(accuracy)

# for col in x_train.columns:
#     plt.figure(figsize=(10,4))
#     plt.subplot(121)
#     sns.distplot(x_train[col])
#     plt.title(col)
#
#     plt.subplot(122)
#     stats.probplot(x_train[col], dist="norm", plot=plt)
#     plt.title(col)
#
#     plt.show()


pt = PowerTransformer()

pt.fit(x_train)
X_train_transformed = pt.transform(x_train)
X_test_transformed = pt.transform(x_test)


print("After function transformer")
lr = LinearRegression()
lr.fit(X_train_transformed, y_train)
y_pred = lr.predict(X_test_transformed)

accuracy = r2_score(y_test,y_pred)
print(accuracy)

df = pd.DataFrame(X_train_transformed, columns=x_train.columns)
for col in df.columns:
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    sns.distplot(df[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(col)

    plt.show()