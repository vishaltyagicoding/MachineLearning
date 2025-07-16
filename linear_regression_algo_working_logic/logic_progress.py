from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("placement.csv")

# print(df.sample(5))
# print(df.isnull().sum())

# sns.scatterplot(data=df, x="cgpa", y="package")
# plt.show()
x = df.drop(columns=['package'])
y = df["package"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

lg=LinearRegression()
lg.fit(x_train,y_train)

print(lg.coef_)#m
print(lg.intercept_)#b
print(x_test.iloc[0])
print(y_test.iloc[0])
predictions = lg.predict(x_test.iloc[0].values.reshape(-1,1))
print(predictions)

sns.scatterplot(data=df, x="cgpa", y="package")
plt.plot(x_train,lg.predict(x_train),color='red')
plt.show()
