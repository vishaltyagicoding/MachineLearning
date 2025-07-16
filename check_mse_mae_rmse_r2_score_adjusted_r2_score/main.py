import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("placement.csv")

x = df.iloc[:,0].values.reshape(-1, 1)
y = df.iloc[:,1].values.reshape(-1, 1)

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LinearRegression()

lr.fit(x_train, y_train)

prediction = lr.predict(x_test)

print(mean_squared_error(y_test,prediction))
print(mean_absolute_error(y_test,prediction))
print(np.sqrt(mean_squared_error(y_test,prediction)))

r2 = r2_score(y_test,prediction)
print("r2")
print(r2)
# adjusted r2


# n = numbers of rows
# k = independent columns

n = x_test.shape[0]
k = x_test.shape[1]
print(n)
print(k)


ad_r2 = 1 - ((1-r2)*(n-1)/(n-1-k))
print("Adjusted r2")
print(ad_r2)







