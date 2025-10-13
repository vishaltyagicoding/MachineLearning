import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv('linear-regression-assumptions\\data.csv')

X = df.drop(['target'], axis=1)
y = df['target']
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TODO 1. Linear Relationship

# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 2.5))

# ax1.scatter(df['feature1'], df['target'])
# ax1.set_title("Feature1")
# ax2.scatter(df['feature2'], df['target'])
# ax2.set_title("Feature2")
# ax3.scatter(df['feature3'], df['target'])
# ax3.set_title("Feature3")

# plt.show()


#TODO 2. Multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = []

for i in range(X_train.shape[1]):
    vif.append(variance_inflation_factor(X_train, i))

data = pd.DataFrame(vif, index=X.columns).T

# print(data)

# TODO 3. Normality of Residual
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residual = y_test - y_pred
# sns.displot(residual,kind='kde')
# plt.show()

# TODO 4. Homoscedasticity
# plt.scatter(y_pred, residual)
# plt.show()

# TODO 5. No Autocorrelation of Residuals
plt.plot(residual)
plt.show()


