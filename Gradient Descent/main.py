import numpy as np
class MeraGradientDescentRegressor:
    def __init__(self):
        self.learning_rate = 0.001
        self.iterations = 50
        self.b = 0
        self.m = 0

    def fit(self, X, y):
        # calcualte the b using GD
        for i in range(self.iterations):
            loss_slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())

            self.b = self.b - (self.learning_rate * loss_slope_b)
            self.m = self.m - (self.learning_rate * loss_slope_m)
        print(self.m, self.b)

    def predict(self, X):
        yPred = self.m * X + self.b
        return yPred



from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

insurance_df = pd.read_csv("insurance.csv")

from sklearn.datasets import make_regression
X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lrg = LinearRegression()
lrg.fit(x_train, y_train)
y_pred = lrg.predict(x_test)
print(r2_score(y_test, y_pred))
print(lrg.coef_)
print(lrg.intercept_)

mera = MeraGradientDescentRegressor()
mera.fit(x_train, y_train)
y_pred = mera.predict(x_test)
print(r2_score(y_test, y_pred))


# print(insurance_df.head())
#
# print(insurance_df.isnull().sum())
# print(insurance_df.shape)
# print(insurance_df.info())
# print(insurance_df["region"].nunique())
# print(insurance_df["sex"].nunique())
# print(insurance_df["smoker"].nunique())

categorical_col = ["sex", "smoker", "region"]
numerical_col = ["age", "bmi", "children", "charges"]



process = ColumnTransformer(remainder="passthrough",
                            transformers=[
                                ("Std", StandardScaler(), numerical_col),
                                ("Ohe", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_col)

                                          ]
                            )



df  = process.fit_transform(insurance_df)
# print(df.shape)

df = pd.DataFrame(df, columns=process.get_feature_names_out())
# print(df.head())
x = df.drop(columns = ["Std__charges"])
y = df["Std__charges"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(x_train.head())
# print(y_train.head())
def pre():
    lrg = LinearRegression()
    lrg.fit(x_train, y_train)
    y_pred = lrg.predict(x_test)
    print(r2_score(y_test, y_pred))
    print(lrg.coef_)
    print(lrg.intercept_)






