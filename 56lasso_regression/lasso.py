from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X,y = load_diabetes(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
# Linear Regression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))

# lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(r2_score(y_test,y_pred))

# Ridge Regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
print(r2_score(y_test,y_pred))

# ElasticNet Regression
elastic = ElasticNet(alpha=0.005, l1_ratio=0.9)
elastic.fit(X_train, y_train)
y_pred = elastic.predict(X_test)
print(r2_score(y_test, y_pred))