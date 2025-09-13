from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

# Create synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize Bagging Regressor
bagging_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=20),
    bootstrap=True, max_samples=0.75, n_estimators=200
)

# Train the Bagging Regressor
bagging_reg.fit(X_train, y_train)

# Make predictions
y_pred = bagging_reg.predict(X_test)

# Evaluate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.4f}")

# Checking cross validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(bagging_reg, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# checking best parameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.25, 0.5, 0.75],
    'bootstrap': [True, False],
    'estimator__max_depth': [None, 10, 20],
}

# grid_search = GridSearchCV(estimator=bagging_reg, param_grid=param_grid, cv=3, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
