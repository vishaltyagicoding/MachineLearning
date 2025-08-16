import pandas as pd
import numpy as np

# Load the dataset
df =  pd.read_csv("Machine_Learning_Models\\heart.csv")

# Prepare the features and target variable(independent and dependent variables)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Identify categorical and numerical columns
cat_columns = X.select_dtypes(include=['object']).columns.tolist()
num_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# print(X[num_columns] .isin([0]).mean())

# replace 0 values in specific columns with nan
for col in ['Cholesterol', 'RestingBP']:
    X[col] = X[col].replace(0, np.nan)

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# column transformer help to apply specific transformations to specific columns
# One Hot Encoding for categorical columns
# Scaler for numerical columns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# import staking classifiers
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier


# create a column transformer

setep1 = Pipeline(steps=[
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

setep2 = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("impute", KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')),
    
])

process = ColumnTransformer(transformers=[("cat", setep1, cat_columns),
                                          ("num", setep2, num_columns)],
                             remainder='passthrough')

model1 = Pipeline(steps=[("step1",process),
                        # ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("linear_model", LogisticRegression(max_iter=500, penalty="l2", solver='saga', C=1)
                         )])

model2 = Pipeline(steps=[("step1",process),
                        ("model_knn", KN(n_neighbors=13, weights='distance', algorithm='auto', leaf_size=30,metric="manhattan"  ,p=1)
                         )])

model3 = Pipeline(steps=[("step1",process),
                        ("model_svc", SVC(kernel="poly", C=1, gamma='auto', probability=True, degree=3, coef0=1, shrinking=True, tol=0.001, cache_size=2000, class_weight=None)
                         )])


model4 = Pipeline(steps=[("step1",process),
                         ("model_nb", GaussianNB(var_smoothing=1e-09)
                         )])

# train all the model
for model in [model1, model2, model3, model4]:
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # print all models name
    print("Model name:", model.steps[-1][0])
    # Print the accuracy of the model
    print(f"Accuracy of {accuracy_score(y_test, y_pred)}")
    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores.mean() * 100:.2f}%")


# staking classifier
# Stacking Classifier combines the predictions of multiple base models and uses a final estimator to make the final prediction.
# Here, we will use the four models we created above as base models and a KNN classifier as the final estimator.
staking = StackingClassifier(estimators=[
    ('model1', model1),
    ('model2', model2),
    ('model3', model3),
    ('model4', model4)
], final_estimator= KN())

staking.fit(X_train, y_train)
y_pred_staking = staking.predict(X_test)
print("Stacking Classifier Performance:")
print(f"Accuracy of Stacking Classifier: {accuracy_score(y_test, y_pred_staking)}")


# Random Forest Classifier
random_forest_model = Pipeline(steps=[("step1",process),
                                      ("rf_model", RandomForestClassifier(n_estimators=100, random_state=42))])
random_forest_model .fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)
print("Random Forest Classifier Performance:")
print(f"Accuracy of Random Forest Classifier: {accuracy_score(y_test, y_pred_rf)}")

# corss-validation scores for Random Forest Classifier
rf_cv_scores = cross_val_score(random_forest_model, X, y, cv=5)
print(f"Cross-validation scores for Random Forest Classifier: {rf_cv_scores.mean() * 100:.2f}%")

# XGBoost Classifier
# xgboost is a powerful gradient boosting algorithm that is widely used for classification and regression tasks.
# its helpful in handling large datasets and provides high accuracy.
# xgboost classifier over come underfitting models into generalization models
from xgboost import XGBClassifier
xgb_model = Pipeline(steps=[("step1",process),
                            ("xgb_model", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42))])
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Classifier Performance:")
print(f"Accuracy of XGBoost Classifier: {accuracy_score(y_test, y_pred_xgb)}")

# adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
ada_model = Pipeline(steps=[("step1",process),
                            ("ada_model", AdaBoostClassifier(n_estimators=100, random_state=42))])
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print("AdaBoost Classifier Performance:")
print(f"Accuracy of AdaBoost Classifier: {accuracy_score(y_test, y_pred_ada)}")

# Gradient Boosting Classifier
gb_model = Pipeline(steps=[("step1",process),
                            ("gb_model", GradientBoostingClassifier(n_estimators=100, random_state=42))])
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Classifier Performance:")
print(f"Accuracy of Gradient Boosting Classifier: {accuracy_score(y_test, y_pred_gb)}")



















































# grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# grid search function to find the best hyperparameters for Logistic Regression model

# Define the parameter grid for hyperparameter tuning
# param_grid = {
#     "model_knn__n_neighbors": [3, 5, 7, 9, 11, 13],
#     "model_knn__weights": ['uniform', 'distance'],
#     "model_knn__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     "model_knn__leaf_size": [20, 30, 40],
#     "model_knn__p": [1, 2]
# }

# Define the parameter grid for hyperparameter tuning for SVC model
param_grid = {
    "model_svc__C": [0.1, 1, 10, 100],
    "model_svc__kernel": ['linear', 'rbf', 'poly'],
    "model_svc__gamma": ['scale', 'auto'],
    "model_svc__degree": [2, 3, 4],
    "model_svc__probability": [True]
}






def search_best_params(model, X_train, y_train, param_grid):


    grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=False, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best parameters found
    print("Best parameters:", grid_search.best_params_)

    # Best estimator
    best_knn = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_knn.predict(X_test)
    print("Test set accuracy:", accuracy_score(y_test, y_pred))

    # Cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    print(cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].T)

# search_best_params(model, X_train, y_train, param_grid)