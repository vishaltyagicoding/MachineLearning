import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import optuna

# Optimizing Multiple ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv("optuna\\ecommerce_customer_behavior_dataset_v2.csv")
# print(df.sample(10))  
# print(df.info())
# print(df.isnull().sum())
# print(df.shape)
X = df.drop(["Is_Returning_Customer","Order_ID", "Customer_ID"], axis=1)
y = df["Is_Returning_Customer"]

# print(X.shape)

# catogorical_cols = X.select_dtypes(include=['object']).columns
# numerical_cols = X.select_dtypes(exclude=['object']).columns

# # kde plots for numerical features

# for col in numerical_cols:
#     plt.figure(figsize=(8, 4))
#     sns.kdeplot(data=X, x=col, hue=y, fill=True)
#     plt.title(f'KDE Plot of {col} by Returning Customer')
#     plt.show()

# # count plots for categorical features
# for col in catogorical_cols:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(data=X, x=col, hue=y)
#     print(df[col].value_counts())
#     plt.title(f'Count Plot of {col} by Returning Customer')
#     plt.xticks(rotation=45)
#     plt.show()

# # pie chart for target variable
# plt.figure(figsize=(6, 6))
# plt.pie(y.value_counts(), labels=y.value_counts().index, autopct='%1.1f%%', startangle=140)
# plt.title('Pie Chart of Returning Customer')
# plt.show()

# Convert the column to datetime
X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
X['year'] = X['Date'].dt.year
X['month'] = X['Date'].dt.month
X['day'] = X['Date'].dt.day
X = X.drop('Date', axis=1)


from imblearn.over_sampling import RandomOverSampler

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
# Applying Random over Sampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print(X_resampled.shape, y_resampled.shape)
print("After OverSampling, counts of label '1': {}".format(sum(y_resampled==1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(y_resampled==0)))


catogorical_cols = X_resampled.select_dtypes(include=['object']).columns
numerical_cols = X_resampled.select_dtypes(exclude=['object']).columns

numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, catogorical_cols)
    ])


X_processed = preprocessor.fit_transform(X_resampled)
X_test_processed = preprocessor.transform(X_test)

def objective_model(trial):
    # XGBoost hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0) 
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )


    model.fit(X_processed, y_resampled)
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


study_model = optuna.create_study(direction='maximize')
study_model.optimize(objective_model, n_trials=50)

# Retrieve the best trial
best_trial = study_model.best_trial
print("Best trial parameters:", best_trial.params)
print("Best trial accuracy:", best_trial.value)

# print confusion matrix and classification report for the best model
best_model = XGBClassifier(**best_trial.params, random_state=42)
best_model.fit(X_processed, y_resampled)
y_pred = best_model.predict(X_test_processed)
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))
# 16+27+32