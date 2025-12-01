import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

train_df = pd.read_csv("kaggle competition\\Predicting Loan Payback\\train (3).csv")
test_df = pd.read_csv("kaggle competition\\Predicting Loan Payback\\test (3).csv")

test_ids = test_df['id']
test_df.drop('id', axis=1, inplace=True)
# combine train and test for preprocessing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# print(train_df.info())
# print(train_df.isnull().sum())
# print(train_df.shape)
# print(train_df['loan_paid_back'].value_counts())
# print(train_df.describe())
# # Visualizing the distribution of the target variable
# plt.figure(figsize=(6, 6))
# sns.countplot(x='loan_paid_back', data=train_df)
# plt.show()
# # check percentage of target variable
# print(train_df['loan_paid_back'].value_counts(normalize=True))

# # Visualizing the distribution of numerical features
# numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
# for col in numerical_cols:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(train_df[col], kde=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()
# # Visualizing the distribution of categorical features
# categorical_cols = train_df.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     plt.figure(figsize=(8, 4))
#     sns.countplot(y=col, data=train_df)
#     plt.title(f'Count Plot of {col}')
#     plt.show()


# # Correlation heatmap for numerical features
# plt.figure(figsize=(10, 8))
# sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap of Numerical Features')
# plt.show()

# split grade-sub_grade into two separate features
sp = combined_df['grade_subgrade'].str.split('', expand=True)
combined_df['grade'] = sp[1]
combined_df['sub_grade'] = sp[2].astype(int)

# print(combined_df['grade'])
# print(combined_df['sub_grade'])
combined_df.drop('grade_subgrade', axis=1, inplace=True)

# split train and test
train_df = combined_df.iloc[:len(train_df)]
test_df = combined_df.iloc[len(train_df):]

# preprocessing

X = train_df.drop('loan_paid_back', axis=1)
y = train_df['loan_paid_back']


numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
test_processed = preprocessor.transform(test_df)

# optuna
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 10.0)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best Parameters:", best_params)
# Train final model with best parameters

# model
model = XGBClassifier(random_state=42, **best_params)

model.fit(X_train_processed, y_train)
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# print precision, recall, f1-score
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')
recall_score = recall_score(y_test, y_pred)
print(f'Recall: {recall_score}')
f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1}')

