import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("kaggle competition\\student_daily_rutine\\student_exam_scores.csv")

# Display basic information about the dataset
# print(f"Train shape: {train_data.shape}")
# print(f"Train columns: {train_data.columns}")
# print(train_data.head())
# print(train_data.info())
# print(train_data.describe())

# Check for null values
# print(train_data.isnull().sum())

# Check for duplicates
# print(train_data.duplicated().sum())

# Check unique values in each column
# for col in train_data.columns:
#     print(f"{col}: {train_data[col].nunique()} unique values")

# numeric_cols = train_data.select_dtypes(include=np.number).columns
# categorical_cols = train_data.select_dtypes(include=['object', 'category']).columns
# KDE plots for numeric columns
# for col in numeric_cols:
#     plt.figure(figsize=(8, 4))
#     sns.kdeplot(train_data[col], fill=True)
#     plt.title(f"KDE Plot of {col}")
#     plt.show()

# Pairplot to see relationships between numeric variables
# sns.pairplot(train_data[numeric_cols])
# plt.suptitle("Pairplot of Numeric Features", y=1.02)
# plt.show()

# Correlation matrix
# corr_matrix = train_data.corr(numeric_only=True)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# box plots for numeric columns
# for col in numeric_cols:
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=train_data[col])
#     plt.title(f"Box Plot of {col}")
#     plt.show()

# feature engineering
# Create new features
train_data['study_sleep_ratio'] = train_data['hours_studied'] / (train_data['sleep_hours'] + 0.1)  # +0.1 to avoid division by zero
train_data['study_attendance_interaction'] = train_data['hours_studied'] * train_data['attendance_percent']
train_data['previous_performance_ratio'] = train_data['previous_scores'] / train_data['exam_score']
train_data['total_engagement'] = train_data['hours_studied'] + (train_data['attendance_percent'] / 10)
train_data['sleep_adequacy'] = np.where(train_data['sleep_hours'] >= 7, 'Adequate', 'Inadequate')
train_data['study_efficiency'] = train_data['exam_score'] / (train_data['hours_studied'] + 0.1)

# Create performance categories
train_data['performance_category'] = pd.cut(train_data['exam_score'], 
                                           bins=[0, 30, 40, 51.3], 
                                           labels=['Low', 'Medium', 'High'])

# Create study intensity categories
train_data['study_intensity'] = pd.cut(train_data['hours_studied'], 
                                      bins=[0, 4, 8, 12], 
                                      labels=['Light', 'Moderate', 'Heavy'])

print("New features created:")
print(train_data[['study_sleep_ratio', 'study_attendance_interaction', 'performance_category', 'study_intensity']].head())

# preprocessing
# Separate features and target
X = train_data.drop(columns=["exam_score", "student_id"], axis=1)
y = train_data["exam_score"]
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# for numeric features
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# for categorical features
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)
# fit and transform the data
X = preprocessor.fit_transform(X)
print(f"Preprocessed feature shape: {X.shape}")

# select best features
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

selector = SelectFromModel(model, prefit=True, threshold='median')
X_selected = selector.transform(X)
print(f"Original feature shape: {X.shape}, Selected feature shape: {X_selected.shape}")


# split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# train the model
model = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=1, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

# evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {np.sqrt(mse)}")
print(f"R^2 Score: {r2_score(y_val, y_pred)}")
# cross-validation

r2_scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
print(f"Cross-validated R^2 scores: {r2_scores}")


# hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 250, 350],
    'max_depth': [6, 8, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_selected, y)
print(f"Best parameters from Grid Search: {grid_search.best_params_}")
