import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import  StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
# Load the dataset
df = pd.read_csv("Machine_Learning_Models//2019.csv")
# print(df.shape)
# print(df.head(5).T)  # Display the first few rows of the dataset transposed for better readability
# print(df.describe().T)  # Display the statistical summary
# find correlation between the features
# correlation_matrix = df.corr(numeric_only=True) # Display the correlation matrix
# print(correlation_matrix)
# print(df.isnull().sum())  # Check for missing values in each column

# num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# cat_columns = df.select_dtypes(include=['object']).columns.tolist()

# analyze the distribution of each column
# for col in df.columns:
#     if df[col].dtype in ['int64', 'float64']:
#         print(f"Column: {col}")
#         print(df[col].describe())  # Display descriptive statistics for numerical columns
#         print("\n")

# check distribution of dataset
# for col in num_columns:
#     plt.figure(figsize=(10, 5))
#     sns.kdeplot(x=df[col])
#     plt.title(f'Kdeplot of {col}')
#     plt.show()


# Box plots for numeric columns
# for col in num_columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# Heatmap for correlation
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()
# print("Before Imputation:")
# print(df["Cholesterol"].value_counts())

# count plot for categorical columns
# for col in cat_columns:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(x=df[col])
#     plt.title(f'Count Plot of {col}')
#     plt.xticks(rotation=45)
#     plt.show()
    # print(df[col].value_counts())

# check any duplicate rows
# print("Duplicate Rows:", df.duplicated().sum())

# if duplicate rows exist, drop them
# df.drop_duplicates(inplace=True)







# check each column how many 0 values are there
# print(df.isin([0]).sum())
# Replace 0 values in specific columns with NaN
# columns_to_replace_with_nan = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Perceptions of corruption']
# for col in columns_to_replace_with_nan:
#     df[col] = df[col].replace(0, np.nan)

# data preprocessing
df_copy = df.copy()


df_copy.drop(columns=["Country or region"], inplace=True)
# print(df_copy.head(5).T)  # Display the first few rows
print(df_copy.isnull().sum())  # Check for missing values in each column

X = df_copy.drop(columns=['Score'])
y = df_copy['Score']
from sklearn.utils import shuffle

# Shuffle X_train and y_train while keeping their alignment
X, y = shuffle(X, y, random_state=42)

print("Shuffled X_train head:\n", X[:5])
print("Shuffled y_train head:\n", y[:5])

features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# train the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.head())
print(y_test.head())

process = ColumnTransformer(transformers=[
    ("knn_impute", KNNImputer(n_neighbors=5), features),
    ("scale", StandardScaler(), ["Overall rank"])
    ], remainder='passthrough')

# create a pipeline for the model
model = Pipeline(steps=[
    ('preprocessor', process),
    ("ada_model", RandomForestRegressor(min_samples_split=5))
    
])

# model = KNeighborsRegressor()

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# check model performance
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.4f}")

# cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
# print(cv_scores)
# print(f"Cross-validation scores: {cv_scores.mean() * 100:.2f}%")



# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Random Forest')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
# for col in df_copy.columns:
#     sns.scatterplot(df_copy, x=col, y='Score')
#     plt.show()




