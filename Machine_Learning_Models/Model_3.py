import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import csv file data
df = pd.read_csv("Machine_Learning_Models\\ford.csv")

# TODO 1. EDA (Exploratory Data Analysis)

# print(df.sample(10))
# df.info()
# print(df.isnull().sum())
# print(df.describe(include='int'))
# print(df.shape)
# for col in df.columns:
#     print(f"Column: {col}, Unique Values: {df[col].nunique()}")
#     print(df[df[col] == 0][col].value_counts().sum())

# print(((df[df["engineSize"] == 0]["engineSize"].value_counts().sum())/ df.shape[0]) * 100, "% of engineSize is 0")
# print((df == 0).mean() * 100, "% of all columns are 0")

# df["engineSize"] = df["engineSize"].replace(0, df["engineSize"].mean())
# print((df == 0).mean() * 100, "% of all columns are 0")

num_columns = df.select_dtypes(include=['float64', 'int64']).columns
catogorical_columns = ['model', 'transmission', 'fuelType']
# print(catogorical_columns)

# KDE plots for numeric columns

# for col in num_columns:
#     plt.figure(figsize=(10, 5))
#     sns.kdeplot(df[col], fill=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()


# Box plots for numeric columns
# for col in num_columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# count plot for categorical columns
# for col in catogorical_columns:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(x=df[col])
#     plt.title(f'Count Plot of {col}')
#     plt.xticks(rotation=45)
#     plt.show()
# print(df.nunique())

# Heatmap for correlation
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()

# TODO 2. Data Preprocessing

X = df.drop(columns=['price'])
num_columns = X.select_dtypes(include=['float64', 'int64']).columns
# print(X.shape)
y = df['price']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline




# One Hot Encoding for categorical columns
# process = ColumnTransformer(
#     transformers=[('cat', OneHotEncoder(drop="first"), ['model', 'transmission', 'fuelType'])],
#     remainder='passthrough')

ohe = OneHotEncoder(drop="first", sparse_output=False)

ohe.fit(X[catogorical_columns])
X_cat = ohe.transform(X[catogorical_columns])
X_cat = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(catogorical_columns))

# print(X_cat.head())

# Standard Scaling for numerical columns
# num_columns.delete(0)  # Remove 'price' column from num_columns
# print(num_columns)
scaler = StandardScaler()
scaler.fit(X[num_columns])
X_num = scaler.transform(X[num_columns])
X_num = pd.DataFrame(X_num, columns=num_columns)

df = pd.concat([X_num, X_cat], axis=1)
# print(df.head())
# df.info()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

power_transformer = PowerTransformer()
power_transformer.fit(X_train)
X_train = power_transformer.transform(X_train)
X_test = power_transformer.transform(X_test)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)





lg = LinearRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Ajusted R2 Score
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of features
adjusted_r2 = 1 - (1 - r2_score(y_test, y_pred)) * (n - 1) / (n - p - 1)
print("Adjusted R2 Score:", adjusted_r2)
