import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df1 = pd.read_csv("AdaBoost Algorithms\\laptop_data_cleaned.csv")
df2 = pd.read_csv("AdaBoost Algorithms\my_laptop_data_cleaned__.csv")

print(df1.info())
print(df2.info())

df1_cols = set(df1.columns)
df2_cols = set(df2.columns)

# for col in df1_cols:
#     print(df1[col].value_counts())

# for col in df2_cols:
#     print(df2[col].value_counts())


# compare both data set kde plot
# for col in df1_cols.intersection(df2_cols):
#     if df1[col].dtype in [np.float64, np.int64] and df2[col].dtype in [np.float64, np.int64]:
#         plt.figure(figsize=(10, 6))
#         sns.kdeplot(df1[col], label='Dataset 1', fill=True)
#         sns.kdeplot(df2[col], label='Dataset 2', fill=True)
#         plt.title(f'KDE Plot of {col}')
#         plt.xlabel(col)
#         plt.ylabel('Density')
#         plt.legend()
#         plt.show()

# comapre both dataset cout plot for cat cols
# for col in df1_cols.intersection(df2_cols):
#     if df1[col].dtype == 'object' and df2[col].dtype == 'object':
#         plt.figure(figsize=(10, 6))
#         sns.countplot(data=df1, x=col, color='blue', alpha=0.5, label='Dataset 1')
#         sns.countplot(data=df2, x=col, color='orange', alpha=0.5, label='Dataset 2')
#         plt.title(f'Count Plot of {col}')
#         plt.xlabel(col)
#         plt.ylabel('Count')
#         plt.legend()
#         plt.show()


# cehck distribution of both cols

# for col in df1_cols.intersection(df2_cols):
#     if df1[col].dtype in [np.float64, np.int64] and df2[col].dtype in [np.float64, np.int64]:
#         plt.figure(figsize=(10, 6))
#         sns.histplot(df1[col], color='blue', alpha=0.5, label='Dataset 1', kde=True)
#         sns.histplot(df2[col], color='orange', alpha=0.5,
#                      label='Dataset 2', kde=True)
#         plt.title(f'Histogram of {col}')
#         plt.xlabel(col)
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.show()


# print(df1["Price"].describe())
# print(df2["Price"].describe())

# target column ked plot
# create 2 subplots

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(df1["Price"], fill=True, color='blue')
plt.title('KDE Plot of Price - Dataset 1')
plt.xlabel('Price')
plt.ylabel('Density')
plt.subplot(1, 2, 2)
sns.kdeplot(df2["Price"], fill=True, color='orange')
plt.title('KDE Plot of Price - Dataset 2')
plt.xlabel('Price')
plt.ylabel('Density')
plt.tight_layout()
plt.show()
