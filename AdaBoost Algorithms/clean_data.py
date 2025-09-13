import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# load uncleaned data

df = pd.read_csv('AdaBoost Algorithms\\laptopData.csv')

# print(df.head())
# print(df.info())

df = df.drop(['Unnamed: 0'], axis=1)

# TODO 1 Company column

# print(df['Company'].nunique())

# drop all companies other than the top 10
# top_10_companies = df['Company'].value_counts().nlargest(15).index
# df = df[df['Company'].isin(top_10_companies)]

# df['Company'] = np.where(df['Company'].isin(top_10_companies), df['Company'], "Other Company")


# print(df["Company"].value_counts())

# check for missing values
# print(df.isnull().sum())

# for col in df.columns:
#     print(f"{col}: {df[col].unique()} unique values")




# TODO 2 Inches column
# convert Inches column to float
df["Inches"] = pd.to_numeric(df["Inches"], errors='coerce')
# df = df.drop(['Inches'], axis=1)
# print(df['Inches'].dtype)
# print(df['Inches'].sample(5))


# TODO 3 Ram column
# convert 'Ram' column to float
df['Ram'] = df['Ram'].str.replace('GB', '').astype(float)
# print(df['Ram'].dtype)
# print(df['Ram'].head())

# TODO 4 Weight column
# convert multiple value  'Weight' column to float
df["Weight"] = df["Weight"].replace('?', pd.NA)
df["Weight"] = df["Weight"].str.replace("kg", "")
df["Weight"] = pd.to_numeric(df["Weight"], errors='coerce')
# print(df['Weight'].dtype)
# print(df['Weight'].sample(5))

# TODO 5 ScreenResolution column
# create a new column 'Touchscreen', IPS, ppi from 'ScreenResolution' column

Touchscreen = []
IPS = []
ppi = []

for str_list in df['ScreenResolution']:
    if str_list == str_list:  # check if not NaN
        if 'Touchscreen' in str_list:
            Touchscreen.append(1)
        else:
            Touchscreen.append(0)

        if 'IPS' in str_list:
            IPS.append(1)
        else:
            IPS.append(0)

        # calculate ppi
        resolution = str_list.split('x')
        X_res = int(resolution[0].split()[-1])
        Y_res = int(resolution[1])
        Inches = df['Inches'][df['ScreenResolution'] == str_list].values[0]

        ppi_value = (math.sqrt(X_res**2 + Y_res**2)) / Inches
        ppi.append(ppi_value)
    else:
        Touchscreen.append(pd.NA)
        IPS.append(pd.NA)
        ppi.append(pd.NA)

    
           

# print(len(Touchscreen), len(IPS), len(ppi))
df['Touchscreen'] = Touchscreen
df['IPS'] = IPS
df['ppi'] = ppi

# drop ScreenResolution column
df = df.drop(['ScreenResolution'], axis=1)
df = df.drop(['Inches'], axis=1)
# print(df.sample(10))

# TODO 6 Cpu column
# convert 'Cpu' column to categorical
# df['Cpu'] = df['Cpu'].apply(lambda x: " ".join(x.split(" ")[:3]))
# df['Cpu'] = df['Cpu'].apply(lambda x: " ".join(str(x).split(" ")[:3]) if pd.notna(x) else x)

# drop Samsung Cortex A72&A53 row
# df = df[df['Cpu'] != 'Samsung Cortex A72&A53']
try:
    df["Cpu"] = df["Cpu"].apply(lambda x: "AMD" if "AMD" in x else x)
    df["Cpu"] = df["Cpu"].apply(lambda x: "Other Intel Processor" if x not in [
        "AMD" , "Intel Core i3" , "Intel Core i5" , "Intel Core i7"] else x)
    
    df['Gpu'] = df['Gpu'].apply(lambda x: " ".join(x.split(" ")[:1]))

    # TODO 9 OpSys column
    df['Os'] = df['OpSys'].apply(lambda x: " ".join(x.split(" ")[:1]))
    # replace macOs with Mac
    df['OpSys'] = df['OpSys'].replace('macOS', 'Mac')
    df['OpSys'] = df['OpSys'].apply(lambda x: "Other" if x not in ['Windows', 'Mac', 'Linux'] else x)

    print(df['Cpu'].value_counts())
except Exception as e:
    print(f"Error processing Cpu column: {e}")


# TODO 7 TypeName column

# print(df['TypeName'].value_counts())

# TODO 8 Gpu column

# df['Gpu'] = df['Gpu'].apply(lambda x: " ".join(x.split(" ")[:1]))
# print(df['Gpu'].value_counts())
# print(df['Gpu'].sample(10))


# # TODO 9 OpSys column
# df['OpSys'] = df['OpSys'].apply(lambda x: " ".join(x.split(" ")[:1]))
# # replace macOs with Mac
# df['OpSys'] = df['OpSys'].replace('macOS', 'Mac')
# df['OpSys'] = df['OpSys'].apply(lambda x: "Other" if x not in ['Windows', 'Mac', 'Linux'] else x)

# print(df['OpSys'].value_counts())

# TODO 10 Memory column
# replace TB Unit values into GB unit values
df['Memory'] = df['Memory'].str.replace('TB', '000GB')

# create SSD, HDD, Flash Storage, Hybrid from Memory column
SSD = []
HDD = []
Flash_Storage = []
Hybrid = []


for str_list in df['Memory']:
    ssd = 0
    hdd = 0
    flash_storage = 0
    hybrid = 0
    if str_list == str_list:  # check if not NaN
        str_list = str_list.split("+")
        for item in str_list:
            item = item.strip()
            if 'SSD' in item:
                ssd += int(float(item.replace('SSD', '').replace('GB', '').strip()))
            elif 'HDD' in item:
                hdd += int(float(item.replace('HDD', '').replace('GB', '').strip()))
            # elif 'Flash Storage' in item:
            #     flash_storage += int(float(item.replace('Flash Storage', '').replace('GB', '').strip()))
            # elif 'Hybrid' in item:
            #     hybrid += int(float(item.replace('Hybrid', '').replace('GB', '').strip()))
        SSD.append(ssd)
        HDD.append(hdd)
        Flash_Storage.append(flash_storage)
        Hybrid.append(hybrid)
    else:
        SSD.append(pd.NA)
        HDD.append(pd.NA)
        # Flash_Storage.append(pd.NA)
        # Hybrid.append(pd.NA)

df['SSD'] = SSD
df['HDD'] = HDD
# df['Flash_Storage'] = Flash_Storage
# df['Hybrid'] = Hybrid

df = df.drop(['Memory'], axis=1)

# drop nan values rows
df = df.dropna()

# check data types
# print(df.dtypes)


# # fill missing values with median for numerical columns
# numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
# for col in numerical_cols:
#     df[col] = df[col].fillna(df[col].median())
# # fill missing values with mode for categorical columns
# categorical_cols = df.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     df[col] = df[col].fillna(df[col].mode()[0])
# check for missing values
# print(df.isnull().sum())

# for col in df.columns:
#     print(df[col].sample(10))


# print(df.info())
# print(df.describe())
# print(df.shape)


# check distribution of columns

# for col in df.columns:
#     plt.figure(figsize=(10, 5))
#     if df[col].dtype == 'object':
#         sns.countplot(y=df[col], order=df[col].value_counts().index)
#     else:
#         sns.histplot(df[col], kde=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()


# box plot to check for outliers
# for col in numerical_cols:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[col])
#     plt.title(f'Box plot of {col}')
#     plt.show()


# check correlation between numerical columns
# Select only numeric columns
# numeric_df = df.select_dtypes(include=[np.number])

# plt.figure(figsize=(12, 8))
# sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Heatmap (Numeric Columns Only)')
# plt.show()


# save cleaned data
df.to_csv('AdaBoost Algorithms\\my_laptop_data_cleaned__.csv', index=False)
  
# print(df.info())

# model training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# import Adaboost regressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
# import knn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor



# Load your data
# df = pd.read_csv('your_data.csv')

# Separate features and target
X = df.drop('Price', axis=1)  # Replace 'Price' with your target column name
y = df['Price']

# stdandardize target variable
y = np.log1p(y)  # log(1 + y) to handle zero values if any

# Find the section around line 282 and modify it:

# remove outlier in target columns IQR

# Q1 = y.quantile(0.25)
# Q3 = y.quantile(0.75)
# IQR = Q3 - Q1
# filter = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
# X = X.loc[filter]
# y = y.loc[filter]



# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Split the data FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),  # Impute missing values
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ("cat_imputer", SimpleImputer(strategy='most_frequent')),  # Impute missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('power_transform', PowerTransformer()),  # Optional: Apply power transformation
    # ('regressor', RandomForestRegressor(n_estimators=100))  # You can replace with AdaBoostRegressor() or RandomForestRegressor()
    ("boost", GradientBoostingRegressor(n_estimators=500))
])

# Now perform cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validated R2 scores: {scores}")


# Fit the model on training data
pipeline.fit(X_train, y_train)

# Evaluate on test data
# test_score = pipeline.score(X_test, y_test)
# print(f"Test R2 score: {test_score:.4f}")

# r2 score
y_pred = pipeline.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# check best parameters using GridSearchCV for random forest

# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'regressor__n_estimators': [50, 100, 200],
#     'regressor__max_depth': [None, 10, 20],
#     'regressor__min_samples_split': [2, 5, 10]
# }
# grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", grid_search.best_score_")
