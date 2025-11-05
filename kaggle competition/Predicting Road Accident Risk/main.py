import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("kaggle competition\\Predicting Road Accident Risk\\train.csv")
test = pd.read_csv("kaggle competition\\Predicting Road Accident Risk\\test.csv")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {train.columns}")

# submission ids
test_ids = test["id"]

# Separate features and target
X = train.drop(columns=["accident_risk"], axis=1)
y = train["accident_risk"]

# combine train and test for consistent preprocessing
all_data = pd.concat([X, test], axis=0, ignore_index=True)
print(f"Combined data shape: {all_data.shape}")

# check null values
print(all_data.isnull().sum())

# check duplicates
print(all_data.duplicated().sum())

# check info
print(all_data.info())

# check head
print(all_data.head())

# check description
print(all_data.describe())

# check correlation
print(all_data.corr(numeric_only=True))

# check unique values
for col in all_data.columns:
    print(f"{col}: {all_data[col].nunique()} unique values")

numeric_cols = all_data.select_dtypes(include=np.number).columns
categorical_cols = all_data.select_dtypes(include=['object']).columns

# kde plots
# for col in numeric_cols:
#     plt.figure(figsize=(8, 4))
#     sns.kdeplot(all_data[col].dropna(), fill=True)
#     plt.title(f"KDE Plot of {col}")
#     plt.show() 

# count plots for categorical variables
# for col in categorical_cols:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(y=all_data[col])
#     plt.title(f"Count Plot of {col}")
#     plt.show()


# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# for numeric features
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# for categorical features
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

all_data_preprocessed = preprocessor.fit_transform(all_data)
print(f"Preprocessed combined data shape: {all_data_preprocessed.shape}")


# split back into train and test
X_processed = all_data_preprocessed[:len(X)]
test_processed = all_data_preprocessed[len(X):]

print(f"Processed train shape: {X_processed.shape}, Processed test shape: {test_processed.shape}")


# create a model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# split the processed train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# fit the model
model.fit(X_train, y_train)

# evaluate the model
val_score = model.score(X_val, y_val)
print(f"Validation R^2 Score: {val_score}")


# make predictions on test set
test_predictions = model.predict(test_processed)
# prepare submission
submission = pd.DataFrame({
    "id": test_ids,
    "accident_risk": test_predictions
})
submission.to_csv("submission.csv", index=False)

# check rmse
from sklearn.metrics import root_mean_squared_log_error
def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# Example usage (assuming you have true values for validation set)
y_val_pred = model.predict(X_val)
rmse = root_mean_squared_log_error(y_val, y_val_pred)
print(f"Validation RMSLE: {rmse}")
