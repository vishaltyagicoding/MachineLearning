from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import seaborn as sns
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
train = pd.read_csv("kaggle competition\\Housing Prices Competition\\train (1).csv")
test = pd.read_csv("kaggle competition\\Housing Prices Competition\\test (1).csv")

# analyze data
# print(train.info())
# print(train.describe())
# for col in train.columns:
#     if train[col].isnull().sum() > 0:
#         print(f"{col}: {train[col].isnull().sum()} missing values, data type {train[col].dtype}")

# print(train.head())



# Store Id for submission
test_ids = test["Id"]

X = train.drop(columns=["SalePrice"])
y = train["SalePrice"]

# combine train and test for consistent preprocessing
all_data = pd.concat([X, test], axis=0, ignore_index=True)
print(f"Train shape: {X.shape}, Test shape: {test.shape}")
print(f"Combined data shape: {all_data.shape}")

# feature engineering
# Add more powerful features
def create_advanced_features(df):
    # Total square footage
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalArea'] = df['TotalSF'] + df['GarageArea'] + df['PoolArea']
    
    # Bathroom features
    df['TotalBath'] = (df['FullBath'] + 
                       df['BsmtFullBath'] + 
                       0.5 * (df['HalfBath'] + df['BsmtHalfBath']))
    
    # Porch features
    df['TotalPorch'] = (df['OpenPorchSF'] + 
                        df['EnclosedPorch'] + 
                        df['3SsnPorch'] + 
                        df['ScreenPorch'])
    
    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    
    # Quality features
    df['OverallQual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['OverallQual_YearBuilt'] = df['OverallQual'] * df['YearBuilt']
    
    # Room features
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['FullBath'] + df['HalfBath']
    df['RoomSize'] = df['TotalSF'] / df['TotalRooms']
    df['BedroomRatio'] = df['BedroomAbvGr'] / df['TotalRooms']
    
    # Lot features
    df['LotRatio'] = df['LotArea'] / df['TotalSF']
    df['LivingAreaRatio'] = df['GrLivArea'] / df['TotalSF']
    
    # External quality scores
    df['ExternalScore'] = df['ExterQual'] + df['ExterCond']
    
    return df

all_data = create_advanced_features(all_data)

# Preprocessing

numeric_features = all_data.select_dtypes(include=["int64", "float64"]).columns
categorical_features = all_data.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

all_data_preprocessed = preprocessor.fit_transform(all_data)

# visualize all features
# print("Visualizing numeric feature distributions...")
# for i, col in enumerate(numeric_features):
#     plt.figure(figsize=(10, 4))
#     sns.kdeplot(all_data[col].dropna(), fill=True)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Density")
#     plt.show()
# print("Visualizing categorical feature distributions...")
# for col in categorical_features:
#     plt.figure(figsize=(10, 4))
#     sns.countplot(y=all_data[col], order=all_data[col].value_counts().index)
#     plt.title(f"Count of {col}")
#     plt.xlabel("Count")
#     plt.ylabel(col)
#     plt.show()

# # visualize target feature
# print("Visualizing SalePrice distribution...")
# print(y.isnull().sum())
# print(y.describe())


# plt.figure(figsize=(10, 6))
# sns.kdeplot(y, shade=True)
# plt.title("Distribution of Sale Prices")
# plt.xlabel("Sale Price")
# plt.ylabel("Density")
# plt.show()


# Split back into train and test
train_processed = all_data_preprocessed[:len(X)]
test_processed = all_data_preprocessed[len(X):]

print(f"Processed train shape: {train_processed.shape}, Processed test shape: {test_processed.shape}")

# Transform target variable
y = np.log1p(y)
# split train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_processed, y, test_size=0.2, random_state=42)

# Ensemble of models
print("Training ensemble models...")

# Model 1: LightGBM (primary)
lgb_model = LGBMRegressor(
    learning_rate=0.01,
    max_depth=4,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    feature_name='auto'
)

# Model 2: XGBoost
xgb_model = XGBRegressor(
    colsample_bytree=0.9,
    learning_rate=0.05,
    max_depth=3,
    n_estimators=2000,
    subsample=0.8
)

# Model 3: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Model 4: Gradient Boosting
gb_model = GradientBoostingRegressor(
   learning_rate=0.01,
   max_depth=4,
   min_samples_leaf=1,
   min_samples_split=2,
   n_estimators=2000,
   subsample=0.8
)
# Individual model validation (for comparison)
print("\nIndividual model performance:")
models = {
    'LightGBM': lgb_model,
    'XGBoost': xgb_model,
    'RandomForest': rf_model,
    'GradientBoosting': gb_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"{name} Validation RMSE: {val_rmse:.4f}")


# Since GradientBoosting performs best, focus on optimizing it

# gb_params = {
#     'n_estimators': [1000, 2000],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 4],
#     'subsample': [0.8, 0.9],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# gb = GradientBoostingRegressor(random_state=42)
# grid_search = GridSearchCV(gb, gb_params, cv=5, 
#                           scoring='neg_root_mean_squared_error',
#                           n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)

# print(f"Best GB params: {grid_search.best_params_}")
# print(f"Best GB score: {-grid_search.best_score_:.4f}")

# use xgb_model as final model
final_model = xgb_model.fit(X_train, y_train)
val_pred = final_model.predict(X_val)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"\nFinal Model (XGBoost) Validation RMSE: {val_rmse:.4f}")

# Predict on test set and prepare submission
test_pred = final_model.predict(test_processed)
test_pred = np.expm1(test_pred)  # inverse of log1p
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

