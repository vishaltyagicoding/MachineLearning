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
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from scipy.stats import skew, boxcox_normmax
from sklearn.linear_model import Lasso, Ridge

# Load the dataset
train = pd.read_csv("kaggle competition\\Housing Prices Competition\\train (1).csv")
test = pd.read_csv("kaggle competition\\Housing Prices Competition\\test (1).csv")

# Store Id for submission
test_ids = test["Id"]

X = train.drop(columns=["SalePrice"])
y = train["SalePrice"]

# combine train and test for consistent preprocessing
all_data = pd.concat([X, test], axis=0, ignore_index=True)
print(f"Train shape: {X.shape}, Test shape: {test.shape}")
print(f"Combined data shape: {all_data.shape}")

# feature engineering
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

def encode_features(df):
    # Label encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Create quality mappings for ordinal features
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    basement_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
    exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
    
    # Apply ordinal encoding
    ordinal_features = {
        'ExterQual': quality_map, 'ExterCond': quality_map,
        'BsmtQual': quality_map, 'BsmtCond': quality_map,
        'HeatingQC': quality_map, 'KitchenQual': quality_map,
        'FireplaceQu': quality_map, 'GarageQual': quality_map,
        'GarageCond': quality_map, 'PoolQC': quality_map,
        'BsmtExposure': exposure_map,
        'BsmtFinType1': basement_map, 'BsmtFinType2': basement_map
    }
    
    for feature, mapping in ordinal_features.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping).fillna(0).astype(int)
    
    # Label encode remaining categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

all_data = encode_features(all_data)

def safe_boxcox(x):
    """Safe boxcox transformation with error handling"""
    try:
        # Ensure all values are positive and handle zeros
        x_positive = x + 1 - min(0, x.min())
        lam = boxcox_normmax(x_positive, brack=(-2.0, 2.0))
        return boxcox1p(x, lam)
    except:
        # Fallback to log transformation if boxcox fails
        return np.log1p(x - min(0, x.min()))

# Identify skewed features
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})

print(f"Found {len(skewed_feats)} numerical features")
print(f"Top 10 most skewed features:")
print(skewness.head(10))

# Transform highly skewed features (threshold > 0.5)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
print(f"Transforming {len(high_skew)} highly skewed features")

for feature in high_skew.index:
    if feature not in ['SalePrice', 'Id']:
        print(f"Transforming {feature} (skew: {skewed_feats[feature]:.2f})")
        all_data[feature] = safe_boxcox(all_data[feature])

# FIXED: Remove outliers BEFORE combining with test data
def outlier_removal(df, target):
    condition1 = (df['GrLivArea'] > 4000) & (target < 300000)
    condition2 = (df['TotalBsmtSF'] > 3000) & (target < 200000)
    outliers = df[condition1 | condition2].index
    return df.drop(outliers), target.drop(outliers)

# Apply outlier removal to training data only
X_clean, y_clean = outlier_removal(X.copy(), y.copy())
print(f"After outlier removal - X: {X_clean.shape}, y: {y_clean.shape}")

# Recreate all_data with cleaned training data
all_data_clean = pd.concat([X_clean, test], axis=0, ignore_index=True)
print(f"Cleaned combined data shape: {all_data_clean.shape}")

# Apply feature engineering and encoding to cleaned data
all_data_clean = create_advanced_features(all_data_clean)
all_data_clean = encode_features(all_data_clean)

# Preprocessing - FIXED: Use simpler preprocessing since we already encoded
numeric_features = all_data_clean.select_dtypes(include=[np.number]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

# Since we already encoded categorical features, we don't need categorical transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)

all_data_preprocessed = preprocessor.fit_transform(all_data_clean)

# Split back into train and test using the cleaned data length
train_processed = all_data_preprocessed[:len(X_clean)]
test_processed = all_data_preprocessed[len(X_clean):]

print(f"Processed train shape: {train_processed.shape}, Processed test shape: {test_processed.shape}")

# Transform target variable
y_clean = np.log1p(y_clean)

# split train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_processed, y_clean, test_size=0.2, random_state=42)

# Define models
lgb_model = LGBMRegressor(
    learning_rate=0.01,
    max_depth=4,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

xgb_model = XGBRegressor(
    colsample_bytree=0.9,
    learning_rate=0.05,
    max_depth=3,
    n_estimators=2000,
    subsample=0.8,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingRegressor(
   learning_rate=0.01,
   max_depth=4,
   min_samples_leaf=1,
   min_samples_split=2,
   n_estimators=2000,
   subsample=0.8,
   random_state=42
)

# Individual model validation
print("\nIndividual model performance:")
models = {
    'LightGBM': lgb_model,
    'XGBoost': xgb_model,
    'RandomForest': rf_model,
    'GradientBoosting': gb_model,
}

for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"{name} Validation RMSE: {val_rmse:.4f}")

# Use the best performing model as final model
final_model = gb_model  # You can change this based on which performs best
final_model.fit(X_train, y_train)
val_pred = final_model.predict(X_val)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"\nFinal Model Validation RMSE: {val_rmse:.4f}")

# Predict on test set and prepare submission
test_pred = final_model.predict(test_processed)
test_pred = np.expm1(test_pred)  # inverse of log1p

# Create submission
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")