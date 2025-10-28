
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
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
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
# Now importing LightGBM and XGBoost
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("kaggle competition\\Housing Prices Competition\\train (1).csv")
test = pd.read_csv("kaggle competition\\Housing Prices Competition\\test (1).csv")


# Store IDs for submission
test_ids = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# Target variable
y_train = np.log1p(train['SalePrice'])
train.drop('SalePrice', axis=1, inplace=True)

# Combine train and test for consistent preprocessing
all_data = pd.concat([train, test], axis=0)


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

def handle_missing_values(df):
    # Fill numerical missing values with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Specific handling for key features
    df['PoolQC'] = df['PoolQC'].fillna('None')
    df['MiscFeature'] = df['MiscFeature'].fillna('None')
    df['Alley'] = df['Alley'].fillna('None')
    df['Fence'] = df['Fence'].fillna('None')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    
    # Garage features
    garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_cols:
        df[col] = df[col].fillna('None')
    
    # Basement features
    bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in bsmt_cols:
        df[col] = df[col].fillna('None')
    
    return df

all_data = handle_missing_values(all_data)

from sklearn.preprocessing import LabelEncoder

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
            df[feature] = df[feature].map(mapping)
    
    # Label encode remaining categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

all_data = encode_features(all_data)


# Handle skewed numerical features - FIXED VERSION
import warnings
from scipy.special import boxcox1p
from scipy.stats import skew, boxcox_normmax

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
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
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


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# Define stacking class
class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Train base models and create out-of-fold predictions
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred
        
        # Train meta-model on out-of-fold predictions
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Fixed Stacking Class
class StackingModels(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    def fit(self, X, y):
        # Convert to numpy arrays to avoid pandas indexing issues
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Train base models and create out-of-fold predictions
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred
        
        # Train meta-model on out-of-fold predictions
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    def predict(self, X):
        # Convert to numpy array
        if hasattr(X, 'values'):
            X = X.values
            
        # Get predictions from each base model
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

# Prepare train/test splits from the combined all_data
n_train = y_train.shape[0]
X_train = all_data.iloc[:n_train, :].copy()
X_test = all_data.iloc[n_train:, :].copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models and meta model for stacking
base_models = (
    Lasso(alpha=0.0005, random_state=42),
    Ridge(alpha=10, random_state=42),
    XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, random_state=42, verbosity=0),
    LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
)
meta_model = Lasso(alpha=0.0005, random_state=42)

# Instantiate and train stacking model
stack_model = StackingModels(base_models=base_models, meta_model=meta_model, n_folds=5)
stack_model.fit(X_train_scaled, y_train)

# Make predictions
final_predictions = np.expm1(stack_model.predict(X_test_scaled))

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_predictions
})

submission.to_csv('house_price_submission.csv', index=False)
print("Submission file created!")

from sklearn.feature_selection import SelectFromModel

# Use Lasso for feature selection
selector = SelectFromModel(Lasso(alpha=0.0005, random_state=42))
selector.fit(X_train_scaled, y_train)
selected_features = X_train.columns[selector.get_support()]

print(f"Selected {len(selected_features)} features from {X_train.shape[1]} total")


from sklearn.model_selection import RandomizedSearchCV

# Optimize XGBoost parameters
param_dist = {
    'n_estimators': [1000, 2000, 3000],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.3, 0.4, 0.5]
}

xgb_optimizer = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

xgb_optimizer.fit(X_train_scaled, y_train)
print(f"Best XGBoost parameters: {xgb_optimizer.best_params_}")

# create final submission with optimized model
optimized_xgb = xgb_optimizer.best_estimator_
optimized_xgb.fit(X_train_scaled, y_train)

optimized_predictions = np.expm1(optimized_xgb.predict(X_test_scaled))

# check the rmse on training set
train_preds = optimized_xgb.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print(f"Training RMSE: {train_rmse}")

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': optimized_predictions
})
submission.to_csv('optimized_house_price_submission.csv', index=False)
print("Optimized submission file created!")


