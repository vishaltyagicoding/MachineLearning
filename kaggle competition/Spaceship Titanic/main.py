# Spaceship Titanic Submission - Kaggle Competition Solution
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import QuantileTransformer
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_train = pd.read_csv("kaggle competition//Spaceship Titanic//train.csv")
df_test = pd.read_csv("kaggle competition//Spaceship Titanic//test.csv")

# Store PassengerId for submission
test_passenger_ids = df_test['PassengerId'].copy()

# Combine train and test for consistent preprocessing
df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
print(df_train.shape, df_test.shape)
print("Combined data shape:", df_combined.shape)
print("Combined data columns:", df_combined.columns.tolist())


# Enhanced Feature Engineering
def create_features(df):
    X = df.copy()
    
    # Cabin feature engineering
    X[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = X['Cabin'].str.split('/', expand=True)
    X['Cabin_Num'] = pd.to_numeric(X['Cabin_Num'], errors='coerce')

    # PassengerId feature engineering
    X['GroupId'] = X['PassengerId'].str.split('_').str[0]
    X['PersonId'] = X['PassengerId'].str.split('_').str[1].astype(int)

    # Advanced group features
    group_sizes = X['GroupId'].value_counts()
    X['GroupSize'] = X['GroupId'].map(group_sizes)
    X['IsAlone'] = (X['GroupSize'] == 1).astype(int)

    # Spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    X['TotalSpending'] = X[spending_cols].sum(axis=1)
    X['SpendingPerPerson'] = X['TotalSpending'] / X['GroupSize']
    X['HasSpending'] = (X['TotalSpending'] > 0).astype(int)
    X['SpendingRatio'] = X['RoomService'] / (X['TotalSpending'] + 1)  # Avoid division by zero

    # Age features
    X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 30, 50, 100], 
                           labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    X['IsChild'] = (X['Age'] < 13).astype(int)
    X['IsElderly'] = (X['Age'] > 60).astype(int)

    # Cabin numerical features
    X['Cabin_Num_Group'] = pd.cut(X['Cabin_Num'], bins=5, labels=False)

    # Luxury spending indicators
    X['LuxurySpending'] = X[['RoomService', 'Spa', 'VRDeck']].sum(axis=1)
    X['BasicSpending'] = X[['FoodCourt', 'ShoppingMall']].sum(axis=1)
    X['LuxuryRatio'] = X['LuxurySpending'] / (X['TotalSpending'] + 1)

    # Handle CryoSleep NaN values before creating interactions
    X['CryoSleep'] = X['CryoSleep'].fillna(False)
    for col in spending_cols:
        X[f'{col}_with_CryoSleep'] = X[col] * X['CryoSleep'].astype(int)

    # Handle other categorical NaN values before interactions
    X['HomePlanet'] = X['HomePlanet'].fillna('Unknown')
    X['Destination'] = X['Destination'].fillna('Unknown')

    # Destination and HomePlanet interactions
    X['Earth_CryoSleep'] = ((X['HomePlanet'] == 'Earth') & (X['CryoSleep'] == True)).astype(int)
    X['Europa_Destination'] = ((X['HomePlanet'] == 'Europa') & (X['Destination'] == 'TRAPPIST-1e')).astype(int)

    # Additional powerful features
    X['NoSpending'] = (X['TotalSpending'] == 0).astype(int)
    X['HighSpender'] = (X['TotalSpending'] > X['TotalSpending'].median()).astype(int)
    X['FamilySize'] = X['GroupSize'] * X['IsChild']  # Family with children

    # Drop original columns
    X = X.drop(['Cabin', 'GroupId', 'PersonId', 'PassengerId', 'Name'], axis=1, errors='ignore')
    
    return X

# Apply feature engineering to combined data
print("Creating features...")
df_combined_processed = create_features(df_combined)
print(df_combined_processed)

# Split back into train and test
train_processed = df_combined_processed[:len(df_train)]
test_processed = df_combined_processed[len(df_train):]

# Prepare features and target
X = train_processed
y = df_train['Transported']
X_test = test_processed

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Total features after engineering: {len(numerical_cols) + len(categorical_cols)}")

# Enhanced transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', QuantileTransformer(output_distribution='normal'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

# Apply preprocessing
print("Preprocessing data...")
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

print(f"Final processed features shape - Train: {X_processed_df.shape}, Test: {X_test_processed_df.shape}")

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed_df, y, test_size=0.20, random_state=42, stratify=y
)

# Feature selection using XGBoost
print("Performing feature selection...")
xgb_selector = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_selector.fit(X_train, y_train)

selector = SelectFromModel(xgb_selector, prefit=True, max_features=35, threshold=-np.inf)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)
X_full_selected = selector.transform(X_processed_df)
X_test_selected = selector.transform(X_test_processed_df)

selected_features = feature_names[selector.get_support()]
print(f"Selected {len(selected_features)} features")

# Train final model on full training data
print("Training final model...")
final_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

final_model.fit(X_full_selected, y)

# Validation score (optional)
val_pred = final_model.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Cross-validation Score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_full_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = final_model.predict(X_test_selected)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Transported': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
print(f"Number of predictions: {len(test_predictions)}")
print(f"Transported rate in predictions: {test_predictions.mean():.4f}")

# Display first few rows of submission
print("\nFirst few rows of submission:")
print(submission.sample())