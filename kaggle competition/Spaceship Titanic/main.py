# Spaceship Titanic Submission - Kaggle Competition Solution
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
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
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
print(f"Combined data shape: {df_combined.shape}")

# Enhanced Feature Engineering
def create_features(df):
    X = df.copy()
    
    # Cabin feature engineering
    X[['Cabin_Deck', 'Cabin_Num', 'Cabin_Side']] = X['Cabin'].str.split('/', expand=True)
    X['Cabin_Num'] = pd.to_numeric(X['Cabin_Num'], errors='coerce')
    
    # Cabin deck encoding (ordinal)
    deck_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    X['Cabin_Deck_Encoded'] = X['Cabin_Deck'].map(deck_order)
    
    # Cabin side encoding
    X['Cabin_Side_Encoded'] = (X['Cabin_Side'] == 'P').astype(int)

    # PassengerId feature engineering
    X['GroupId'] = X['PassengerId'].str.split('_').str[0]
    X['PersonId'] = X['PassengerId'].str.split('_').str[1].astype(int)

    # Advanced group features
    group_sizes = X['GroupId'].value_counts()
    X['GroupSize'] = X['GroupId'].map(group_sizes)
    X['IsAlone'] = (X['GroupSize'] == 1).astype(int)
    
    # Group position features
    X['GroupPosition'] = X.groupby('GroupId')['PersonId'].transform('rank')
    X['IsFirstInGroup'] = (X['GroupPosition'] == 1).astype(int)
    X['IsLastInGroup'] = (X['GroupPosition'] == X['GroupSize']).astype(int)

    # Spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    X['TotalSpending'] = X[spending_cols].sum(axis=1)
    X['SpendingPerPerson'] = X['TotalSpending'] / (X['GroupSize'] + 0.1)  # Avoid division by zero
    X['HasSpending'] = (X['TotalSpending'] > 0).astype(int)
    
    # Individual spending ratios
    for col in spending_cols:
        X[f'{col}_Ratio'] = X[col] / (X['TotalSpending'] + 1)
    
    # Log transform spending (handle zeros)
    for col in spending_cols + ['TotalSpending']:
        X[f'Log_{col}'] = np.log1p(X[col])

    # Age features
    X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 30, 50, 65, 100], 
                           labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior', 'Elderly'])
    X['IsChild'] = (X['Age'] < 13).astype(int)
    X['IsElderly'] = (X['Age'] > 60).astype(int)
    X['Age_GroupSize_Interaction'] = X['Age'] * X['GroupSize']

    # Cabin numerical features
    X['Cabin_Num_Group'] = pd.cut(X['Cabin_Num'], bins=10, labels=False)

    # Luxury spending indicators
    luxury_cols = ['RoomService', 'Spa', 'VRDeck']
    basic_cols = ['FoodCourt', 'ShoppingMall']
    X['LuxurySpending'] = X[luxury_cols].sum(axis=1)
    X['BasicSpending'] = X[basic_cols].sum(axis=1)
    X['LuxuryRatio'] = X['LuxurySpending'] / (X['TotalSpending'] + 1)
    X['BasicRatio'] = X['BasicSpending'] / (X['TotalSpending'] + 1)

    # Handle CryoSleep NaN values before creating interactions
    X['CryoSleep'] = X['CryoSleep'].fillna(False)
    
    # CryoSleep interactions with spending
    for col in spending_cols:
        X[f'{col}_CryoSleep_Interaction'] = X[col] * X['CryoSleep'].astype(int)
    
    X['CryoSleep_NoSpending'] = (X['CryoSleep'] & (X['TotalSpending'] == 0)).astype(int)

    # Handle other categorical NaN values before interactions
    X['HomePlanet'] = X['HomePlanet'].fillna('Unknown')
    X['Destination'] = X['Destination'].fillna('Unknown')
    
    # HomePlanet and Destination encoding
    planet_encoding = {'Earth': 0, 'Europa': 1, 'Mars': 2, 'Unknown': -1}
    dest_encoding = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2, 'Unknown': -1}
    X['HomePlanet_Encoded'] = X['HomePlanet'].map(planet_encoding)
    X['Destination_Encoded'] = X['Destination'].map(dest_encoding)

    # VIP handling
    X['VIP'] = X['VIP'].fillna(False)

    # Destination and HomePlanet interactions
    X['Earth_CryoSleep'] = ((X['HomePlanet'] == 'Earth') & (X['CryoSleep'] == True)).astype(int)
    X['Europa_Destination'] = ((X['HomePlanet'] == 'Europa') & (X['Destination'] == 'TRAPPIST-1e')).astype(int)
    X['Mars_VIP'] = ((X['HomePlanet'] == 'Mars') & (X['VIP'] == True)).astype(int)

    # Additional powerful features
    X['NoSpending'] = (X['TotalSpending'] == 0).astype(int)
    X['HighSpender'] = (X['TotalSpending'] > X['TotalSpending'].median()).astype(int) if X['TotalSpending'].notna().any() else 0
    X['FamilyWithChildren'] = ((X['GroupSize'] > 1) & (X['IsChild'] == 1)).astype(int)
    
    # Cabin region features (based on cabin number)
    X['Cabin_Region'] = pd.cut(X['Cabin_Num'], bins=[0, 300, 600, 900, 1200, 1500, 2000], 
                              labels=['Front', 'MidFront', 'Mid', 'MidRear', 'Rear', 'FarRear'])
    
    # Spending patterns
    X['SpendingDiversity'] = X[spending_cols].std(axis=1, skipna=True)
    X['MaxSpendingCategory'] = X[spending_cols].idxmax(axis=1)
    
    # Drop original columns
    X = X.drop(['Cabin', 'GroupId', 'PersonId', 'PassengerId', 'Name'], axis=1, errors='ignore')
    
    return X

# Apply feature engineering to combined data
print("Creating features...")
df_combined_processed = create_features(df_combined)

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

# Enhanced transformers with multiple imputation strategies
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=7)),
    ('scaler', PowerTransformer(method='yeo-johnson'))
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
    X_processed_df, y, test_size=0.15, random_state=42, stratify=y
)

# Feature selection using multiple methods
print("Performing feature selection...")

# Method 1: XGBoost importance
xgb_selector = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_selector.fit(X_train, y_train)

# Method 2: Random Forest importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# Combine feature importances
feature_importance = (
    pd.Series(xgb_selector.feature_importances_, index=feature_names) * 0.6 +
    pd.Series(rf_selector.feature_importances_, index=feature_names) * 0.4
)

# Select top features
top_features = feature_importance.nlargest(40).index
X_train_selected = X_train[top_features]
X_val_selected = X_val[top_features]
X_full_selected = X_processed_df[top_features]
X_test_selected = X_test_processed_df[top_features]

print(f"Selected {len(top_features)} top features")

# Ensemble of models
print("Training ensemble models...")

# Model 1: LightGBM (primary)
lgb_model = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

# Model 2: XGBoost
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Model 3: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Model 4: Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42
)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',
    n_jobs=-1
)

# Train ensemble model
print("Training ensemble model...")
voting_clf.fit(X_full_selected, y)

# Individual model validation (for comparison)
print("\nIndividual model performance:")
models = {
    'LightGBM': lgb_model,
    'XGBoost': xgb_model,
    'RandomForest': rf_model,
    'GradientBoosting': gb_model
}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    val_pred = model.predict(X_val_selected)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"{name} Validation Accuracy: {val_accuracy:.4f}")

# Ensemble validation
val_pred_ensemble = voting_clf.predict(X_val_selected)
val_accuracy_ensemble = accuracy_score(y_val, val_pred_ensemble)
print(f"\nEnsemble Validation Accuracy: {val_accuracy_ensemble:.4f}")

# Cross-validation Score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_clf, X_full_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance visualization
plt.figure(figsize=(12, 8))
feature_importance[top_features].sort_values().tail(20).plot(kind='barh')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Make predictions on test set using ensemble
print("Making predictions on test set...")
test_predictions_proba = voting_clf.predict_proba(X_test_selected)[:, 1]

# Adjust threshold for better performance (optional)
test_predictions = (test_predictions_proba > 0.5).astype(bool)

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
print(submission.head())

# Additional analysis
print(f"\nKey statistics:")
print(f"Training set size: {len(X_full_selected)}")
print(f"Test set size: {len(X_test_selected)}")
print(f"Number of features used: {len(top_features)}")
print(f"Target distribution in training: {y.value_counts(normalize=True).to_dict()}")