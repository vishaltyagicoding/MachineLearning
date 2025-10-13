import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import xgboost
from catboost import CatBoostClassifier

df = pd.read_csv("C:\\Users\\alber\\Documents\\Cursor_Machine_Learning_Clone\\MachineLearning\\Stacking Ensemble Learning\\mnist_train.csv")

X = df.drop(columns=['label'])
y = df['label']


# Convert to numpy arrays if needed
X_array = X.values if isinstance(X, pd.DataFrame) else X
y_array = y.values if isinstance(y, pd.Series) else y

# Split into 4 equal parts
X_parts = np.array_split(X_array, 4)
y_parts = np.array_split(y_array, 4)

# Access each part
X_part1, X_part2, X_part3, X_part4 = X_parts
y_part1, y_part2, y_part3, y_part4 = y_parts

print(f"Part 1: {len(X_part1)} samples")
print(f"Part 2: {len(X_part2)} samples")
print(f"Part 3: {len(X_part3)} samples")
print(f"Part 4: {len(X_part4)} samples")

# Define base classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
    'SVM': SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        random_state=42,
        probability=True
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    ),
    'LGBM': LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    "XGBOOST": xgboost.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
}

y_pred_a = []

# train all the classifiers
for name, classifier in classifiers.items():
    print(f"Training {name}...")
    classifier.fit(X_part1, y_part1)
    y_pred = classifier.predict(X_part2)
    y_pred_a.append(y_pred)
    print(f"{name}: {accuracy_score(y_part2, y_pred)}")

# create a new dataframe with the predictions
df_pred = pd.DataFrame(y_pred_a).T
df_pred.columns = classifiers.keys()
print("\nFirst 5 rows of predictions:")
print(df_pred.head())

print("\nFirst 5 rows with true labels:")
print(df_pred.head())

# create 4 different models for each classifier - FIXED SYNTAX
models = {
    "CatBoost": CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        verbose=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    "XGBoost": xgboost.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
}

y_pred_a = []

# train all the classifiers
for name, classifier in models.items():
    print(f"Training {name}...")
    classifier.fit(df_pred, y_part2)
    y_pred = classifier.predict(X_part3)
    y_pred_a.append(y_pred)
    print(f"{name}: {accuracy_score(y_part3, y_pred)}")

# create a new dataframe with the predictions
df_pred_meta = pd.DataFrame(y_pred_a).T
df_pred_meta.columns = models.keys()
print("\nFirst 5 rows of meta predictions:")
print(df_pred_meta.head())

print("\nFirst 5 true labels from Part 3:")
print(y_part3[:5].values)

# create meta model
meta_model = LogisticRegression(max_iter=1000)

meta_model.fit(df_pred_meta, y_part3)
y_pred = meta_model.predict(X_part4)
print(f"Final Meta model accuracy on Part 4: {accuracy_score(y_part4, y_pred)}")