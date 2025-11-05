import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the data
train = pd.read_csv("kaggle competition\\data_science_london\\train (2).csv")
test = pd.read_csv("kaggle competition\\data_science_london\\test.csv")
train_labels = pd.read_csv("kaggle competition\\data_science_london\\trainLabels.csv")

# Reset column names to simple integers for both train and test
train.columns = [str(i) for i in range(train.shape[1])]
test.columns = [str(i) for i in range(test.shape[1])]

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train labels shape: {train_labels.shape}")

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Create and train the model with pipeline
final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(
        n_estimators=250, 
        random_state=42, 
        learning_rate=0.2, 
        max_depth=5
    ))
])

# Train the model on training data
final_model.fit(X_train, y_train.values.ravel())

# Validate on validation set
y_val_pred = final_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# check rmse
roc_auc_score = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC Score: {roc_auc_score:.4f}")


# Make predictions on test set
test_predictions = final_model.predict(test)

# Create submission file
submission = pd.DataFrame({
    'Id': range(1, len(test_predictions) + 1),
    'Solution': test_predictions
})

# Save to CSV
submission.to_csv("kaggle competition\\data_science_london\\submission.csv", index=False)

