import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import cross_val_score


# Load the dataset
df =  pd.read_csv("Machine_Learning_Models\\heart.csv")

# Prepare the features and target variable(independent and dependent variables)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Identify categorical and numerical columns
cat_columns = X.select_dtypes(include=['object']).columns.tolist()
num_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# print(X[num_columns] .isin([0]).mean())

# replace 0 values in specific columns with nan
for col in ['Cholesterol', 'RestingBP']:
    X[col] = X[col].replace(0, np.nan)



# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# column transformer help to apply specific transformations to specific columns
# One Hot Encoding for categorical columns
# Scaler for numerical columns
process = ColumnTransformer(transformers=[
    ("cat", Pipeline(steps=[("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat_columns),
    ("num", Pipeline(steps=[("impute", KNNImputer(n_neighbors=5)), 
                            ("scale", StandardScaler())]), num_columns)
    
], remainder='passthrough')

# create a pipeline for the model
model = Pipeline(steps=[
    ('preprocessor', process),
    ("poy", PolynomialFeatures(degree=2, include_bias=False)),
    ('classifier', LogisticRegression())
])


print("Linear Regression Model Performance:")
# train linear model by the calling fit method
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Model Accuracy: {round(accuracy_score(y_test, y_pred), 2) * 100}%")

# check cross-validation score
cross_val_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cross_val_scores.mean() * 100:.2f}%")

# check the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# check the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# create pickled model
import pickle
with open("Machine_Learning_Models/heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)





















