import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np

df =  pd.read_csv("Machine_Learning_Models\\heart.csv")

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

cat_columns = X.select_dtypes(include=['object']).columns.tolist()
num_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# print(X[num_columns] .isin([0]).mean())

for col in ['Cholesterol', 'RestingBP']:
    X[col] = X[col].replace(0, np.nan)

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
X[num_columns] = imputer.fit_transform(X[num_columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

process = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', handle_unknown="ignore"), cat_columns),
    ('scale', StandardScaler(), num_columns),
], remainder='passthrough')

# Define a function to check model performance
def check_model_performance(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {round(accuracy_score(y_test, y_pred), 4)}%")


first_model = Pipeline(steps=[
    ('preprocessor', process),
    ("poy", PolynomialFeatures(degree=2, include_bias=False)),
    ("PowerTransform", FunctionTransformer()),
    ('classifier', LogisticRegression())
])

second_model = Pipeline(steps=[
    ('preprocessor', process),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

third_model = Pipeline(steps=[
    ('preprocessor', process),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

fourth_model_pipline = Pipeline(steps=[
    ('preprocessor', process),
    ('classifier', RandomForestClassifier(random_state=42))
])

fifth_model_pipline = Pipeline(steps=[
    ('preprocessor', process),
    ('classifier', SVC(kernel='linear', random_state=42))
])

sixth_model_pipline = Pipeline(steps=[
    ('preprocessor', process),
    ('classifier', GaussianNB())
])

print("Linear Regression Model Performance:")
check_model_performance(first_model)


























