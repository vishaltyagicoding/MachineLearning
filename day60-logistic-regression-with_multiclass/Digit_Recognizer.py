import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("day60-logistic-regression-with_multiclass\\train.csv")


# print(df.shape)
# print(df.head())

X = df.iloc[:,1:]
y = df.iloc[:,0]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline = Pipeline(steps=[
                                ("pca", PCA(n_components=150)),
                                ("scale", StandardScaler()),
                                ("lg", LogisticRegression(solver='lbfgs'))])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = model_pipeline.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(round(cross_val_score(model_pipeline, X, y, cv=5).mean(), 4))

