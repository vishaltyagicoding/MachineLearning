import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv("Machine Learning Projects\\Loan Default Prediction\\Loan_default.csv").drop(columns=["LoanID"])
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())

# how many 0 in each columns
# print(df.isin([0]).sum())

# for col in df.columns:
#     print(df[col].describe())

# print(df.corr(numeric_only=True))

# df.drop_duplicates(inplace=True)

num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_columns = df.select_dtypes(include=['object']).columns.tolist()

# create kdeplot
# for col in num_columns:
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(x = df[col])
#     plt.title(f"kdeplot of {col}")
#     plt.show()

# create boxplot

# for col in num_columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(df[col])
#     plt.title(col)
#     plt.show()


# Heatmap for correlation
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()



# count plot for categorical columns
# for col in cat_columns:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(x=df[col])
#     plt.title(f'Count Plot of {col}')
#     plt.xticks(rotation=45)
#     plt.show()
#     print(df[col].value_counts())

# print(df["Default"].value_counts())

X = df.drop(columns=['Default'])
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

process = ColumnTransformer(transformers=[("power", PowerTransformer(), X.select_dtypes(include=['int64', 'float64']).columns.tolist()),
                                          ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), X.select_dtypes(include=['object']).columns.tolist()),
                                          ("scale", StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns.tolist())]
                            , remainder="passthrough")

model = Pipeline(steps=[("pro", process),
                        ("model", BaggingClassifier(LogisticRegression(penalty='l2'), n_estimators=10, max_samples=0.5))])

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# check model performance
ac = accuracy_score(y_test, y_pred)
print(f"Score: {ac:.4f}")

cv_scores = cross_val_score(model, X, y, cv=5)
print(cv_scores)
print(f"Cross-validation scores: {cv_scores.mean() * 100:.2f}%")









