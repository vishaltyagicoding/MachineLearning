import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
import numpy as np

# TODO 1. EDA (Exploratory Data Analysis)
df =  pd.read_csv("Machine_Learning_Models\\heart.csv")
# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())

cat_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_columns = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# for col in cat_columns:
#     print(df[col].value_counts())


# KDE plots for numeric columns


# for col in num_columns:
#     plt.figure(figsize=(10, 5))
#     sns.kdeplot(x=df[col])
#     plt.title(f'Kdeplot of {col}')
#     plt.show()

# Box plots for numeric columns
# for col in num_columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# Heatmap for correlation
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()
# print("Before Imputation:")
# print(df["Cholesterol"].value_counts())

# count plot for categorical columns
# for col in cat_columns:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(x=df[col])
#     plt.title(f'Count Plot of {col}')
#     plt.xticks(rotation=45)
#     plt.show()
#     print(df[col].value_counts())


# TODO 2. Data Preprocessing
from sklearn.model_selection import train_test_split
df_cleaned = df.copy()

# Removing duplicates
df_cleaned.drop_duplicates(inplace=True)


# print(df.describe().T)
df_before_imputation = df.copy()

# ditect outliers in numeric columns
def detect_outliers_iqr(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])
        
    
# print("Outliers detected using IQR method:")
# detect_outliers_iqr(df, num_columns)
# print("Before Imputation:")
# print(df[num_columns].isnull().sum())


# fill 0 values in numeric columns with knn values
columns_values_replace_by_Null = ["Age", "RestingBP", "Cholesterol", "MaxHR"]

for col in columns_values_replace_by_Null:
    df[col] = df[col].replace(0, np.nan)





from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[num_columns] = imputer.fit_transform(df[num_columns])

# print(df.describe().T)
# print("After Imputation:")
# print(df[num_columns].isnull().sum())
def plot_distribution_before_after_imputation(df_before_imputation, df, columns_values_replace_by_Null):
    for col in columns_values_replace_by_Null:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        # Plotting the distribution of 'Cholesterol' before and after imputation
        # sns.kdeplot(df_before_imputation[col], ax=axes[0], fill=True)
        sns.boxplot(x=df_before_imputation[col], ax=axes[0])
        axes[0].set_title(f'{col} Distribution Before Imputation')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density')

        # sns.kdeplot(df[col], ax=axes[1], fill=True, color='green')
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'{col} Distribution After Imputation')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Density')
        plt.tight_layout()
        plt.show()


# plot_distribution_before_after_imputation(df_before_imputation, df, columns_values_replace_by_Null)

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']
# print(X.shape)
# print(X.info())


# One Hot Encoding for categorical columns
process = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), cat_columns),
                  ('scale', StandardScaler(), num_columns)],
    remainder='passthrough')

X = process.fit_transform(X)

# Convert the transformed data back to a DataFrame
X = pd.DataFrame(X, columns=process.get_feature_names_out())

# print(X.shape)
# print(X.info())
# print(X.head().T)













# # TODO 3. Feature Engineering and Extraction
# # Feature Engineering
all_columns = X.columns.tolist()
correlation = X.corrwith(y).sort_values(ascending=False)
# print("Correlation with target variable:")
# print(correlation)


from scipy.stats import chi2_contingency
alpha = 0.05
chi2_results = {}

for col in all_columns:
    contingency = pd.crosstab(X[col], y)
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
# print(chi2_df)

drop_features = chi2_df[chi2_df['Decision'] == 'Accept Null (Drop Feature)'].index.tolist()
# print("Final Features to Keep:", final_features)

# X.drop(columns=drop_features, inplace=True)




# TODO 4. Model Training and Evaluation
def check_model_performance(model, X, y):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import PowerTransformer, PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)


    # Train the model
    model.fit(X_train, y_train)
    print(X_test.shape)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"Model Accuracy: {round(accuracy, 4)}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    from sklearn.model_selection import cross_val_score
    cross_val_score = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores:", cross_val_score.mean(), "\n")



from sklearn.linear_model import LogisticRegression

# Logistic Regression Model

model = LogisticRegression(max_iter=100)
check_model_performance(model, X, y)

# K-Nearest Neighbors Model
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
check_model_performance(knn_model, X, y)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
check_model_performance(dt_model, X, y)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
check_model_performance(rf_model, X, y)

# Support Vector Classifier
from sklearn.svm import SVC
svc_model = SVC(kernel='linear', random_state=42)
check_model_performance(svc_model, X, y)

# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
check_model_performance(gnb_model, X, y)
# XGBoost Classifier
from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
check_model_performance(xgb_model, X, y)








# df_cleaned.drop_duplicates(inplace=True)
# # print(df_cleaned.duplicated().sum())

# # Encoding categorical variables
# # for i in cat_columns:
# #     print(X[i].value_counts())

# df_cleaned["Sex"] = df_cleaned["Sex"].map({"M": 0, "F": 1})
# df_cleaned["ExerciseAngina"] = df_cleaned["ExerciseAngina"].map({"N": 0, "Y": 1})
# # print(df_cleaned["Sex"].value_counts())
# # print(df_cleaned["HeartDisease"].head())

# # Encoding categorical variables using one-hot encoding
# cat_columns_ohe = ["ChestPainType", "RestingECG","ST_Slope"]

# ct = ColumnTransformer(transformers=[('cat', OneHotEncoder(drop="first"), cat_columns_ohe)], remainder='passthrough')
# df_cleaned_encode = ct.fit_transform(df_cleaned)
# df_cleaned= pd.DataFrame(df_cleaned_encode, columns=ct.get_feature_names_out())

# # print(df_cleaned["remainder__HeartDisease"].head())
# # print(df_cleaned.info())
# # print(df_cleaned.shape)
