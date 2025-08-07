import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Machine_Learning_Models\\insurance.csv")
# print(df.head())

#TODO 1. EDA (Exploratory Data Analysis)
# print(df.describe())
# print(df.info())
# print(df.isnull().sum())
# print(df.shape)
# print(df.columns)
# print(df["charges"].head())

numeric_columns = ['age', 'bmi', 'children','charges']
categorical_columns = ['sex', 'smoker', 'region']

# kdeplot(kernel density estimate plot)
# for col in numeric_columns:
#     plt.figure(figsize=(10, 5))
#     sns.kdeplot(x=df[col])
#     plt.title(f'Kdeplot of {col}')
#     plt.show()

# countplot
# for col in categorical_columns:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(x=df[col])
#     plt.title(f'Countplot of {col}')
#     plt.show()

# boxplot
# for col in numeric_columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()

# Heatmap for correlation
# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.title('Correlation Heatmap')
# plt.show()

# print(df.corr(numeric_only=True))




# TODO 2. Data Preprocessing
df_cleaned = df.copy()
# df_cleaned.drop_duplicates(inplace = True)
# print(df_cleaned.duplicated().sum())

# dfc = df_cleaned['sex'].value_counts()
# print(dfc)





# Encoding categorical variables
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# first way
# print(df_cleaned['region'].value_counts())
# print(df_cleaned["sex"].value_counts())
# print(df_cleaned["smoker"].value_counts())



process = ColumnTransformer(remainder='passthrough',
                            transformers=[('cat', OneHotEncoder(drop="first"), ["region"]),
                                          ('ord_sex', OrdinalEncoder(), ['sex']),
                                          ('ord_smoker', OrdinalEncoder(), ['smoker'])])

df_cleaned = process.fit_transform(df_cleaned)
df_cleaned = pd.DataFrame(df_cleaned, columns=process.get_feature_names_out())
# print(df_cleaned.head())
# print(df_cleaned["ord_sex__sex"].head())

# print(df_cleaned.info())




# second way
# df_cleaned['sex'] = df_cleaned['sex'].map({"male" : 0,"female" : 1})
# df_cleaned['smoker'] = df_cleaned['smoker'].map({"no" : 0,"yes" : 1})



# rename columns names
# df_cleaned.rename(columns={
#     'sex' :'is_female',
#     'smoker': 'is_smoker'
#                           },inplace = True)
# df_cleaned = df_cleaned.astype(int)



# TODO 3. Feature Engineering and Extraction



# Creating new features
df_cleaned["bmi_category"] = pd.cut(df_cleaned['remainder__bmi'], bins=[0, 18.5, 24.9, 29.9, np.inf],
                                     labels=['underweight', 'normal', 'overweight', 'obese'])

# print(df_cleaned["bmi_category"].value_counts())
# print(df_cleaned.head())




ohe = OneHotEncoder()
bmi_category_encoded = ohe.fit_transform(df_cleaned[['bmi_category']])
bmi_category_encoded_df = pd.DataFrame(bmi_category_encoded.toarray(), columns=ohe.get_feature_names_out(['bmi_category']))
df_cleaned = pd.concat([df_cleaned, bmi_category_encoded_df], axis=1)
df_cleaned.drop(columns=['bmi_category'], inplace=True)



# standard scaling
scaler = StandardScaler()
cols = ['remainder__age', 'remainder__bmi', 'remainder__children']
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
# print(df_cleaned["remainder__age"].head())
# print(df_cleaned['remainder__charges'].head())

# find correlation with target variable
correlation = df_cleaned.corr(numeric_only=True)['remainder__charges'].sort_values(ascending=False)
# print(correlation)
# print(df_cleaned.columns)



# check cai square test features correlation with target variable
all_feature = df_cleaned.columns.tolist()
all_feature.remove('remainder__charges')

from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05

df_cleaned['remainder__charges_bin'] = pd.qcut(df_cleaned['remainder__charges'], q=4, labels=False)
chi2_results = {}

for col in all_feature:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['remainder__charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)

drop_features = chi2_df[chi2_df['Decision'] == 'Accept Null (Drop Feature)'].index.tolist()
# print("Final Features to Keep:", final_features)

df_cleaned.drop(columns=drop_features, inplace=True)





# TODO 4. Model Training and Evaluation
from sklearn.model_selection import train_test_split
X = df_cleaned.drop(columns=['remainder__charges'])
y = df_cleaned['remainder__charges']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# implementing the power transform
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
pt.fit(X_train)
x_train = pt.transform(X_train)
x_test = pt.transform(X_test)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred))

# knn Regression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN Regression R2 Score:", r2_score(y_test, y_pred))




# cat_features = ['cat__region_northwest',
#        'cat__region_southeast', 'cat__region_southwest', 'ord_sex__sex',
#        'ord_smoker__smoker','bmi_category_normal',
#        'bmi_category_obese', 'bmi_category_overweight',
#        'bmi_category_underweight']