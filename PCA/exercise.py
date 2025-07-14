import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def remove_outliers(column):

    col = column.name

    # Q1 = df.quantile(0.25)
    # Q3 = df.quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # d =  df[(df >= lower_bound) & (df <= upper_bound)]

    # highest = df.mean() + 3 * df.std()
    # lowest = df.mean() - 3 * df.std()
    #
    # print(highest)
    # print(lowest)
    #
    # df = df[(df <= highest) & (df >= lowest)]
    cgpa_25_percentile = np.percentile(column, 25)
    cgpa_75_percentile = np.percentile(column, 75)

    # print(cgpa_25_percentile)
    # print(cgpa_75_percentile)
    #
    # print(df["cgpa"].describe())

    cgpa_iqr = cgpa_75_percentile - cgpa_25_percentile
    # print(cgpa_iqr)

    higher_value = cgpa_75_percentile + (cgpa_iqr * 1.5)
    lower_value = cgpa_25_percentile - (cgpa_iqr * 1.5)

    # caping

    column = np.where(column > higher_value,higher_value, np.where(column < lower_value, lower_value, column))
    return pd.DataFrame(column, columns=[col])


def box_plot(before_data, after_data):
    plt.figure(figsize=(10, 4))

    # Plot before transformation
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    sns.boxplot(before_data, color='blue', label='Before', fill=True)
    plt.title("Before Transformation")

    # Plot after transformation
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    sns.boxplot(after_data, color='red', label='After', fill=True)
    plt.title("After Transformation")

    plt.tight_layout()
    plt.show()


def handle_data():

    titanic = pd.read_csv("titanic.csv")[["Survived","Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    # print(titanic.shape)
    # print(titanic.sample(5))

    # find percentage of missing values
    # print(titanic.isnull().mean()*100)

    # if data is missing less than 5% you can delete those rows or fill by the (mena, median, mode)
    # check missing values are random or not
    # print(titanic[titanic["Embarked"].isnull()])
    # print(titanic[titanic["Age"].isnull()])

    titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])
    # titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
    # titanic.dropna(inplace=True)

    x = titanic.drop(columns="Survived")
    y = titanic["Survived"]
    # print(x.sample(5))

    imputer = KNNImputer(n_neighbors=2)
    df_clean = pd.DataFrame(imputer.fit_transform(x.select_dtypes(include=['float64', 'int64'])),
                           columns=x.select_dtypes(include=['float64', 'int64']).columns)

    x["Age"] = df_clean["Age"]




    # print(df_clean["Age"].isnull().sum())
    # print(df_clean.sample(5))
    #
    # print(titanic["Embarked"].isnull().sum())
    # print(titanic["Age"].isnull().sum())

    x["Family"] = x["SibSp"] + x["Parch"] + 1

    x.drop(columns=["SibSp", "Parch"], inplace=True)

    def myfunc(num):
        if num == 1:
            #alone
            return 0
        elif 1 < num <= 4:
            # small family
            return 1
        else:
            # large family
            return 2


    x["Family"] = x["Family"].apply(myfunc)

    return x, y








def analysis(before_data, after_data):

    plt.figure(figsize=(10, 4))

    # Plot before transformation
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    sns.kdeplot(before_data, color='blue', label='Before', fill=True)
    plt.title("Before Transformation")

    # Plot after transformation
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    sns.kdeplot(after_data, color='red', label='After', fill=True)
    plt.title("After Transformation")

    plt.tight_layout()
    plt.show()



def transformation():
    x, y = handle_data()

    categorical_cols = ["Sex", "Embarked"]
    numerical_cols = ["Pclass", "Age", "Fare", "Family"]
    # for column_name in numerical_cols:
    #     # be = x[column_name]
    #     x[column_name] = remove_outliers(x[column_name])
    #     # print("df_cleaned")
    #     # af = x[column_name]
    #     # box_plot(be, af)
    #     # print(df_cleaned)



    preprocessor  = ColumnTransformer(remainder="passthrough", transformers=[("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
                                                                             ("Scale", StandardScaler(), numerical_cols),
                                                                             ("pt", PowerTransformer(), numerical_cols)])

    clf_trans = preprocessor.fit_transform(x)

    all_columns = preprocessor.get_feature_names_out()

    df = pd.DataFrame(clf_trans, columns=all_columns)

    # analysis(x["Fare"], df["pt__Fare"])
    # analysis(x["Age"], df["pt__Age"])
    # analysis(x["Family"], df["pt__Family"])
    # analysis(x["Pclass"], df["pt__Pclass"])



    # print(df.sample(5))
    # print(df.shape)



    pca = PCA(n_components=6)
    return pca.fit_transform(df) , y


def let_prediction():

    pca_trans, y = transformation()

    x_train, x_test, y_train, y_test = train_test_split(pca_trans, y, test_size=0.2, random_state=42)

    # logreg = LogisticRegression()
    logreg = KNeighborsClassifier()
    # logreg = DecisionTreeClassifier()
    # logreg = RandomForestClassifier()
    # logreg = AdaBoostClassifier()
    # logreg = GradientBoostingClassifier()
    # logreg = XGBClassifier()
    logreg.fit(x_train, y_train)

    prediction = logreg.predict(x_test)

    print(round(accuracy_score(y_test, prediction) * 100, 3))


let_prediction()








