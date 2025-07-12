import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

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
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
# titanic.dropna(inplace=True)

x = titanic.drop(columns="Survived")
y = titanic["Survived"]


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

# print(titanic.sample(5))

categorical_cols  = ["Sex","Embarked"]
numerical_cols = ["Pclass", "Age", "Fare", "Family"]


preprocessor  = ColumnTransformer(remainder="passthrough", transformers=[("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
                                                                         ("Scale", StandardScaler(), numerical_cols)])

clf_trans = preprocessor.fit_transform(x)

all_columns = preprocessor.get_feature_names_out()

df = pd.DataFrame(clf_trans, columns=all_columns)
# print(df.sample(5))
# print(df.shape)


pca = PCA(n_components=6)
pca_trans = pca.fit_transform(df)

x_train, x_test, y_train, y_test = train_test_split(pca_trans, y, test_size=0.2, random_state=42)


logreg = KNeighborsClassifier()
logreg.fit(x_train, y_train)

pred = logreg.predict(x_test)

print(round(accuracy_score(y_test, pred)*100, 3))












