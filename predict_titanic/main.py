import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

titanic = sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "sibsp","parch", "fare", "embarked"]]
titanic.fillna({'age': titanic["age"].mean(), 'embarked': titanic["embarked"].mode()[0]}, inplace=True)
# print(titanic.sample(10))
print(titanic.info())
print(titanic.isnull().sum())

x_train, x_test, y_train, y_test = train_test_split(titanic.drop(columns=["survived"]), titanic["survived"], test_size=0.2, random_state=0)


update_missing_value = ColumnTransformer([

                                            ("sex_", OneHotEncoder(sparse_output=False, drop="first"), [1, 6]),
                                            ("scale", StandardScaler(), ["pclass", "age", "fare"])
                                          ],
                                         remainder='passthrough')


model = LogisticRegression()

pipe = Pipeline([
    ('trf1',update_missing_value),
    ('trf3',model),

])

# train
pipe.fit(x_train,y_train)

# Predict
predictions = pipe.predict(x_test)




# x_train_update = update_missing_value.fit_transform(x_train)

# print(x_train_update)
# x_test_update = update_missing_value.fit_transform(x_test)

# print(x_train_update)

# model = DecisionTreeClassifier()

# model.fit(x_train_update, y_train)

# predictions = model.predict(x_test_update)
# print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))