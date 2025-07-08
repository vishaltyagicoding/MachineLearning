from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
import pandas as pd


df = pd.read_csv("train.csv", usecols=['Age','Pclass','Fare','Survived'])

# print(df.sample(5))

x = df.drop(columns=['Survived'])
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = KNNImputer(n_neighbors=3)
knn.fit(x_train, y_train)
knn.fit(x_train)
x_train_transformed = knn.transform(x_train)
x_test_transformed = knn.transform(x_test)

logreg = LogisticRegression()
logreg.fit(x_train_transformed, y_train)

predictions = logreg.predict(x_test_transformed)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)


