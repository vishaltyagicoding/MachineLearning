from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
titanic = sns.load_dataset("titanic")[['survived', 'age', 'fare']]
# print(titanic.isnull().sum())
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())
# print(titanic.isnull().sum())
# print(titanic.info())

x = titanic.drop("survived", axis=1)
y = titanic['survived']
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tnf = ColumnTransformer([("first", KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile"), ["age"]),
                         ("second", KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile"), ["fare"])])

tnf.fit(x_train)

x_train_trans = tnf.transform(x_train)
x_test_trans = tnf.transform(x_test)

# print(tnf.named_transformers_['first'].bin_edges_)

dt = DecisionTreeClassifier()

dt.fit(x_train_trans, y_train)
y_pred = dt.predict(x_test_trans)

print(accuracy_score(y_test, y_pred))