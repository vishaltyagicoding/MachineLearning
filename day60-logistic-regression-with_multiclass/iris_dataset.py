from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


df = sns.load_dataset("iris")
df = df[['sepal_length','petal_length','species']]

# print(df.sample(5))
# print(df["species"].value_counts())

# train to split 
x = np.array(df.iloc[:,0:2])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=42)

# label encoder
lable = LabelEncoder()

lable.fit(y_train)

y_train = lable.transform(y_train)
y_test = lable.transform(y_test)

lgmodel = LogisticRegression()
lgmodel.fit(X_train,y_train)

y_pred = lgmodel.predict(X_test)

print(accuracy_score(y_test,y_pred))

print(pd.DataFrame(confusion_matrix(y_test,y_pred)))

print(classification_report(y_test,y_pred))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train, lgmodel, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.title('Softmax on Iris')

plt.show()