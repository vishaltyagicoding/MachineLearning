import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# read dataset and delete colmuns id
df = pd.read_csv('voting ensemble learning\\Phishing_Legitimate_full.csv')
df = df.drop(columns=['id', "HttpsInHostname"])

# check class balance
# print(df['CLASS_LABEL'].value_counts())

# df.info()

# print columns name
# print(df.columns)

# print(df.sample(10))

# check missing values
# print(df.isnull().sum())

# check duplicate values
# print(df.duplicated().sum())

# print(df.describe())

# check unique value in HttpsInHostname columns
# print(df['HttpsInHostname'].unique().sum())
# print(df["HttpsInHostname"])




# see each colum kde plot

# for col in df.columns:
#     sns.kdeplot(df[col])
#     plt.title(col)
#     plt.show()
    
# # see each column histogram plot

# for col in df.columns:
#     sns.histplot(df[col])
#     plt.title(col)
#     plt.show()
    
# # see each column box plot

# for col in df.columns:
#     sns.boxplot(df[col])
#     plt.title(col)
#     plt.show()
    
# # see each column scatter plot

# for col in df.columns:
#     sns.scatterplot(df[col])
#     plt.title(col)
#     plt.show()
    
# # see each column pair plot

# sns.pairplot(df)
# plt.show()

# # see each column heatmap plot

# sns.heatmap(df.corr())
# plt.show()


X = df.drop(columns=['CLASS_LABEL'])
y = df['CLASS_LABEL']

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create voting classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC())
], voting='hard')

# fit voting classifier
voting_clf.fit(X_train, y_train)

# predict on test set
y_pred = voting_clf.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# check cross validation score
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

class VotingClassifier_():
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        self.estimators_ = []

        
    def fit(self, X, y):
        for estimator in self.estimators:
            estimator[1].fit(X, y)
            
        return self

    def predict(self, X):
        pred = []
        if self.voting == 'hard':
            predictions = [estimator[1].predict(X) for estimator in self.estimators]

        for i in range(len(predictions[0])):
            temp = []
            for j in range(len(predictions)):
                temp.append(predictions[j][i])
            pred.append(np.bincount(temp).argmax())
        return np.array(pred)
            

voting_clf = VotingClassifier_(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC())
], voting='hard')


# fit voting classifier
voting_clf.fit(X_train, y_train)

# predict on test set
y_pred = voting_clf.predict(X_test)

# # evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

