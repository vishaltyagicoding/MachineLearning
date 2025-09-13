from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
X, y = load_breast_cancer(return_X_y=True)

print(X.shape, y.shape)
# print feature names
# print(load_breast_cancer().feature_names)
# print(load_breast_cancer().target_names)


clf = AdaBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

y_pred = clf.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))

# check feature importances
# print("Feature importances:", clf.feature_importances_)

# check cross validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# check overfitting by plotting learning curve
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training size')
plt.ylabel('Score')
plt.legend()
plt.show()

# check different way to overfit by plotting validation curve
from sklearn.model_selection import validation_curve
param_range = np.arange(1, 201, 10)
train_scores, test_scores = validation_curve(clf, X, y, param_name="n_estimators", param_range=param_range, cv=5, n_jobs=-1)
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
plt.plot(param_range, train_scores_mean, label='Training score')
plt.plot(param_range, test_scores_mean, label='Cross-validation score')
plt.xlabel('Number of estimators')
plt.ylabel('Score')
plt.legend()
plt.show()



