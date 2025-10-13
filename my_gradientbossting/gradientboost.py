from sklearn.tree import DecisionTreeRegressor
import numpy as np


# import data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
X = load_breast_cancer().data
y = load_breast_cancer().target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class GradientBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth  
        self.trees = []

    def fit(self, X, y):
        # Initialize the prediction with the mean of the target variable
        self.f0 = np.mean(y)
        predictions = self.f0
        # print(f)

        for _ in range(self.n_estimators):
            # Calculate the residuals
            residuals = y - predictions
            # print(residuals)

            # Fit a decision tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update the prediction
            predictions += self.learning_rate * tree.predict(X)
            # print(predictions)

            # Add the tree to the ensemble
            self.trees.append(tree)


    def predict(self, X):
        # Initialize the prediction with the mean of the target variable
        f = self.f0
        # print(f)

        # Make predictions using each tree in the ensemble
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
            # print(f)

        return f
        


gb = GradientBoost(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

# check accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred.round()))


