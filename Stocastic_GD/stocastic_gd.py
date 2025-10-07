## Stocastic gradient descent.

## Importing libraries
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset.
X, y = load_diabetes(return_X_y = True)
print(X.shape, "\n", y.shape)

## Splitting the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

## Implementing Linear Regression.
reg = LinearRegression()
reg.fit(X_train, y_train)

print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Making predictions on the test set.
y_hat = reg.predict(X_test)


# Evaluating the model.
'''
mean_error = mean_squared_error(y_test, y_hat)
root_mean_error = sqrt(mean_error)
R2_score = r2_score(y_test, y_hat)
print("R2 score:", R2_score)
print("Mean Squared Error:", mean_error)
print("Root Mean Squared Error:", root_mean_error)
'''

## Implementing Stocastic GD class.

class SGDRegressor:
    def __init__(self, alpha = 0.01, epochs = 100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = alpha
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

