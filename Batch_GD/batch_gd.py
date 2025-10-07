## Batch stocastic gradient descent.

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

## Creating GD Class.
class BatchGD:
    def __init__(self, alpha = 0.01, epochs = 100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = alpha
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0
        #print(self.coef_)
        #print(self.intercept_)
        
        ## Updating the intercept first.
        for i in range(self.epochs):
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)

        ## Updating the coefficient.
        for i in range(self.epochs):
            coef_der = -2 * np.dot((y_train - y_hat), X_train)/X_train.shape[0]
            ## We find the derivatives of all the features (columns (X)) in a single operation using the above formula.
            self.coef_ = self.coef_ - (self.lr * coef_der)
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
        
        print("\nFinal Coefficient: ", self.coef_)
        print("Final Intercept: ", self.intercept_)
                
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_

gdr = BatchGD(alpha = 0.45, epochs=1500)
gdr.fit(X_train, y_train)


gdr_predict = gdr.predict(X_test)
R2Score = r2_score(y_test, gdr_predict)
print("R2 Score:", R2Score)
