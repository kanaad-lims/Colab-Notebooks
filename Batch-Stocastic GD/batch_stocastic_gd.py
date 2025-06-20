import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


df = pd.read_csv(
    'D:/Kanaad/Colab-Notebooks/Linear_Regression/scattered_linear_regression_data.csv')

X = df[['startRating']].values
y = df['endRating'].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)  # Predictions made on the basis of input X.

rmse = sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)


print("=== Linear Regression (Initial) ===")
print(f"Coefficient (w): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")
print("Root Mean Square Error: ", rmse)
print("R2 score: ", r2)


class BatchGD:
    def __init__(self, alpha=0.01, epochs=100):
        self.alpha = alpha
        self.epochs = epochs
        self.w = w
        self.b = b
        self.loss_list = []

    def fit(self, X, y):
        n = len(X)

        for i in range(self.epochs):
            # Prediction
            y_pred = self.w * X + self.b

            # Calculating Gradients
            dJ_dw = (-2 / n) * np.sum((y - y_pred) * X)
            dJ_db = (-2 / n) * np.sum((y - y_pred))

            # Updating the w and b.
            self.w = w - (self.alpha * dJ/dw)
            self.b = b - (self.alpha * dJ/db)

            # Tracking the loss
            loss = np.mean((y - y_pred) ** 2)
            self.loss_list.append(loss)

        if i % 10 == 0:
            print(f"Epoch {i}: Loss is: {loss:.4f}")

    def predict(self, X):
        return self.w * X + self.b

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return rmse, r2

    def getParams(self)
    return self.w, self.b


# Creating the BGD object.
bgd_model = BatchGD(alpha=0.01, epochs=100)
bgd_model.fit(X, y)

rmse_bgd, r2_bgd = bgd_model.evaluate(X, y)
w, b = bgd_model.getParams()

print("\n=== Custom Batch Gradient Descent ===")
print(f"w = {w:.4f}, b = {b:.4f}")
print(f"RMSE: {rmse_bgd:.4f}")
print(f"R² Score: {r2_bgd:.4f}")
