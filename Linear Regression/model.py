## Linear Regression.

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('scattered_linear_regression_data.csv')


def lossFunction(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].startRating
        y = points.iloc[i].endRating
        
        total_error += (y - (m * x + b)) ** 2
    
    total_error / float(len(points))
 
    

def gradient_descent(m_now, b_now, points, omega):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].startRating
        y = points.iloc[i].endRating
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    
    m = m_now - (m_gradient * omega)
    b = b_now - (b_gradient * omega)
    return m, b

m = 0
b = 0
omega = 0.0001
epochs = 650

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, omega)

print(m, b)

plt.title("Linear Regression Model")
plt.xlabel("startRating")
plt.ylabel("endRating")
plt.scatter(data.startRating, data.endRating, color="black")
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color = "red")
plt.show()
