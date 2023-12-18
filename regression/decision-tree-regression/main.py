# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Loading the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extracting the independent variable (X) and dependent variable (y)
X = dataset.iloc[:, 1:-1].values  # Position levels
y = dataset.iloc[:, -1].values  # Salaries

# Creating and fitting the Decision Tree Regression model
regressor = DecisionTreeRegressor(max_depth=5, random_state=0)
regressor.fit(X, y)

# Making a prediction for a specific position level (6.5)
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Creating a finer grid of X values for plotting the decision tree regression results
X_flat = X.flatten()  # Flattening X to a 1D array
X_grid = np.arange(min(X_flat), max(X_flat), 0.1)  # Generating a range of values for X
X_grid = X_grid.reshape(len(X_grid), 1)  # Reshaping back to a 2D array for compatibility

# Plotting the original data points
plt.scatter(X, y, color='red')

# Plotting the regression model's predictions
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

# Adding title and labels to the plot
plt.title('Decision Tree Regressor')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
