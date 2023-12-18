# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Loading the dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Extracting the independent variable (Position Level) and dependent variable (Salary) from the dataset
X = dataset.iloc[:, 1:-1].values  # Independent Variable
y = dataset.iloc[:, -1].values  # Dependent Variable

# Creating and fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Creating polynomial features of degree 4
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Creating and fitting a linear regression model with polynomial features
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Plotting the linear regression results
plt.scatter(X, y, color="red")  # Actual data points
plt.plot(X, lin_reg.predict(X), color="blue")  # Linear regression prediction
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Plotting the polynomial regression results
plt.scatter(X, y, color="red")  # Actual data points
plt.plot(X, lin_reg_poly.predict(X_poly), color="blue")  # Polynomial regression prediction
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_poly.predict(poly_reg.fit_transform([[6.5]])))
