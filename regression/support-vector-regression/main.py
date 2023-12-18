# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Loading the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extracting the independent (X) and dependent (y) variables
X = dataset.iloc[:, 1:-1].values  # Position levels
y = dataset.iloc[:, -1].values  # Salaries
y = y.reshape(len(y), 1)  # Reshaping y into a 2D array for scaling

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  # Scaling X
y = sc_y.fit_transform(y)  # Scaling y

# Creating and fitting the SVR model to the dataset
regressor = SVR(kernel='rbf')  # Using Radial Basis Function (RBF) kernel
regressor.fit(X, y)  # Fitting the model

# Making a prediction for a 6.5 position level
# Transforming 6.5 to the scaled value, predicting and then inverse transforming the prediction
predicted_salary = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))

# Plotting the results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')  # Actual data points
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
         color='blue')  # SVR predictions
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
