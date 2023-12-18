# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading the dataset from a CSV file
dataset = pd.read_csv('Social_Network_Ads.csv')

# Extracting features (X) and target variable (y) from the dataset
X = dataset.iloc[:, :-1].values  # Selecting all rows and all but the last column as features
y = dataset.iloc[:, -1].values  # Selecting all rows and only the last column as the target variable

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # 25% data for testing

