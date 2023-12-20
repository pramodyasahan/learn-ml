import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
