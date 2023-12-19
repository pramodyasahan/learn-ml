import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for raw in range(0, 7501):
    transactions.append([str(dataset.values[raw, j]) for j in range(0, 20)])
