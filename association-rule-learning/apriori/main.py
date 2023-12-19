import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for raw in range(0, 7501):
    transactions.append([str(dataset.values[raw, j]) for j in range(0, 20)])

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=00.2, min_lift=3, min_length=2, max_length=2)
