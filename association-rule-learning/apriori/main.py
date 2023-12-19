import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for raw in range(0, 7501):
    transactions.append([str(dataset.values[raw, j]) for j in range(0, 20)])

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=00.2, min_lift=3, min_length=2,
                max_length=2)

results = list(rules)
print(results)

resultsinDataFrame = pd.DataFrame(inspect(results),
                                  columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame.nlargest(n=10, columns='Lift'))
