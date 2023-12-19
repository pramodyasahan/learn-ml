import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori


# Function to extract data from the results object
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]  # Left hand side items
    rhs = [tuple(result[2][0][1])[0] for result in results]  # Right hand side items
    supports = [result[1] for result in results]  # Support values
    return list(zip(lhs, rhs, supports))


# Loading the dataset from a CSV file
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Preparing the list of transactions
transactions = []
for i in range(0, 7501):  # Looping over each transaction
    transactions.append(
        [str(dataset.values[i, j]) for j in range(0, 20)])  # Converting all items in the transaction to strings

# Running Apriori algorithm to generate rules
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2)

# Converting rules to a list
results = list(rules)
print(results)

# Converting the results into a pandas DataFrame for better visualization
resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# Printing the top 10 association rules sorted by support
print(resultsinDataFrame.nlargest(n=10, columns='Support'))
