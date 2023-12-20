import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Initialize variables
N = 5000  # Number of rounds (users)
d = len(dataset.columns)  # Number of ads
ads_selected = []  # List to store ads selected in each round
numbers_of_selections = [0] * d  # List to store number of times each ad was selected
sums_of_rewards = [0] * d  # List to store the sum of rewards of each ad
total_reward = 0  # Total reward

# Implementing the UCB algorithm
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        # Check if the ad was selected at least once
        if numbers_of_selections[i] > 0:
            # Calculate the average reward of ad i
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            # Calculate the confidence interval for ad i
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            # Set a large upper bound for ads that haven't been selected yet
            upper_bound = 1e400

        # Select the ad with the highest upper bound
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    # Append the selected ad to the list
    ads_selected.append(ad)
    # Update the numbers of selections and sums of rewards
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Plotting the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Number of ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
