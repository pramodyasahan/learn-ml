import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Initialize variables
N = 10000  # Number of rounds (users)
d = 10  # Number of ads
ads_selected = []  # List to store ads selected in each round
numbers_of_rewards_1 = [0] * d  # List to store the number of times the ad i got reward 1
numbers_of_rewards_0 = [0] * d  # List to store the number of times the ad i got reward 0
total_reward = 0  # Total reward

# Implementing the Thompson Sampling algorithm
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        # Generate a random draw from the beta distribution for ad i
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        # Select the ad with the highest beta distribution value
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    # Update the rewards
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Plotting the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Number of ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
