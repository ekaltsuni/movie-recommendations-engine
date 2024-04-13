import numpy as np
import pandas as pd
from dataset import load_dataframe

# Προ-επεξεργασία Δεδομένων
# Ερώτημα 2

# Load the DataFrame
X = load_dataframe()

# Count the occurrences of each user
user_counts = X.iloc[:, 0].value_counts()

# Filter users who appear between Rmin = 3 and Rmax = 50 times (length: 358, for Rmax = 2 we'd have 8662)
# no one has given more than 29 ratings
filtered_users = user_counts[(user_counts >= 3) & (user_counts <= 50)]

# Create R dataframe with rows of users appearing in filtered_users
R = X[X.iloc[:, 0].isin(filtered_users.index)]

# Print the number of occurrences for each user
# print("Number of occurrences for each user:")
# print(filtered_users)
