import numpy as np
import pandas as pd
from dataset import load_dataframe

# Προ-επεξεργασία Δεδομένων
# Ερώτημα 2

def R_dataframe():
    # Load the DataFrame
    X = load_dataframe()
    # Count the occurrences of each user
    user_counts = X.iloc[:, 0].value_counts()

    # Filter users who appear between Rmin = 3 and Rmax = 50 times (length: 358, for Rmax = 2 we'd have 8662)
    # no one has given more than 29 ratings
    filtered_users = user_counts[(user_counts >= 3) & (user_counts <= 50)]

    # Create R dataframe with rows of users appearing in filtered_users
    R = X[X.iloc[:, 0].isin(filtered_users.index)]
    # Splitting the single column into separate columns
    R = R[0].str.split(',', expand=True)
    R.columns = ['Users', 'Items', 'Ratings', 'Timestamp']

    # Convert to numeric values
    R['Ratings'] = pd.to_numeric(R['Ratings'])


    R['Users'] = R['Users'].str.extract('(\d+)').astype(int)
    R['Items'] = R['Items'].str.extract('(\d+)').astype(int)

    # Convert 'Timestamp' column to datetime object
    R['Timestamp'] = pd.to_datetime(R['Timestamp'])
    return R


# Print R
R = R_dataframe()
print(R)
