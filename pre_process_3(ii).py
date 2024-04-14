import numpy as np
import pandas as pd
from dataset import load_dataframe
from pre_process_2 import R_dataframe
import matplotlib.pyplot as plt

# 3(ii). Ιστόγραμμα συχνότητας για το χρονικό εύρος
# των αξιολογήσεων του κάθε χρήστη.


# Load the updated dataframe that occurred from Q2 where Rmin = 3 and Rmax = 50
R = R_dataframe()

# Grouping by the values of ratings and users
#grouped = R.groupby('Timestamp')['Users'].nunique()

# Plotting the histogram
user_timestamp_range = (R.groupby('Users')['Timestamp']
                        .agg(lambda x: (x.max() - x.min()).days))

values, bins, _ = plt.hist(user_timestamp_range, bins=50)

# Showing only time ranges that matches user activity (non-zero values).
nonzero_indices = [i for i, val in enumerate(values) if val != 0]
plt.xticks(bins[nonzero_indices])

# Showing count of users for each time range
for i in range(len(bins) - 1):
    if values[i] != 0:  # Check if the count is non-zero
        plt.text(bins[i] + (bins[i+1] - bins[i]) / 2, values[i], str(int(values[i])), ha='center', va='bottom')

# Setting labels and title
plt.xlabel('Time range in days')
plt.ylabel('No. of Users')
plt.title('Histogram')

# Display the plot
plt.show()