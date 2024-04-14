import numpy as np
import pandas as pd
from dataset import load_dataframe
from pre_process_2 import R_dataframe
import matplotlib.pyplot as plt

# Προ-επεξεργασία Δεδομένων
# 3(i). Ιστόγραμμα συχνότητας για το πλήθος των αξιολογήσεων του κάθε χρήστη.

# Load the updated dataframe that occurred from Q2 where Rmin = 3 and Rmax = 50
R = R_dataframe()

# Grouping by the values of ratings and users
ratings_per_user = R.groupby('Ratings')['Users'].nunique()

plt.hist(ratings_per_user, bins=50)

# Plotting the histogram
ratings_per_user.plot(kind='bar', stacked=True)

# Setting labels and title
plt.xlabel('Ratings')
plt.ylabel('No. of Users')
plt.title('Histogram')

# Display the plot
plt.show()