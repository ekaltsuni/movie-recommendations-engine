import numpy as np
import pandas as pd
from dataset import load_dataframe

# Προ-επεξεργασία Δεδομένων
# 1. Να βρείτε το σύνολο των μοναδικών χρηστών U και το σύνολο των μοναδικών αντικειμένων I.

X = load_dataframe()

# Split the users' and items' columns
X['users'] = X[0].str.split(',').str[0]
X['items'] = X[0].str.split(',').str[1]

# Getting the unique count of users and items
unique_users = X['users'].nunique()
unique_items = X['items'].nunique()

print("Unique count of users:", unique_users)  # result is: 1499238
print("Unique count of items:", unique_items)  # result is: 351109