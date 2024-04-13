import numpy as np
import pandas as pd


def load_dataframe():

    # Load the dataset
    dataset = np.load('Dataset.npy')
    # Convert to dataframe
    X = pd.DataFrame(dataset)

    return X

# Print the first five entries

X = load_dataframe()
print(X.head())


# Remove the first two non-numeric characters from the first two columns
    #X.iloc[:, :2] = X.iloc[:, :2].applymap(lambda x: ''.join(c for c in str(x) if c.isdigit()))
    # Convert user, item, and rating from strings to integers
    #X.iloc[:, :3] = X.iloc[:, :3].astype('int64')

    # Convert the last column to datetime
    #X.iloc[:, 3] = pd.to_datetime(X.iloc[:, 3])
