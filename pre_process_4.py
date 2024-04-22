
import pandas as pd
import numpy as np
from dataset import load_dataframe
from scipy.sparse import csr_matrix 

# Προ-επεξεργασία Δεδομένων
# 4.

# Load the DataFrame
X = load_dataframe()


# Splitting the single column into separate columns
X = X[0].str.split(',', expand=True)
X = X.iloc[:, 0:3]
X.columns = ['Users', 'Items', 'Ratings']

# Dropping possible duplicates
X = X.drop_duplicates()


# Convert to numeric values
X['Ratings'] = pd.to_numeric(X['Ratings'])
X['Users'] = X['Users'].str.extract('(\d+)').astype(int)
X['Items'] = X['Items'].str.extract('(\d+)').astype(int)




#Get unique items
itemSet=set(X['Items'])
sortedItems=sorted(itemSet)

#Get unique users
userSet=set(X['Users'])
sortedUsers=sorted(userSet)

#map each item to array index
itemToIndex=dict()
for i in range(len(sortedItems)):
    itemToIndex[sortedItems[i]]=i

#map each user to array index
userToIndex=dict()
for i in range(len(sortedUsers)):
    userToIndex[sortedUsers[i]]=i

#Create an empty sparse array with int8 elements
sparseMatrix = csr_matrix((len(userSet), len(itemToIndex)),
                          dtype = np.int8).toarray()


#Fill in the sparse array
for i in range(len(X)):
    if i%100000==0:
      print('{0:.0%}'.format(i/len(X)))
    sparseMatrix[userToIndex[X.iloc[i]["Users"]]][itemToIndex[X.iloc[i]["Items"]]]=X.iloc[i]["Ratings"]
    
    

