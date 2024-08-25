
import pandas as pd
import numpy as np
from dataset import load_dataframe
from scipy.sparse import lil_matrix


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

#map each user and each item to array index
def mapData(S):
    #Get unique items
    unique_itemSet=set(S['Items'].unique())
    sortedItems=sorted(unique_itemSet)
    
    #Get unique users
    unique_userSet=set(S['Users'].unique())
    sortedUsers=sorted(unique_userSet)
    
    #map each item to array index
    itemToIndex=dict()
    for i in range(len(sortedItems)):
        itemToIndex[sortedItems[i]]=i
    
    #map each user to array index
    userToIndex=dict()
    for i in range(len(sortedUsers)):
        userToIndex[sortedUsers[i]]=i
    return [unique_itemSet,itemToIndex,unique_userSet,userToIndex]



[unique_itemSet,itemToIndex,unique_userSet,userToIndex]=mapData(X)


R = lil_matrix((len(unique_userSet), len(itemToIndex)), dtype=np.int8)
for i in range(len(X)):
    if i%100000==0:
      print('{0:.0%}'.format(i/len(X)))
    R[userToIndex[X.iloc[i]["Users"]], itemToIndex[X.iloc[i]["Items"]]] = int(X.iloc[i]["Ratings"])
    
