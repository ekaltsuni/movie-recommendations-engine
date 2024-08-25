
import pandas as pd
import numpy as np
from dataset import load_dataframe



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




 
#Return the constrained dataset of those users having more than minR and less than maxR number of reviews
def R_dataframe(minR,maxR,X):
    user_counts = X.iloc[:, 0].value_counts()
    filtered_users = user_counts[(user_counts >= minR) & (user_counts < maxR)]
    constrainedX = X[X.iloc[:, 0].isin(filtered_users.index)]
    return constrainedX


#Get those records of the users within Rmin and Rmax
constrainedX=R_dataframe(69,77,X)


    
#### function to get a basic statistic analysis of the dataset ####
# def stats():
#     X = load_dataframe()
#     X = X[0].str.split(',', expand=True)
#     X = X.iloc[:, 0:3]
#     X.columns = ['Users', 'Items', 'Ratings']
    
#     # Dropping possible duplicates
#     X = X.drop_duplicates()
    
    
#     # Convert to numeric values
#     X['Ratings'] = pd.to_numeric(X['Ratings'])
#     X['Users'] = X['Users'].str.extract('(\d+)').astype(int)
#     X['Items'] = X['Items'].str.extract('(\d+)').astype(int)
#     user_counts = X.iloc[:, 0].value_counts()
#     #Total number of Unique Users
#     maxR=len(X.iloc[:,0].unique())
#     print("Total number of unique users that gave 1 or more reviews: "+str(maxR)+"\n")
    
#     print("Maximum number of reviews a user (one) has ever given: "+str(max(user_counts.iloc[:]))+"\n")
    
    
    
#     #y[0]:number of reviews
#     #y[1]:percent of number of users per reviews
#     #y[2]
#     y=[[],[],[],[]]
#     maxUsers=200
#     for i in range(1,maxUsers,1):
#         y[0].append(i)
#         filtered_users = user_counts[(user_counts > i-1) & (user_counts <= i)]
#         S=X[X.iloc[:, 0].isin(filtered_users.index)]
#         y[1].append(len(set(S['Users'].unique())))
#         y[2].append(y[1][i-1]*i)
#         if i==1:
#             y[3].append(y[1][i-1]*i)
#         else:
#             y[3].append(y[3][i-2]+y[1][i-1]*i)
            
#     y[3] = [x/len(X) for x in y[3][:]]
    
#     y[1] = [x/maxR for x in y[1][:]]
    
#     print("Number of users with just one review: "+str(round(y[1][0]*maxR,0))+", which is "+str(round(y[1][0]*100,1))+"%  of the total number of users; they have left "+str(round(y[2][0]/len(X),2)*100)+"% of the total reviews\n")
#     print("Number of users with two reviews: "+str(round(y[1][1]*maxR,0))+", which is "+str(round(y[1][1]*100,1))+"% of the total number of users; they have left "+str(round(y[2][1]/len(X),2)*100)+"% of the total reviews\n")
    
#     print("Number of users having between 3 and "+ str(maxUsers)+" reviews is: "+str(round(sum(y[1][3:(maxUsers-1)])*maxR,0))+", which is "+str(round(sum(y[1][3:(maxUsers-1)]),2)*100)+"% of the total number of users; they have left "+str(sum(y[2][3:(maxUsers-1)]))+" reviews or "+str(round(sum(y[2][3:(maxUsers-1)])/len(X),2)*100)+"% of the total reviews\n")

#     plt.plot(y[0], y[1])
    
#     plt.ylabel('Number of Users')
#     plt.xlabel('Number of Movie Ratings')
#     plt.title('Cummulative number of users per number of ratings')

#     # Display the plot
#     plt.show()
#     return y

# stats()