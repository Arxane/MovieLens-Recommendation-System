#importing libraries
import pandas as pd
import numpy as np
import zipfile

#Extract the zip file
with zipfile.ZipFile('C:/Users/mayan/Pandas-Demo/ml-100k.zip', 'r') as zip_ref:
    zip_ref.extractall('C:/Users/mayan/Pandas-Demo')

#reading data_set
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#reading ratings for all movies
r_cols = ['users_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
ratings[['users_id', 'movie_id']] = ratings[['users_id', 'movie_id']].astype(int)
#reading items file
i_cols=['movie_id', 'movie_title', 'release date', 'video release date', 'IMDb url', 'unknown', 
        'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep = '|', names=i_cols,encoding='latin-1')

#training and testing the dataset

ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

#number of unique users
n_users = ratings.users_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

#user-time matrix to find similarity between users
data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

#this gives us item-item and user-user similarity in array form

#next step is to make predictions based on that
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings-mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

from sklearn.metrics import mean_absolute_error

predicted_user_ratings = []
predicted_item_ratings = []

for index, row in ratings_test.iterrows():
    user_id = row['users_id'] - 1
    movie_id = row['movie_id'] - 1
    predicted_user_ratings.append(user_prediction[user_id, movie_id])
    predicted_item_ratings.append(item_prediction[user_id, movie_id])

user_mae = mean_absolute_error(ratings_test['rating'], predicted_user_ratings)
item_mae = mean_absolute_error(ratings_test['rating'], predicted_item_ratings)

print("User-based MAE: ", user_mae)
print("Item-based MAE: ", item_mae)
