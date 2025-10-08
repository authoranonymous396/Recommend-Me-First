import pandas as pd 
import numpy as np 
import pickle
import random 
import os

# Get the absolute path of the current file
script_path = os.path.dirname(os.path.abspath(__file__))
# Load Movie data and Quality 
movies = pd.read_csv(os.path.join(script_path, "ml-32m", "movies.csv"))
ratings = pd.read_csv(os.path.join(script_path, "ml-32m", "ratings.csv"))

movies['genre_list']= movies['genres'].str.split('|')
pkl_path = os.path.join(script_path, "final_quality.pkl")
with open(pkl_path, "rb") as f:
    movie_keep = pickle.load(f)
movie_len = 100
movie_keep = movie_keep[:movie_len]
ratings = ratings[ratings['movieId'].isin(movie_keep)]
# create matrix 
percentage = int(input("Enter Percentage: "))
number_iterations = ratings['timestamp'].nunique()
max_time =  int((number_iterations/100)*percentage)
t = sorted(ratings['timestamp'].unique())[:max_time][-1]
movies = 100
g = 5 
n_users = g*movies*(movies-1) #49500
ratings_count = ratings.groupby("userId").size()
users_keep = ratings_count.sort_values(ascending=False).head(n_users).index.to_numpy()

# Contrust matrix of time we would like 
R_t = ratings[ratings['timestamp']<= t].copy()
R_t = R_t.pivot(index='userId', columns='movieId', values='rating').sort_index(axis=0).sort_index(axis=1)  # Only if you actually need sorting
print(R_t.shape)
R_t_final = ratings.pivot(index='userId', columns= 'movieId',values='rating').sort_index(axis=0).sort_index(axis=1) 
threshold = R_t_final.mean(axis= 1)
dict_quality = {}
for (i, movie) in enumerate(movie_keep):
    dict_quality[movie] = i

E_t_final = pd.DataFrame(
        np.full((n_users, movies),np.nan),     
        index = users_keep,
        columns=movie_keep     # list of all movies
    )
E_t = R_t.where(R_t.isna(), R_t.gt(threshold, axis=0).astype(float))
print(f"At time t={t}, data is {(E_t.count(axis= 0).sum()/ratings.shape[0])*100} %")
common_users = list(set(users_keep).intersection(set(E_t.index)))
if E_t.shape[0]< n_users: 
    E_t_final.reset_index(inplace=True, drop = True)
    E_t.reset_index(inplace=True, drop = True)
    E_t_final.loc[E_t.index, :] = E_t
elif len(common_users) < n_users: 
    ratings_count = ratings[ratings['timestamp']<= t].groupby("userId").size()
    users_keep_past = ratings_count.sort_values(ascending=False).head(n_users).index.to_numpy()
    E_t = R_t.where(R_t.isna(), R_t.gt(threshold, axis=0).astype(float))
    E_t_final = E_t[E_t.index.isin(users_keep_past)]
elif len(common_users) == n_users: 
    E_t = E_t[E_t.index.isin(users_keep)][movie_keep]
    E_t_final = E_t
final_E_df = E_t_final.rename(columns=dict_quality)
final_E_df.reset_index(inplace=True, drop = True)
arr_none = E_t_final.astype(object)
arr_none[np.isnan(E_t_final)] = None
p = E_t_final.count(axis= 0).sum()/ratings.shape[0]*100 
file_path = os.path.join(script_path, f"Initial_matrix_{int(round(p, 2)*100)}.csv")
E_t_final.to_csv(file_path, index=True)  