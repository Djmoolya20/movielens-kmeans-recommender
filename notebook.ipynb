import os
import gdown
import zipfile
import pandas as pd

# Configuration
drive_url = 'https://drive.google.com/uc?id=1NYhqi5_9b2_Y28mB5gc9nghmChA6uGky'
zip_path = 'ml-1m.zip'
extract_path = '/content/ml-1m_extracted'
data_path = os.path.join(extract_path, 'ml-1m')

# 1. Download and Extract
if not os.path.exists(data_path):
    if not os.path.exists(zip_path):
        print('Downloading dataset...')
        gdown.download(drive_url, zip_path, quiet=False)

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print('Extraction complete.')

# 2. Load Data
ratings = pd.read_csv(os.path.join(data_path, 'ratings.dat'), sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin1')
movies = pd.read_csv(os.path.join(data_path, 'movies.dat'), sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin1')

# 3. Merge DataFrames
movielens_combined = pd.merge(ratings, movies, on='movieId')
print("Data loaded and merged successfully!")
display(movielens_combined.head())
import pandas as pd
import os

# Point to the local extraction path from the gdown download
data_path = '/content/ml-1m_extracted/ml-1m'

try:
    # Load the data using '::' separator for ml-1m
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.dat'), sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin1')
    movies = pd.read_csv(os.path.join(data_path, 'movies.dat'), sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin1')
    print("Data loaded successfully from local session storage!")
    display(ratings.head())
except Exception as e:
    print(f"Loading error: {e}\nMake sure cell bee473f0 finished successfully.")
import numpy as np

# Filtering for manageable size
ratings_small = ratings.sample(100000, random_state=42)
active_users = ratings_small['userId'].value_counts()[lambda x: x > 5].index
popular_movies = ratings_small['movieId'].value_counts()[lambda x: x > 5].index

ratings_filtered = ratings_small[
    (ratings_small['userId'].isin(active_users)) &
    (ratings_small['movieId'].isin(popular_movies))
]

# Create Pivot Table
user_movie_rating = ratings_filtered.pivot_table(index='userId', columns='movieId', values='rating')

# Mean-centering and filling NaNs
user_movie_rating = user_movie_rating.sub(user_movie_rating.mean(axis=1), axis=0).fillna(0).astype('float32')

scaled_data = user_movie_rating.values
print(f"Matrix Shape: {user_movie_rating.shape}")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

inertia = []
silhouette_scores = []
davies_bouldin_indices = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    davies_bouldin_indices.append(davies_bouldin_score(scaled_data, kmeans.labels_))

# Plotting Metrics
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method (Inertia)")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='green')
plt.title("Silhouette Score (Higher is better)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(k_values, davies_bouldin_indices, marker='o', color='red')
plt.title("Davies-Bouldin Index (Lower is better)")
plt.grid(True)

plt.tight_layout()
plt.show()
from sklearn.cluster import KMeans

# Based on the plots, k=3 showed a strong silhouette score
optimal_k = 3
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=5,
    max_iter=100
)

kmeans.fit(scaled_data)
user_movie_rating['Cluster'] = kmeans.labels_

print(f"Users assigned to {optimal_k} clusters.")
print("Cluster distribution:")
print(user_movie_rating['Cluster'].value_counts())
import numpy as np

def recommend_movies(user_id, top_n=10):
    if user_id not in user_movie_rating.index:
        return "User not found"

    cluster = user_movie_rating.loc[user_id, 'Cluster']

    similar_users = user_movie_rating[
        user_movie_rating['Cluster'] == cluster
    ]

    avg_ratings = similar_users.drop('Cluster', axis=1).mean()

    # Remove already watched movies
    user_ratings = user_movie_rating.loc[user_id].drop('Cluster')
    unseen_movies = user_ratings[user_ratings == 0].index

    filtered_ratings = avg_ratings[unseen_movies]

    # Conditional check to toggle noise based on user_id
    if user_id % 2 == 0: # Example condition: add noise for even user_ids
        np.random.seed(user_id) # Seed based on user_id if noise is added

        random_noise = np.random.normal(0, 0.1, len(filtered_ratings))
        filtered_ratings_with_noise = filtered_ratings + random_noise
    else:
        # If no noise is added, use the original filtered_ratings
        filtered_ratings_with_noise = filtered_ratings

    top_movies = filtered_ratings_with_noise.sort_values(ascending=False).head(top_n)

    return top_movies
def get_movie_names(movie_ids):
    return movies[movies['movieId'].isin(movie_ids)][['title', 'genres']]
# Select a sample user and generate recommendations
test_user_id = user_movie_rating.index[32]
recommended_movie_ids = recommend_movies(test_user_id, top_n=5)

if isinstance(recommended_movie_ids, str):
    print(recommended_movie_ids)
else:
    recommended_movie_names = movies[movies['movieId'].isin(recommended_movie_ids.index)]
    print(f"\n🎬 Top 5 Recommendations for User {test_user_id}:\n")
    for _, row in recommended_movie_names.iterrows():
        print(f"- {row['title']} ({row['genres']})")
