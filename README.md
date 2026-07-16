# MovieLens K-Means Recommender

A cluster-based collaborative filtering movie recommendation system built on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset (1M+ ratings, 6,000+ users, 3,900+ movies).

Users are grouped into behavioral clusters with K-Means, and recommendations for a user are generated from the average ratings of everyone else in their cluster. This is a scalable alternative to pairwise user-based collaborative filtering — instead of comparing one user against every other user, you compare them against their cluster.

## How it works

1. **Data loading** — Downloads the MovieLens 1M dataset (`ratings.dat`, `movies.dat`) via Google Drive, extracts it, and merges both files on `movieId` into a single dataframe.

2. **Sampling & filtering** — Samples 100,000 ratings from the full dataset and keeps only users and movies with more than 5 ratings, to avoid an overly sparse matrix.

3. **Matrix construction** — Pivots the filtered ratings into a user × movie matrix. Each user's ratings are **mean-centered** (their personal average is subtracted from each rating) to normalize individual rating bias — e.g. a user who tends to rate everything 5★ vs. one who rates everything 3★. Missing (unrated) entries are filled with 0.

4. **Choosing k** — K-Means is run for k = 2 to 10, and three metrics are tracked for each:
   - **Inertia** (Elbow Method)
   - **Silhouette Score** (cluster separation quality — higher is better)
   - **Davies-Bouldin Index** (cluster compactness vs. separation — lower is better)

   k = 3 is selected based on the strongest silhouette score from this sweep.

5. **Clustering** — K-Means is fit with k = 3 on the mean-centered matrix, and each user is assigned a cluster label.

6. **Recommendation** — For a given user:
   - Find their cluster.
   - Average the ratings of all users in that cluster across all movies.
   - Filter out movies the user has already rated.
   - Rank remaining movies by cluster-average score and return the top-N.

   A small amount of random noise (seeded per user ID) is optionally added to scores for even-numbered user IDs, as a simple mechanism to introduce variation into otherwise identical cluster-based rankings.

7. **Output** — Recommended movie IDs are mapped back to titles and genres via the `movies` dataframe for display.

## Setup

```bash
pip install -r requirements.txt
```

The dataset is downloaded automatically from Google Drive on first run via `gdown`.

## Usage

Run the notebook cells in order (data loading → matrix construction → k selection → clustering → recommendation). To get recommendations for a user:

```python
recommended = recommend_movies(user_id=<some_user_id>, top_n=5)
recommended_movie_names = get_movie_names(recommended.index)
```

## Evaluation metrics used

| Metric | Purpose |
|---|---|
| Inertia (Elbow Method) | Measures within-cluster sum of squares; used to spot diminishing returns as k increases |
| Silhouette Score | Measures how well-separated and cohesive clusters are (higher is better) |
| Davies-Bouldin Index | Measures average similarity between clusters (lower is better) |

## Known limitation

Cluster-average recommendations can converge toward broadly popular movies within a cluster, since the averaging step includes zero-filled (unrated) entries alongside actual ratings. This is a common effect in simple mean-centered, zero-filled collaborative filtering setups and is an area for further refinement (e.g. averaging only over actually-rated entries, or clustering on a reduced latent representation of the rating matrix).

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — GroupLens Research, University of Minnesota.
