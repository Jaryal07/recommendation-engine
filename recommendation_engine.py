"""
Personalized Movie Recommendation Engine
Dataset: MovieLens 1M
Tech: Python, scikit-learn, pandas, matplotlib, seaborn
Techniques: Collaborative Filtering + Content-Based Hybrid
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ==================== DIRECTORY SETUP ====================
def create_output_dirs():
    """Create required directories for saved models and outputs"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


# ==================== DATA LOADING ====================
def load_movie_data():
    """
    Load MovieLens 1M dataset
    Expected folder structure:
    ml-1m/
        ratings.dat
        movies.dat
        users.dat
    """
    print("=" * 70)
    print("MOVIE RECOMMENDATION ENGINE - DATA LOADING")
    print("=" * 70)

    ratings = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
        encoding="latin-1",
    )

    movies = pd.read_csv(
        "ml-1m/movies.dat",
        sep="::",
        header=None,
        names=["MovieID", "Title", "Genres"],
        engine="python",
        encoding="latin-1",
    )

    users = pd.read_csv(
        "ml-1m/users.dat",
        sep="::",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        engine="python",
        encoding="latin-1",
    )

    print(f"\n✓ Loaded {len(ratings)} ratings")
    print(f"✓ Loaded {len(movies)} movies")
    print(f"✓ Loaded {len(users)} users")
    print(f"\nRatings shape: {ratings.shape}")
    print(f"Movies shape: {movies.shape}")
    print(f"Users shape: {users.shape}")
    print(f"\nRating distribution:\n{ratings['Rating'].value_counts().sort_index()}")

    return ratings, movies, users


# ==================== COLLABORATIVE FILTERING ====================
class CollaborativeFiltering:
    """User-based and Item-based Collaborative Filtering"""

    def __init__(self, ratings):
        self.ratings = ratings
        self.user_item_matrix = ratings.pivot_table(
            index="UserID", columns="MovieID", values="Rating"
        ).fillna(0)

        print(f"\nUser-Item Matrix Shape: {self.user_item_matrix.shape}")
        sparsity = (
            (self.user_item_matrix == 0).sum().sum()
            / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])
            * 100
        )
        print(f"Sparsity: {sparsity:.2f}%")

        self.item_similarity_df = None
        self.user_similarity_df = None
        self._build_similarity_matrices()

    def _build_similarity_matrices(self):
        """Precompute similarity matrices"""
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns,
        )

        user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def item_based_cf(self, user_id, n_recommendations=5):
        """Recommend movies using item-item similarity"""
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index

        if len(rated_movies) == 0:
            return []

        recommendations = {}

        for rated_movie in rated_movies:
            similar_movies = self.item_similarity_df[rated_movie].sort_values(
                ascending=False
            )[1 : n_recommendations + 10]

            for movie_id, similarity in similar_movies.items():
                if movie_id not in rated_movies:
                    if movie_id not in recommendations:
                        recommendations[movie_id] = []
                    recommendations[movie_id].append(similarity)

        recommendation_scores = {
            movie: np.mean(scores) for movie, scores in recommendations.items()
        }

        return sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[
            :n_recommendations
        ]

    def user_based_cf(self, user_id, n_recommendations=5, k_neighbors=10):
        """Recommend movies using similar users"""
        if user_id not in self.user_item_matrix.index:
            return []

        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[
            1 : k_neighbors + 1
        ]

        user_rated = set(
            self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index
        )

        recommendations = {}

        for similar_user, similarity in similar_users.items():
            similar_user_ratings = self.user_item_matrix.loc[similar_user]
            rated_by_similar = similar_user_ratings[similar_user_ratings > 0]

            for movie_id, rating in rated_by_similar.items():
                if movie_id not in user_rated:
                    if movie_id not in recommendations:
                        recommendations[movie_id] = []
                    recommendations[movie_id].append(rating * similarity)

        recommendation_scores = {
            movie: np.mean(scores) for movie, scores in recommendations.items()
        }

        return sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[
            :n_recommendations
        ]

    def matrix_factorization_cf(self, n_factors=50):
        """SVD-based collaborative filtering"""
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        user_factors = svd.fit_transform(self.user_item_matrix)

        reconstructed_ratings = user_factors @ svd.components_
        reconstructed_df = pd.DataFrame(
            reconstructed_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
        )

        explained_variance = svd.explained_variance_ratio_.sum()
        return reconstructed_df, explained_variance


# ==================== CONTENT-BASED FILTERING ====================
class ContentBasedFiltering:
    """Content-based recommendations using movie genres"""

    def __init__(self, movies):
        self.movies = movies.copy()

        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split("|"), lowercase=False)
        self.genre_vectors = tfidf.fit_transform(self.movies["Genres"])

        self.content_similarity = cosine_similarity(self.genre_vectors)
        self.content_similarity_df = pd.DataFrame(
            self.content_similarity,
            index=self.movies["MovieID"],
            columns=self.movies["MovieID"],
        )

    def recommend_similar_movies(self, movie_id, n_recommendations=5):
        """Recommend similar movies based on genres"""
        if movie_id not in self.content_similarity_df.index:
            return []

        similar_movies = self.content_similarity_df[movie_id].sort_values(
            ascending=False
        )[1 : n_recommendations + 1]

        return list(similar_movies.index)


# ==================== HYBRID RECOMMENDER ====================
class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based filtering"""

    def __init__(self, cf, cbf, movies, alpha=0.7):
        self.cf = cf
        self.cbf = cbf
        self.movies = movies
        self.alpha = alpha

    def recommend(self, user_id, n_recommendations=5):
        """Generate hybrid recommendations"""
        if user_id not in self.cf.user_item_matrix.index:
            return []

        cf_recs = self.cf.user_based_cf(
            user_id, n_recommendations=n_recommendations * 2
        )
        cf_rec_dict = {movie_id: score for movie_id, score in cf_recs}

        user_ratings = self.cf.user_item_matrix.loc[user_id]
        top_rated = user_ratings[user_ratings > 0]

        cb_rec_dict = {}
        if len(top_rated) > 0:
            top_rated_movie = top_rated.idxmax()
            if top_rated_movie in self.cbf.content_similarity_df.index:
                cb_movie_ids = self.cbf.recommend_similar_movies(
                    top_rated_movie, n_recommendations=n_recommendations * 2
                )
                cb_rec_dict = {
                    movie_id: (len(cb_movie_ids) - i) / max(len(cb_movie_ids), 1)
                    for i, movie_id in enumerate(cb_movie_ids)
                }

        all_movies = set(cf_rec_dict.keys()) | set(cb_rec_dict.keys())
        hybrid_scores = {}

        for movie_id in all_movies:
            cf_score = cf_rec_dict.get(movie_id, 0)
            cb_score = cb_rec_dict.get(movie_id, 0)
            hybrid_scores[movie_id] = (
                self.alpha * cf_score + (1 - self.alpha) * cb_score
            )

        recommendations = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:n_recommendations]

        return recommendations


# ==================== SAVE MODELS ====================
def save_models(cf, cbf, hybrid):
    """Save trained models as pickle files"""
    with open("models/collaborative_filter.pkl", "wb") as f:
        pickle.dump(cf, f)

    with open("models/content_filter.pkl", "wb") as f:
        pickle.dump(cbf, f)

    with open("models/hybrid_model.pkl", "wb") as f:
        pickle.dump(hybrid, f)

    print("\n✓ Saved collaborative_filter.pkl")
    print("✓ Saved content_filter.pkl")
    print("✓ Saved hybrid_model.pkl")


# ==================== EXPORT RECOMMENDATIONS ====================
def export_recommendations_to_csv(
    recommendations, movies, user_id, method_name="Hybrid"
):
    """Export recommendations to CSV"""
    rows = []

    for rank, (movie_id, score) in enumerate(recommendations, start=1):
        movie_info = movies[movies["MovieID"] == movie_id]
        if not movie_info.empty:
            rows.append(
                {
                    "UserID": user_id,
                    "Method": method_name,
                    "Rank": rank,
                    "MovieID": movie_id,
                    "Title": movie_info.iloc[0]["Title"],
                    "Genres": movie_info.iloc[0]["Genres"],
                    "Score": round(float(score), 4),
                }
            )

    recommendations_df = pd.DataFrame(rows)
    recommendations_df.to_csv("outputs/user_recommendations.csv", index=False)

    print("✓ Saved outputs/user_recommendations.csv")
    return recommendations_df


# ==================== EVALUATION METRICS ====================
def evaluate_recommendations(ratings, model, n_test_users=100):
    """Evaluate model using simple train-test split"""
    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 70)

    train_ratings = ratings.sample(frac=0.8, random_state=42)
    test_ratings = ratings[~ratings.index.isin(train_ratings.index)]

    temp_model = CollaborativeFiltering(train_ratings)

    precision_scores = []
    recall_scores = []

    test_users = test_ratings["UserID"].unique()[:n_test_users]

    for user_id in test_users:
        if user_id not in temp_model.user_item_matrix.index:
            continue

        test_items = set(test_ratings[test_ratings["UserID"] == user_id]["MovieID"])

        if len(test_items) == 0:
            continue

        recs = temp_model.item_based_cf(user_id, n_recommendations=5)
        rec_items = set([movie_id for movie_id, _ in recs])

        if len(rec_items) > 0:
            intersection = len(rec_items & test_items)
            precision = intersection / len(rec_items)
            recall = intersection / len(test_items) if len(test_items) > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

    if precision_scores:
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)

        print(f"\nMean Precision@5: {mean_precision:.4f}")
        print(f"Mean Recall@5: {mean_recall:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(
            precision_scores, bins=20, color="#3498db", alpha=0.7, edgecolor="black"
        )
        axes[0].set_xlabel("Precision@5")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Precision Distribution (Mean: {mean_precision:.3f})")
        axes[0].axvline(mean_precision, color="red", linestyle="--", linewidth=2)

        axes[1].hist(
            recall_scores, bins=20, color="#e74c3c", alpha=0.7, edgecolor="black"
        )
        axes[1].set_xlabel("Recall@5")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Recall Distribution (Mean: {mean_recall:.3f})")
        axes[1].axvline(mean_recall, color="darkred", linestyle="--", linewidth=2)

        plt.tight_layout()
        plt.savefig("outputs/recommendation_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("✓ Saved outputs/recommendation_metrics.png")
    else:
        print("No evaluation scores could be calculated.")


# ==================== DISPLAY RECOMMENDATIONS ====================
def display_recommendations(recommendations, movies):
    """Print recommendations in readable format"""
    print("\n" + "-" * 70)
    print("TOP RECOMMENDATIONS")
    print("-" * 70)

    for idx, (movie_id, score) in enumerate(recommendations, 1):
        movie_info = movies[movies["MovieID"] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]["Title"]
            genres = movie_info.iloc[0]["Genres"]
            print(f"\n{idx}. {title}")
            print(f"   Genres: {genres}")
            print(f"   Score: {score:.4f}")


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    create_output_dirs()

    ratings, movies, users = load_movie_data()

    cf = CollaborativeFiltering(ratings)
    cbf = ContentBasedFiltering(movies)
    hybrid = HybridRecommender(cf, cbf, movies, alpha=0.7)

    save_models(cf, cbf, hybrid)

    print("\n" + "=" * 70)
    print("RECOMMENDATION METHODS COMPARISON")
    print("=" * 70)

    test_user = 1

    print(f"\n1. ITEM-BASED COLLABORATIVE FILTERING (User {test_user})")
    item_cf_recs = cf.item_based_cf(test_user, n_recommendations=5)
    display_recommendations(item_cf_recs, movies)

    print(f"\n2. USER-BASED COLLABORATIVE FILTERING (User {test_user})")
    user_cf_recs = cf.user_based_cf(test_user, n_recommendations=5)
    display_recommendations(user_cf_recs, movies)

    print(f"\n3. HYBRID RECOMMENDATION (User {test_user})")
    hybrid_recs = hybrid.recommend(test_user, n_recommendations=5)
    display_recommendations(hybrid_recs, movies)

    export_recommendations_to_csv(hybrid_recs, movies, test_user, method_name="Hybrid")

    evaluate_recommendations(ratings, cf, n_test_users=100)

    print("\n" + "=" * 70)
    print("✓ Recommendation Engine Complete!")
    print("=" * 70)
