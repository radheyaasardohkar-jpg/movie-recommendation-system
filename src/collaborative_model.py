import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings):

    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_id_to_index = {
        user_id: i
        for i, user_id in enumerate(user_ids)
    }

    movie_id_to_index = {
        movie_id: i
        for i, movie_id in enumerate(movie_ids)
    }

    user_indices = ratings["userId"].map(
        user_id_to_index
    ).values

    movie_indices = ratings["movieId"].map(
        movie_id_to_index
    ).values

    rating_values = ratings["rating"].values

    user_item_sparse = csr_matrix(
        (
            rating_values,
            (user_indices, movie_indices)
        ),
        shape=(
            len(user_ids),
            len(movie_ids)
        )
    )

    return (
        user_item_sparse,
        user_ids,
        movie_ids,
        user_id_to_index,
        movie_id_to_index
    )


def recommend_collaborative(
    user_id,
    ratings,
    user_item_sparse,
    user_ids,
    user_id_to_index,
    movies,
    n=10
):

    user_index = user_id_to_index[user_id]

    user_vector = user_item_sparse[user_index]

    similarities = cosine_similarity(
        user_vector,
        user_item_sparse
    ).flatten()

    similar_indices = similarities.argsort()[::-1]

    similar_indices = similar_indices[
        similar_indices != user_index
    ]

    top_indices = similar_indices[:10]

    similar_users = pd.DataFrame({
        "userId": user_ids[top_indices],
        "similarity": similarities[top_indices]
    })

    similar_ratings = ratings[
        ratings["userId"].isin(
            similar_users["userId"]
        )
    ].merge(
        similar_users,
        on="userId",
        how="inner"
    )

    similar_ratings = similar_ratings[
        similar_ratings["rating"] >= 4
    ].copy()

    similar_ratings["weighted_score"] = (
        similar_ratings["rating"]
        * similar_ratings["similarity"]
    )

    scores = similar_ratings.groupby(
        "movieId"
    ).agg(
        weighted_score=("weighted_score", "sum"),
        similarity_sum=("similarity", "sum"),
        rating_count=("rating", "count")
    )

    scores["score"] = (
        scores["weighted_score"]
        / scores["similarity_sum"]
    )

    watched = ratings[
        ratings["userId"] == user_id
    ]["movieId"]

    scores = scores[
        ~scores.index.isin(watched)
    ]

    recommendations = (
        scores
        .sort_values(
            "score",
            ascending=False
        )
        .head(n)
        .reset_index()
        .merge(
            movies[
                ["movieId", "title", "genres"]
            ],
            on="movieId",
            how="left"
        )
    )

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "score",
            "rating_count"
        ]
    ]