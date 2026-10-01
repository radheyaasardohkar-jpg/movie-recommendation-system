import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_content_model(movies):

    genre_matrix = movies[
        "genres"
    ].str.get_dummies(sep="|")

    genre_sparse = csr_matrix(
        genre_matrix
    )

    return genre_sparse


def recommend_content(
    movie_title,
    movies,
    genre_sparse,
    n=10
):

    matches = movies[
        movies["title"] == movie_title
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    movie_vector = genre_sparse[
        movie_index
    ]

    similarities = cosine_similarity(
        movie_vector,
        genre_sparse
    ).flatten()

    similarities[movie_index] = -1

    top_indices = similarities.argsort()[
        -n:
    ][::-1]

    recommendations = movies.iloc[
        top_indices
    ][
        ["movieId", "title", "genres"]
    ].copy()

    recommendations[
        "content_score"
    ] = similarities[top_indices]

    return recommendations