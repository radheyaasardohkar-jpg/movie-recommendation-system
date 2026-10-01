from src.data_loader import load_data
from src.content_model import build_content_model
from src.collaborative_model import build_user_item_matrix


def build_models():

    movies, ratings = load_data()

    genre_sparse = build_content_model(
        movies
    )

    (
        user_item_sparse,
        user_ids,
        movie_ids,
        user_id_to_index,
        movie_id_to_index
    ) = build_user_item_matrix(
        ratings
    )

    return {
        "movies": movies,
        "ratings": ratings,
        "genre_sparse": genre_sparse,
        "user_item_sparse": user_item_sparse,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index
    }