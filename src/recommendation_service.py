from src.content_model import recommend_content
from src.collaborative_model import recommend_collaborative
from src.hybrid_model import recommend_hybrid


def get_content_recommendations(
    movie_title,
    models,
    n=10
):

    return recommend_content(
        movie_title,
        models["movies"],
        models["genre_sparse"],
        n=n
    )


def get_collaborative_recommendations(
    user_id,
    models,
    n=10
):

    return recommend_collaborative(
        user_id,
        models["ratings"],
        models["user_item_sparse"],
        models["user_ids"],
        models["user_id_to_index"],
        models["movies"],
        n=n
    )


def get_hybrid_recommendations(
    movie_title,
    user_id,
    models,
    n=10,
    alpha=0.5
):

    return recommend_hybrid(
        movie_title,
        user_id,
        models["movies"],
        models["ratings"],
        models["genre_sparse"],
        models["user_item_sparse"],
        models["user_ids"],
        models["user_id_to_index"],
        n=n,
        alpha=alpha
    )