import pandas as pd


def min_max_normalize(series):
    """
    Normalize a pandas Series to a 0-1 range.
    """

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


def recommend_hybrid(
    movie_title,
    user_id,
    movies,
    ratings,
    genre_sparse,
    user_item_sparse,
    user_ids,
    user_id_to_index,
    n=10,
    alpha=0.5
):
    """
    Generate hybrid movie recommendations.

    Combines:
    - Content-based recommendations
    - Collaborative filtering recommendations

    alpha controls the contribution of the
    content-based model.

    alpha = 1.0 -> 100% content
    alpha = 0.5 -> 50% content + 50% collaborative
    alpha = 0.0 -> 100% collaborative
    """

    # Import recommendation functions
    from src.content_model import recommend_content
    from src.collaborative_model import recommend_collaborative

    # --------------------------------------------------
    # 1. Get content-based recommendations
    # --------------------------------------------------

    content_recs = recommend_content(
        movie_title,
        movies,
        genre_sparse,
        n=30
    )

    # If the movie doesn't exist
    if content_recs.empty:
        return pd.DataFrame()

    # Keep only the columns needed for hybrid scoring
    content_scores = content_recs[
        [
            "movieId",
            "content_score"
        ]
    ].copy()

    # --------------------------------------------------
    # 2. Get collaborative recommendations
    # --------------------------------------------------

    collab_recs = recommend_collaborative(
        user_id,
        ratings,
        user_item_sparse,
        user_ids,
        user_id_to_index,
        movies,
        n=30
    )

    # If collaborative model returns nothing
    if collab_recs.empty:
        collab_scores = pd.DataFrame(
            columns=[
                "movieId",
                "score"
            ]
        )
    else:
        collab_scores = collab_recs[
            [
                "movieId",
                "score"
            ]
        ].copy()

    # --------------------------------------------------
    # 3. Combine both recommendation sources
    # --------------------------------------------------

    hybrid = pd.merge(
        content_scores,
        collab_scores,
        on="movieId",
        how="outer"
    )

    # --------------------------------------------------
    # 4. Handle missing scores
    # --------------------------------------------------

    hybrid["content_score"] = (
        hybrid["content_score"]
        .fillna(0)
    )

    hybrid["score"] = (
        hybrid["score"]
        .fillna(0)
    )

    # --------------------------------------------------
    # 5. Normalize content scores
    # --------------------------------------------------

    hybrid["content_normalized"] = (
        min_max_normalize(
            hybrid["content_score"]
        )
    )

    # --------------------------------------------------
    # 6. Normalize collaborative scores
    # --------------------------------------------------

    hybrid["collab_normalized"] = (
        min_max_normalize(
            hybrid["score"]
        )
    )

    # --------------------------------------------------
    # 7. Calculate hybrid score
    # --------------------------------------------------

    hybrid["hybrid_score"] = (
        alpha
        * hybrid["content_normalized"]
        +
        (1 - alpha)
        * hybrid["collab_normalized"]
    )

    # --------------------------------------------------
    # 8. Add movie information
    # --------------------------------------------------

    hybrid = hybrid.merge(
        movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ],
        on="movieId",
        how="inner"
    )

    # --------------------------------------------------
    # 9. Sort by hybrid score
    # --------------------------------------------------

    hybrid = hybrid.sort_values(
        "hybrid_score",
        ascending=False
    )

    # --------------------------------------------------
    # 10. Return final recommendations
    # --------------------------------------------------

    return hybrid[
        [
            "movieId",
            "title",
            "genres",
            "content_score",
            "score",
            "hybrid_score"
        ]
    ].head(n)