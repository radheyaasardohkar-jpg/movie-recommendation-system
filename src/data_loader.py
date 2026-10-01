import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data():

    movies_path = PROJECT_ROOT / "data" / "ml-32m" / "movies.csv"
    ratings_path = PROJECT_ROOT / "data" / "ml-32m" / "ratings.csv"

    movies = pd.read_csv(movies_path)

    ratings = pd.read_csv(ratings_path)

    return movies, ratings