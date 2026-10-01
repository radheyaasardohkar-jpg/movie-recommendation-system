import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data():

    deploy_data = PROJECT_ROOT / "data" / "deploy"

    movies_path = deploy_data / "movies.csv"
    ratings_path = deploy_data / "ratings.csv"

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    return movies, ratings