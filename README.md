# 🎬 CineMatch — Movie Recommendation System

A machine-learning powered movie recommendation system built using the **MovieLens 32M dataset**.

CineMatch combines **content-based filtering**, **collaborative filtering**, and a **hybrid recommendation model** to generate personalized movie recommendations.

## 🚀 Live Demo

**Streamlit App:**
`YOUR_STREAMLIT_APP_URL`

> Replace the URL above after deployment.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)

---

## 📌 Project Overview

Movie recommendation systems are designed to help users discover movies that match their interests.

This project explores three recommendation approaches:

### 1. Content-Based Filtering

Recommends movies based on the characteristics of a movie the user already likes.

For this project, movie genres are converted into numerical feature vectors using **one-hot encoding**.

Similarity between movies is calculated using **cosine similarity**.

### 2. Collaborative Filtering

Recommends movies based on the behavior of similar users.

The system builds a sparse user-movie rating matrix and identifies users with similar rating patterns.

Movies highly rated by similar users are then used to generate recommendations.

### 3. Hybrid Recommendation

The final recommendation system combines both approaches:

```text
Hybrid Score =
α × Content Score
+
(1 - α) × Collaborative Score
```

The two recommendation signals are normalized before being combined.

---

## 🧠 Machine Learning Pipeline

```text
MovieLens 32M
      │
      ├── Movies
      │     │
      │     └── Genre Features
      │             │
      │             ▼
      │      Content-Based Model
      │
      └── Ratings
            │
            └── User-Movie Matrix
                    │
                    ▼
            Collaborative Model
                    │
                    ▼
             Hybrid Model
                    │
                    ▼
             Recommendations
                    │
                    ▼
             Streamlit Application
```

---

## 🗂️ Project Structure

```text
movie-recommendation-system/
│
├── data/
│   └── ml-32m/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── content_model.py
│   ├── collaborative_model.py
│   ├── hybrid_model.py
│   ├── model_pipeline.py
│   ├── model_loader.py
│   └── recommendation_service.py
│
├── app/
│   ├── app.py
│   ├── components.py
│   └── styles.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🎥 Movies You Can Search

The application uses MovieLens movie titles, so enter the movie title as it appears in the dataset.

### Popular examples

| Movie                                                     |
| --------------------------------------------------------- |
| Toy Story (1995)                                          |
| Dark Knight, The (2008)                                   |
| Inception (2010)                                          |
| Forrest Gump (1994)                                       |
| Matrix, The (1999)                                        |
| Pulp Fiction (1994)                                       |
| Interstellar (2014)                                       |
| Titanic (1997)                                            |
| Jurassic Park (1993)                                      |
| Shawshank Redemption, The (1994)                          |
| Godfather, The (1972)                                     |
| Godfather: Part II, The (1974)                            |
| Lord of the Rings: The Fellowship of the Ring, The (2001) |
| Lord of the Rings: The Two Towers, The (2002)             |
| Lord of the Rings: The Return of the King, The (2003)     |
| Star Wars: Episode IV - A New Hope (1977)                 |
| Star Wars: Episode V - The Empire Strikes Back (1980)     |
| Star Wars: Episode VI - Return of the Jedi (1983)         |
| Avengers: Endgame (2019)                                  |
| Avengers, The (2012)                                      |
| Iron Man (2008)                                           |
| Guardians of the Galaxy (2014)                            |
| Spider-Man (2002)                                         |
| Batman Begins (2005)                                      |
| Fight Club (1999)                                         |
| Goodfellas (1990)                                         |
| Se7en (1995)                                              |
| Saving Private Ryan (1998)                                |
| Back to the Future (1985)                                 |
| Terminator 2: Judgment Day (1991)                         |

> **Note:** MovieLens stores titles using its own title format. For example, `Dark Knight, The (2008)` should be entered exactly as shown rather than `The Dark Knight (2008)`.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Scikit-learn
* Streamlit
* Jupyter Notebook
* Git & GitHub

---

## 📊 Dataset

This project uses the **MovieLens 32M dataset** from GroupLens.

The dataset contains approximately:

* 32 million ratings
* 87,000+ movies
* User ratings
* Movie genres
* Movie metadata

Dataset:

https://grouplens.org/datasets/movielens/

The dataset is not included in this repository because of its size.

---

## ⚙️ Running Locally

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd movie-recommendation-system
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate it on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Download MovieLens 32M

Download the dataset from GroupLens and place the extracted files inside:

```text
data/ml-32m/
```

The application expects:

```text
data/ml-32m/movies.csv
data/ml-32m/ratings.csv
```

### Run the application

```bash
python -m streamlit run app/app.py
```

---

## 🔬 Recommendation Techniques

### Content-Based Filtering

Movie genres are represented as binary feature vectors.

Cosine similarity measures the similarity between movie vectors:

```text
cos(A,B) = (A · B) / (||A|| ||B||)
```

A higher cosine similarity indicates greater similarity between the genre profiles of two movies.

### Collaborative Filtering

A sparse user-movie matrix is created from the rating data.

Users are compared based on their rating patterns, and highly rated movies from similar users are used to generate recommendations.

### Hybrid Model

The hybrid model combines the two recommendation signals after normalization.

This allows the system to consider both:

* similarity between movie characteristics
* similarity between user preferences

---

## 📈 Evaluation

The recommendation system uses a temporal train/test split for evaluation.

The evaluation focuses on:

* Precision@K
* Recall@K

The temporal split helps avoid using future interactions when evaluating recommendations.

---

## 💡 Key Engineering Decisions

### Sparse matrices

The MovieLens dataset contains millions of ratings, resulting in a highly sparse user-movie matrix.

Sparse matrix representations allow the system to avoid storing millions of unnecessary zero values.

### Avoiding full similarity matrices

The system calculates similarity for the relevant target user or movie rather than constructing unnecessarily large all-to-all similarity matrices.

### Modular architecture

The recommendation logic is separated from the Streamlit interface.

```text
Streamlit UI
     ↓
Recommendation Service
     ↓
Recommendation Models
     ↓
Data / Sparse Matrices
```

This makes the recommendation engine easier to test, maintain, and reuse.

---

## 🔮 Future Improvements

Possible future improvements include:

* Movie poster integration
* More advanced collaborative filtering
* Matrix factorization
* Neural recommendation models
* Better cold-start handling
* Personalized user profiles
* More detailed recommendation explanations
* Improved ranking and re-ranking
* Hyperparameter tuning
* Additional evaluation metrics

---

## 👨‍💻 Author

**Your Name**

Data Science / Machine Learning Portfolio Project

---

## ⭐ If you found this project interesting

Feel free to explore the code, experiment with the recommendation models, and try different movies in the application.
