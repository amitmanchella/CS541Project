"""
Phase 1 - Step 1.2: Build the controlled IMDB dataset.
Loads raw JOB/IMDB tables, extracts movies with title, plot, language, genre.
Produces data/movies_full.csv with 1000 movies.
"""

import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "movies_full.csv")


def load_raw_tables():
    title = pd.read_csv(
        os.path.join(RAW_DIR, "title.csv"),
        header=None,
        names=["id", "title", "imdb_index", "kind_id", "production_year",
               "imdb_id", "phonetic_code", "episode_of_id", "season_nr",
               "episode_nr", "series_years", "md5sum"],
        low_memory=False,
        on_bad_lines="skip",
    )
    movie_info = pd.read_csv(
        os.path.join(RAW_DIR, "movie_info.csv"),
        header=None,
        names=["id", "movie_id", "info_type_id", "info", "note"],
        low_memory=False,
        on_bad_lines="skip",
    )
    # kind_id=1 means "movie" (not TV series, episode, etc.)
    movies = title[title["kind_id"] == 1][["id", "title"]].copy()
    movies.rename(columns={"id": "movie_id"}, inplace=True)

    # info_type_id: 3=genres, 4=languages, 98=plot
    genres = movie_info[movie_info["info_type_id"] == 3][["movie_id", "info"]].copy()
    genres.rename(columns={"info": "genre"}, inplace=True)
    genres = genres.drop_duplicates(subset="movie_id", keep="first")

    languages = movie_info[movie_info["info_type_id"] == 4][["movie_id", "info"]].copy()
    languages.rename(columns={"info": "language"}, inplace=True)
    languages = languages.drop_duplicates(subset="movie_id", keep="first")

    plots = movie_info[movie_info["info_type_id"] == 98][["movie_id", "info"]].copy()
    plots.rename(columns={"info": "plot"}, inplace=True)
    plots = plots.drop_duplicates(subset="movie_id", keep="first")

    return movies, genres, languages, plots


def build_dataset():
    movies, genres, languages, plots = load_raw_tables()

    df = movies.merge(languages, on="movie_id", how="inner")
    df = df.merge(genres, on="movie_id", how="inner")
    df = df.merge(plots, on="movie_id", how="inner")

    df = df.dropna(subset=["title", "plot", "language", "genre"])
    df = df[df["plot"].str.len() >= 50]
    df = df[df["title"].str.len() > 0]

    print(f"Total movies with all fields: {len(df)}")
    print(f"Top languages:\n{df['language'].value_counts().head(10)}")
    print(f"Top genres:\n{df['genre'].value_counts().head(10)}")

    sample = df.sample(n=min(1000, len(df)), random_state=42)
    sample = sample[["movie_id", "title", "plot", "language", "genre"]].reset_index(drop=True)

    sample.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(sample)} movies to {OUT_PATH}")
    return sample


if __name__ == "__main__":
    build_dataset()
