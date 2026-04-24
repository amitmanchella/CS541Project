"""
Phase 1 - Step 1.3: Build 25 selectivity configurations.
Each config has 1000 movies with controlled language/genre selectivity.
Selectivity levels: 0%, 25%, 50%, 75%, 100%.
"""

import pandas as pd
import numpy as np
import os
from .build_dataset import load_raw_tables


DATA_DIR = os.path.dirname(__file__)
FULL_PATH = os.path.join(DATA_DIR, "movies_full.csv")
CONFIG_DIR = os.path.join(DATA_DIR, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

SELECTIVITY_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
TARGET_LANG = "English"
TARGET_GENRE = "Comedy"
N = 1000


def load_full():
    df = pd.read_csv(FULL_PATH)
    return df


def build_all_configs():
    # We need a large pool to sample from. Reload from raw with more movies.
    from .build_dataset import load_raw_tables
    movies, genres, languages, plots = load_raw_tables()

    df = movies.merge(languages, on="movie_id", how="inner")
    df = df.merge(genres, on="movie_id", how="inner")
    df = df.merge(plots, on="movie_id", how="inner")
    df = df.dropna(subset=["title", "plot", "language", "genre"])
    df = df[df["plot"].str.len() >= 50]
    df = df[df["title"].str.len() > 0]

    # Split into groups
    eng_comedy = df[(df["language"] == TARGET_LANG) & (df["genre"] == TARGET_GENRE)]
    eng_notcomedy = df[(df["language"] == TARGET_LANG) & (df["genre"] != TARGET_GENRE)]
    noteng_comedy = df[(df["language"] != TARGET_LANG) & (df["genre"] == TARGET_GENRE)]
    noteng_notcomedy = df[(df["language"] != TARGET_LANG) & (df["genre"] != TARGET_GENRE)]

    print(f"Pool sizes: eng_comedy={len(eng_comedy)}, eng_notcomedy={len(eng_notcomedy)}, "
          f"noteng_comedy={len(noteng_comedy)}, noteng_notcomedy={len(noteng_notcomedy)}")

    for lang_sel in SELECTIVITY_LEVELS:
        for genre_sel in SELECTIVITY_LEVELS:
            # How many of 1000 should pass each filter
            n_eng = int(N * lang_sel)       # pass language filter
            n_noteng = N - n_eng
            n_comedy = int(N * genre_sel)   # pass genre filter
            n_notcomedy = N - n_comedy

            # Need: n_eng_comedy, n_eng_notcomedy, n_noteng_comedy, n_noteng_notcomedy
            # that sum to N and satisfy marginals
            n_ec = min(n_eng, n_comedy)
            n_enc = n_eng - n_ec
            n_nec = n_comedy - n_ec
            n_nenc = N - n_ec - n_enc - n_nec

            # Ensure non-negative
            if any(x < 0 for x in [n_ec, n_enc, n_nec, n_nenc]):
                print(f"  Skipping lang{int(lang_sel*100)}_genre{int(genre_sel*100)}: infeasible")
                continue

            # Check pool availability
            avail = {
                "ec": len(eng_comedy), "enc": len(eng_notcomedy),
                "nec": len(noteng_comedy), "nenc": len(noteng_notcomedy),
            }
            needed = {"ec": n_ec, "enc": n_enc, "nec": n_nec, "nenc": n_nenc}

            feasible = True
            for k in needed:
                if needed[k] > avail[k]:
                    feasible = False
                    break

            if not feasible:
                # Adjust: allow sampling with replacement for scarce categories
                pass

            rng = np.random.RandomState(42 + int(lang_sel * 100) + int(genre_sel * 10))
            parts = []
            for pool, count, name in [
                (eng_comedy, n_ec, "ec"),
                (eng_notcomedy, n_enc, "enc"),
                (noteng_comedy, n_nec, "nec"),
                (noteng_notcomedy, n_nenc, "nenc"),
            ]:
                if count > 0:
                    replace = count > len(pool)
                    sampled = pool.sample(n=count, random_state=rng, replace=replace)
                    parts.append(sampled)

            config_df = pd.concat(parts, ignore_index=True)
            config_df = config_df.sample(frac=1, random_state=rng).reset_index(drop=True)
            config_df = config_df[["movie_id", "title", "plot", "language", "genre"]]

            fname = f"lang{int(lang_sel*100)}_genre{int(genre_sel*100)}.csv"
            config_df.to_csv(os.path.join(CONFIG_DIR, fname), index=False)
            actual_lang = (config_df["language"] == TARGET_LANG).mean()
            actual_genre = (config_df["genre"] == TARGET_GENRE).mean()
            print(f"  {fname}: {len(config_df)} rows, "
                  f"lang_sel={actual_lang:.2f}, genre_sel={actual_genre:.2f}")


if __name__ == "__main__":
    build_all_configs()
