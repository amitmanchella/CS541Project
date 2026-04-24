"""
Build non-stationary dataset configurations where selectivity shifts partway
through the data.  These test whether an adaptive routing system (Eddy) can
detect and react to changes in data characteristics, unlike fixed-ordering
approaches.

Three configs are produced (1000 rows each):
  - nonstat_shift50.csv   : abrupt shift at row 500
  - nonstat_shift75.csv   : late shift at row 750
  - nonstat_gradual.csv   : smooth 10-segment linear transition
"""

import pandas as pd
import numpy as np
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(DATA_DIR, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

# Make sure the data directory is on the import path so we can import
# build_dataset even when running from elsewhere.
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

TARGET_LANG = "English"
TARGET_GENRE = "Comedy"
N = 1000  # total rows per config


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_pools():
    """Load raw IMDB tables and split into the four selectivity pools."""
    from build_dataset import load_raw_tables
    movies, genres, languages, plots = load_raw_tables()

    df = movies.merge(languages, on="movie_id", how="inner")
    df = df.merge(genres, on="movie_id", how="inner")
    df = df.merge(plots, on="movie_id", how="inner")
    df = df.dropna(subset=["title", "plot", "language", "genre"])
    df = df[df["plot"].str.len() >= 50]
    df = df[df["title"].str.len() > 0]

    pools = {
        "ec":   df[(df["language"] == TARGET_LANG) & (df["genre"] == TARGET_GENRE)],
        "enc":  df[(df["language"] == TARGET_LANG) & (df["genre"] != TARGET_GENRE)],
        "nec":  df[(df["language"] != TARGET_LANG) & (df["genre"] == TARGET_GENRE)],
        "nenc": df[(df["language"] != TARGET_LANG) & (df["genre"] != TARGET_GENRE)],
    }

    print(f"Pool sizes: ec={len(pools['ec'])}, enc={len(pools['enc'])}, "
          f"nec={len(pools['nec'])}, nenc={len(pools['nenc'])}")
    return pools


def _sample_segment(pools, n_rows, lang_sel, genre_sel, rng):
    """
    Sample *n_rows* rows whose marginal selectivities approximate
    the requested *lang_sel* (fraction English) and *genre_sel*
    (fraction Comedy).

    Uses the same deterministic decomposition as build_selectivity_configs.py.
    """
    n_eng = int(round(n_rows * lang_sel))
    n_noteng = n_rows - n_eng
    n_comedy = int(round(n_rows * genre_sel))
    n_notcomedy = n_rows - n_comedy

    # Decompose into four group counts that respect both marginals.
    n_ec = min(n_eng, n_comedy)
    n_enc = n_eng - n_ec
    n_nec = n_comedy - n_ec
    n_nenc = n_rows - n_ec - n_enc - n_nec

    assert all(x >= 0 for x in [n_ec, n_enc, n_nec, n_nenc]), (
        f"Infeasible segment: lang_sel={lang_sel}, genre_sel={genre_sel}"
    )

    parts = []
    for key, count in [("ec", n_ec), ("enc", n_enc),
                        ("nec", n_nec), ("nenc", n_nenc)]:
        if count > 0:
            pool = pools[key]
            replace = count > len(pool)
            parts.append(pool.sample(n=count, random_state=rng, replace=replace))

    seg = pd.concat(parts, ignore_index=True)
    # Shuffle *within* the segment so the four groups aren't in blocks,
    # but the segment itself keeps its position in the final ordering.
    seg = seg.sample(frac=1, random_state=rng).reset_index(drop=True)
    return seg[["movie_id", "title", "plot", "language", "genre"]]


# ── config builders ──────────────────────────────────────────────────────────

def _build_shift(pools, split_point, seed):
    """
    Abrupt shift: first *split_point* rows use (lang=0.2, genre=0.8),
    remaining rows use (lang=0.8, genre=0.2).
    """
    rng1 = np.random.RandomState(seed)
    rng2 = np.random.RandomState(seed + 1)

    seg1 = _sample_segment(pools, split_point, lang_sel=0.2, genre_sel=0.8, rng=rng1)
    seg2 = _sample_segment(pools, N - split_point, lang_sel=0.8, genre_sel=0.2, rng=rng2)

    return pd.concat([seg1, seg2], ignore_index=True)


def _build_gradual(pools, n_segments=10, seed=200):
    """
    Smooth transition: lang_sel linearly increases from 0.2 → 0.8 and
    genre_sel linearly decreases from 0.8 → 0.2 across *n_segments* segments.
    """
    seg_size = N // n_segments
    lang_vals = np.linspace(0.2, 0.8, n_segments)
    genre_vals = np.linspace(0.8, 0.2, n_segments)

    parts = []
    for i in range(n_segments):
        rng = np.random.RandomState(seed + i)
        # Last segment absorbs any rounding remainder.
        rows = seg_size if i < n_segments - 1 else N - seg_size * (n_segments - 1)
        seg = _sample_segment(pools, rows, lang_sel=lang_vals[i],
                              genre_sel=genre_vals[i], rng=rng)
        parts.append(seg)

    return pd.concat(parts, ignore_index=True)


# ── main entry point ─────────────────────────────────────────────────────────

def build_nonstationary_configs():
    pools = _build_pools()

    configs = {
        "nonstat_shift50":  lambda: _build_shift(pools, split_point=500, seed=100),
        "nonstat_shift75":  lambda: _build_shift(pools, split_point=750, seed=150),
        "nonstat_gradual":  lambda: _build_gradual(pools, n_segments=10, seed=200),
    }

    for name, builder in configs.items():
        df = builder()
        assert len(df) == N, f"{name}: expected {N} rows, got {len(df)}"

        fpath = os.path.join(CONFIG_DIR, f"{name}.csv")
        df.to_csv(fpath, index=False)

        actual_lang = (df["language"] == TARGET_LANG).mean()
        actual_genre = (df["genre"] == TARGET_GENRE).mean()
        print(f"  {name}.csv: {len(df)} rows, "
              f"overall lang_sel={actual_lang:.2f}, genre_sel={actual_genre:.2f}")

    print("\nDone – non-stationary configs saved to", CONFIG_DIR)


if __name__ == "__main__":
    build_nonstationary_configs()
