"""
preprocessing.py — Data loading, genre merging, sampling, the two-stream text
cleaning pipeline (Stream A for LDA/VADER, Stream B for DistilBERT/BERTopic),
and per-review metadata carry-through for richer bot detection / legitimacy.

MEMORY NOTE
-----------
The reviews CSV (~8 GB / ~48M rows) is streamed in chunks and filtered per
chunk; the working set is capped at config.POOL_CAP rows so peak RAM stays flat.

METADATA NOTE
-------------
The najzeko file is a flattened export, so column spellings can vary. We read
the header first and keep only the metadata columns that actually exist (trying
a few known spellings), so a name mismatch is skipped rather than crashing.
"""

import re

import nltk
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config

# Download NLTK resources once
for resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_stop_words = set(stopwords.words("english")) - config.GAMING_TERMS
_lemmatizer = WordNetLemmatizer()

# Optional per-review metadata to carry through IF present in the CSV header.
# Several spellings are listed; whichever exists is used. These feed the
# enriched bot detector (this step) and the legitimacy score (a later step).
_DESIRED_METADATA = [
    "votes_funny",
    "votes_helpful", "votes_up",
    "weighted_vote_score",
    "received_for_free",
    "steam_purchase",
    "written_during_early_access",
    "author.playtime_forever", "playtime_forever",
    "author.playtime_at_review", "playtime_at_review",
    "timestamp_created", "unix_timestamp_created",
]


def _coerce_recommended(s: pd.Series) -> pd.Series:
    """Handle 'recommended' whether it arrives as bool (najzeko default) or text."""
    if s.dtype == bool:
        return s.astype("int8")
    return s.map(
        {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}
    ).astype("Int8")


def load_merge_and_sample() -> pd.DataFrame:
    """
    Stream the reviews file in chunks, attach genre, filter short/low-signal
    reviews, carry through available metadata, then draw an equal stratified
    sample per genre. Memory-bounded.
    """
    # --- small genre file first: build a name -> genre lookup dict -----------
    print("Loading game metadata for genre lookup...")
    games = pd.read_csv(config.GAMES_PATH, usecols=["name", "genres"]).dropna()
    games["genre"] = games["genres"].str.split(";").str[0].str.strip()
    genre_map = dict(zip(games["name"], games["genre"]))
    del games

    # --- detect which metadata columns actually exist in the CSV header ------
    header = pd.read_csv(config.REVIEWS_PATH, nrows=0).columns.tolist()
    available_meta = [c for c in _DESIRED_METADATA if c in header]
    print(f"[META] Found {len(available_meta)} metadata column(s): "
          f"{available_meta if available_meta else 'none'}")

    base_cols = ["app_name", "review", "recommended"]
    if config.ENGLISH_ONLY:
        base_cols.append("language")
    usecols = base_cols + available_meta
    keep_cols = ["app_name", "review", "recommended", "genre"] + available_meta

    print(f"Streaming reviews in {config.READ_CHUNK:,}-row chunks "
          f"(engine=c){' | English only' if config.ENGLISH_ONLY else ''}...")

    pool, pool_rows = [], 0
    n_raw = n_genre = n_len = 0

    reader = pd.read_csv(
        config.REVIEWS_PATH,
        usecols=usecols,
        chunksize=config.READ_CHUNK,
        engine="c",
        dtype={"review": "string", "app_name": "string"},
        on_bad_lines="skip",
    )
    for chunk in reader:
        n_raw += len(chunk)
        chunk = chunk.dropna(subset=["review", "recommended", "app_name"])

        if config.ENGLISH_ONLY:
            chunk = chunk[chunk["language"] == "english"].drop(columns="language")

        chunk["recommended"] = _coerce_recommended(chunk["recommended"])
        chunk = chunk.dropna(subset=["recommended"])
        chunk["recommended"] = chunk["recommended"].astype("int8")

        chunk["genre"] = chunk["app_name"].map(genre_map)
        chunk = chunk.dropna(subset=["genre"])
        n_genre += len(chunk)

        tok_len = chunk["review"].str.split().str.len()
        chunk = chunk[tok_len >= config.MIN_REVIEW_LEN]
        n_len += len(chunk)

        if len(chunk):
            pool.append(chunk[keep_cols])
            pool_rows += len(chunk)

        if pool_rows > config.POOL_CAP:
            pool = [pd.concat(pool, ignore_index=True)
                      .sample(config.POOL_CAP, random_state=config.RANDOM_STATE)]
            pool_rows = config.POOL_CAP

    full = pd.concat(pool, ignore_index=True)
    del pool

    print(f"After genre merge: {n_genre:,} reviews, {full['genre'].nunique()} genres")
    if n_genre:
        removed = n_genre - n_len
        print(f"[FILTER] Removed {removed:,} reviews under "
              f"{config.MIN_REVIEW_LEN} tokens ({removed / n_genre * 100:.1f}%)")

    # --- equal stratified sample per genre (robust target selection) ---------
    genre_counts = full[config.GENRE_COL].value_counts()
    ranked = genre_counts.sort_values(ascending=False)
    valid_genres, target_per_genre = [], 0
    for k in range(len(ranked), 0, -1):
        kept = ranked.iloc[:k]
        candidate_target = config.SAMPLE_SIZE // k
        if kept.min() >= candidate_target:
            valid_genres = kept.index.tolist()
            target_per_genre = candidate_target
            break

    if not valid_genres:
        raise ValueError(
            "No genre has enough reviews for the requested SAMPLE_SIZE. "
            "Lower SAMPLE_SIZE or raise POOL_CAP in config.py."
        )

    full = full[full[config.GENRE_COL].isin(valid_genres)]
    samples = [
        full[full[config.GENRE_COL] == g].sample(
            n=target_per_genre, random_state=config.RANDOM_STATE
        )
        for g in valid_genres
    ]
    result = pd.concat(samples).reset_index(drop=True)

    # plain str for text/label cols; metadata left as-is (numeric/bool)
    for col in ("app_name", "review", "genre"):
        result[col] = result[col].astype(str)

    print(f"[SAMPLE] {len(result):,} reviews | {len(valid_genres)} genres × "
          f"{target_per_genre:,} each")
    return result


def stream_a(text: str) -> str:
    """Stream A — LDA/VADER. Lowercase, strip non-alpha, drop stops, lemmatise."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = simple_preprocess(text, deacc=True)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def stream_b(text: str) -> str:
    """Stream B — DistilBERT/BERTopic. Minimal cleaning; keep case/punctuation."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply both preprocessing streams to the dataframe."""
    print("Running Stream A (LDA/VADER)...")
    df["clean_text"] = df[config.TEXT_COL].apply(stream_a)
    print("Running Stream B (DistilBERT/BERTopic)...")
    df["bert_text"] = df[config.TEXT_COL].apply(stream_b)
    return df