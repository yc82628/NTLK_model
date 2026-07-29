"""
bot_detection.py — Review authenticity screening.

IMPORTANT FRAMING
-----------------
This module does NOT identify "bots". No ground-truth labels for inauthentic
reviews exist in the Steam 2021 corpus, so nothing here can be validated as a
bot classifier. What it does is two separate, honestly-labelled things:

  1. PROVENANCE FLAGS (rule-based, observable facts). Each flag records
     something directly visible in the data — duplicated text, a free review
     copy, a verdict written at near-zero playtime. These are evidence, not
     inference, and each is individually reportable and defensible.

  2. BEHAVIOURAL ANOMALY SCORE (Isolation Forest, unsupervised). Flags reviews
     that are statistically unusual in feature space. Note that
     contamination=0.05 means 5% are flagged BY CONSTRUCTION — this is not a
     discovery that 5% are fake. On Steam, "unusual" often means genuine
     copypasta jokes, ASCII art, or non-native English.

Defensible wording for the thesis:
  "X% of reviews carried at least one authenticity concern flag, of which
   duplicated text accounted for Y%."
NOT defensible:
  "The model identified X% of reviews as bot-generated."
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config

# ── 7 original linguistic features (kept stable for reporting continuity) ──
FEATURE_COLS = [
    "feat_length", "feat_ttr", "feat_punct_density",
    "feat_exclamation", "feat_digit_ratio",
    "feat_vader_abs", "feat_bigram_diversity",
]

# ── duplicate-text features (always available: computed from review text) ──
DUP_FEATURE_COLS = [
    "feat_dup_group_size_log", "feat_is_duplicate", "feat_dup_distinct_games",
]

# ── behavioural metadata features (only if source columns were carried) ──
META_FEATURE_COLS = [
    "feat_votes_funny_log", "feat_funny_ratio", "feat_votes_helpful_log",
    "feat_weighted_vote_score", "feat_received_for_free",
    "feat_steam_purchase", "feat_early_access",
    "feat_playtime_log", "feat_chars_per_minute_played",
]

# ── rule-based provenance flags, reported individually ──
PROVENANCE_FLAGS = [
    "flag_copypasta", "flag_coordinated_duplicate",
    "flag_free_copy", "flag_near_zero_playtime", "flag_joke_review",
]

# Plain-English names used in figures (non-technical readers)
FEATURE_LABELS = {
    "feat_length": "Review length (characters)",
    "feat_ttr": "Vocabulary variety (unique / total words)",
    "feat_punct_density": "Punctuation density",
    "feat_exclamation": "Exclamation marks per character",
    "feat_digit_ratio": "Digits per character",
    "feat_vader_abs": "Sentiment strength (0 = neutral, 1 = extreme)",
    "feat_bigram_diversity": "Phrase variety (unique word pairs)",
    "feat_dup_group_size_log": "Times this text appears (log scale)",
    "feat_is_duplicate": "Text is duplicated elsewhere (0/1)",
    "feat_dup_distinct_games": "Number of games sharing this text",
    "feat_votes_funny_log": "'Funny' votes received (log scale)",
    "feat_funny_ratio": "Share of votes that were 'funny'",
    "feat_votes_helpful_log": "'Helpful' votes received (log scale)",
    "feat_weighted_vote_score": "Steam helpfulness score",
    "feat_received_for_free": "Copy received free (0/1)",
    "feat_steam_purchase": "Purchased on Steam (0/1)",
    "feat_early_access": "Written during early access (0/1)",
    "feat_playtime_log": "Playtime in minutes (log scale)",
    "feat_chars_per_minute_played": "Characters written per minute played",
}


# ── linguistic helpers ─────────────────────────────────────────────────
def _type_token_ratio(text: str) -> float:
    tokens = text.split()
    return len(set(tokens)) / max(len(tokens), 1)


def _punct_density(text: str) -> float:
    return sum(1 for c in text if c in ".,!?;:") / max(len(text), 1)


def _exclamation_ratio(text: str) -> float:
    return text.count("!") / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    return sum(c.isdigit() for c in text) / max(len(text), 1)


def _bigram_diversity(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / max(len(bigrams), 1)


# ── metadata helpers ───────────────────────────────────────────────────
def _first_present(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _to_int_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return (
        s.map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
        .fillna(0).astype(int)
    )


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


# ── duplicate / copypasta detection (observable, no metadata needed) ────
def add_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect reviews whose text is reused elsewhere in the corpus.

    This is the strongest authenticity signal available without labels, because
    it is an observed fact rather than a stylistic inference. We separate two
    very different phenomena:

      - COPYPASTA: identical text appearing across SEVERAL DIFFERENT games.
        On Steam this is usually a genuine human meme template, NOT fraud.
      - COORDINATED DUPLICATE: identical text repeated on a SINGLE game.
        This is the pattern consistent with review manipulation.

    Conflating the two would badly overstate fraud, so they are flagged apart.
    """
    norm = (
        df[config.TEXT_COL].astype(str).str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    group_size = norm.groupby(norm).transform("size")

    df["feat_dup_group_size_log"] = np.log1p(group_size)
    df["feat_is_duplicate"] = (group_size > 1).astype(int)

    tmp = pd.DataFrame({"_key": norm, "_game": df[config.GAME_COL].astype(str)})
    df["feat_dup_distinct_games"] = tmp.groupby("_key")["_game"].transform("nunique")

    df["flag_copypasta"] = (
        (df["feat_is_duplicate"] == 1) & (df["feat_dup_distinct_games"] > 1)
    ).astype(int)
    df["flag_coordinated_duplicate"] = (
        (df["feat_is_duplicate"] == 1)
        & (df["feat_dup_distinct_games"] == 1)
        & (group_size >= config.DUP_COORDINATED_MIN)
    ).astype(int)

    n_dup = int(df["feat_is_duplicate"].sum())
    print(f"[AUTH] Duplicated text: {n_dup:,} reviews ({n_dup/len(df)*100:.1f}%) "
          f"| copypasta across games: {int(df['flag_copypasta'].sum()):,} "
          f"| repeated on one game: {int(df['flag_coordinated_duplicate'].sum()):,}")
    return df


def _add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features from whatever metadata columns are present."""
    funny = _first_present(df, ["votes_funny"])
    helpful = _first_present(df, ["votes_helpful", "votes_up"])
    wscore = _first_present(df, ["weighted_vote_score"])
    free = _first_present(df, ["received_for_free"])
    purchase = _first_present(df, ["steam_purchase"])
    early = _first_present(df, ["written_during_early_access"])
    playtime = _first_present(
        df, ["author.playtime_at_review", "playtime_at_review",
             "author.playtime_forever", "playtime_forever"]
    )

    if funny is not None:
        f = _num(funny)
        h = _num(helpful) if helpful is not None else 0
        df["feat_votes_funny_log"] = np.log1p(f)
        df["feat_funny_ratio"] = f / (f + h + 1)
        # A review the community found mostly FUNNY rather than HELPFUL is a
        # joke review: sincere-but-not-literal, a known divergence source.
        df["flag_joke_review"] = (
            (f >= config.JOKE_MIN_FUNNY_VOTES) &
            (df["feat_funny_ratio"] >= config.JOKE_FUNNY_RATIO)
        ).astype(int)
    if helpful is not None:
        df["feat_votes_helpful_log"] = np.log1p(_num(helpful))
    if wscore is not None:
        df["feat_weighted_vote_score"] = _num(wscore)
    if free is not None:
        df["feat_received_for_free"] = _to_int_bool(free)
        df["flag_free_copy"] = df["feat_received_for_free"]
    if purchase is not None:
        df["feat_steam_purchase"] = _to_int_bool(purchase)
    if early is not None:
        df["feat_early_access"] = _to_int_bool(early)
    if playtime is not None:
        mins = _num(playtime)
        df["feat_playtime_log"] = np.log1p(mins)
        # A long, confident verdict written at near-zero playtime is a
        # content/experience mismatch that is directly observable.
        df["feat_chars_per_minute_played"] = (
            df[config.TEXT_COL].astype(str).str.len() / (mins + 1)
        )
        df["flag_near_zero_playtime"] = (
            (mins <= config.MIN_PLAUSIBLE_PLAYTIME_MIN) &
            (df[config.TEXT_COL].astype(str).str.len() >= config.LONG_REVIEW_CHARS)
        ).astype(int)

    return df


def extract_bot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Linguistic + duplicate + (any available) behavioural metadata features."""
    text = df[config.TEXT_COL].astype(str)
    df["feat_length"] = text.apply(len)
    df["feat_ttr"] = text.apply(_type_token_ratio)
    df["feat_punct_density"] = text.apply(_punct_density)
    df["feat_exclamation"] = text.apply(_exclamation_ratio)
    df["feat_digit_ratio"] = text.apply(_digit_ratio)
    df["feat_vader_abs"] = df["vader_compound"].abs()
    df["feat_bigram_diversity"] = text.apply(_bigram_diversity)

    df = add_duplicate_features(df)
    df = _add_metadata_features(df)
    return df


def summarise_provenance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count each rule-based flag separately and build an overall concern count.
    Reporting flags individually (rather than one opaque score) is what makes
    this defensible: each number traces back to an observable fact.
    """
    present = [f for f in PROVENANCE_FLAGS if f in df.columns]
    for f in PROVENANCE_FLAGS:
        if f not in df.columns:
            df[f] = 0

    df["authenticity_concern_count"] = df[PROVENANCE_FLAGS].sum(axis=1)
    df["has_authenticity_concern"] = (df["authenticity_concern_count"] > 0).astype(int)

    rows = []
    labels = {
        "flag_copypasta": "Duplicated text across multiple games (likely meme)",
        "flag_coordinated_duplicate": "Duplicated text on a single game",
        "flag_free_copy": "Review copy received free",
        "flag_near_zero_playtime": "Long review at near-zero playtime",
        "flag_joke_review": "Voted mostly 'funny' rather than 'helpful'",
    }
    for f in PROVENANCE_FLAGS:
        n = int(df[f].sum())
        rows.append({
            "flag": f, "description": labels[f],
            "n_reviews": n, "pct_of_corpus": round(n / len(df) * 100, 2),
            "available": f in present,
        })
    table = pd.DataFrame(rows)

    print("\n[AUTH] Provenance flags (observable evidence, not a bot verdict):")
    for r in rows:
        status = "" if r["available"] else "  (metadata unavailable)"
        print(f"   {r['description']}: {r['n_reviews']:,} "
              f"({r['pct_of_corpus']:.2f}%){status}")
    n_any = int(df["has_authenticity_concern"].sum())
    print(f"[AUTH] At least one concern flag: {n_any:,} "
          f"({n_any/len(df)*100:.1f}% of corpus)")
    return table


def run_bot_detection(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Fit Isolation Forest on all available features and flag the top
    `contamination` fraction as behaviourally anomalous.

    Reminder: contamination fixes the flagged proportion in advance. The useful
    output is not "how many" but "whether flagged reviews behave differently"
    — hence the divergence comparison below.
    """
    model_features = (
        FEATURE_COLS
        + [c for c in DUP_FEATURE_COLS if c in df.columns]
        + [c for c in META_FEATURE_COLS if c in df.columns]
    )
    n_extra = len(model_features) - len(FEATURE_COLS)
    print(f"\n[ANOMALY] Using {len(model_features)} features "
          f"({len(FEATURE_COLS)} linguistic + {n_extra} duplicate/behavioural)")

    X = df[model_features].fillna(0).values
    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    df["anomaly_score"] = iso.fit_predict(X)
    df["is_suspicious"] = (df["anomaly_score"] == -1).astype(int)
    print(f"[ANOMALY] Flagged {df['is_suspicious'].sum():,} reviews "
          f"({df['is_suspicious'].mean()*100:.1f}%) — proportion set by "
          f"contamination={contamination}, not discovered")

    if "distilbert_divergence_confident" in df.columns:
        div_susp = df[df["is_suspicious"] == 1]["distilbert_divergence_confident"].mean()
        div_norm = df[df["is_suspicious"] == 0]["distilbert_divergence_confident"].mean()
        print(f"[ANOMALY] Confident divergence — anomalous: {div_susp:.3f} | "
              f"typical: {div_norm:.3f} | difference: {(div_susp-div_norm)*100:+.1f}pp")

        # same comparison for the rule-based flags, which is the stronger claim
        if df["has_authenticity_concern"].sum() > 0:
            d_flag = df[df["has_authenticity_concern"] == 1]["distilbert_divergence_confident"].mean()
            d_clean = df[df["has_authenticity_concern"] == 0]["distilbert_divergence_confident"].mean()
            print(f"[AUTH] Confident divergence — flagged: {d_flag:.3f} | "
                  f"unflagged: {d_clean:.3f} | difference: {(d_flag-d_clean)*100:+.1f}pp")

    return df