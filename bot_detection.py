"""
bot_detection.py — Unsupervised adversarial/bot review detection.

Combines 7 hand-crafted linguistic features with per-review behavioural
metadata (funny-vote signal, free-copy flag, playtime, community vote score)
when those columns are available. Isolation Forest flags the top N% most
anomalous reviews; we then check whether flagged reviews disproportionately
drive sentiment-label divergence.

No labelled fraud data is required. Isolation Forest is tree-based and
scale-invariant, so mixing log-counts, 0-1 ratios and booleans is fine.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config

# 7 linguistic features — kept as a fixed list so evaluation.save_bot_feature_plots
# (which imports FEATURE_COLS) still renders its 2x4 grid unchanged.
FEATURE_COLS = [
    "feat_length", "feat_ttr", "feat_punct_density",
    "feat_exclamation", "feat_digit_ratio",
    "feat_vader_abs", "feat_bigram_diversity",
]

# Behavioural metadata features, added to the model ONLY when their source
# columns were carried through by preprocessing. The model uses FEATURE_COLS
# plus whichever of these exist in the dataframe.
META_FEATURE_COLS = [
    "feat_votes_funny_log", "feat_funny_ratio", "feat_votes_helpful_log",
    "feat_weighted_vote_score", "feat_received_for_free",
    "feat_steam_purchase", "feat_early_access", "feat_playtime_log",
]


# ── linguistic feature helpers ─────────────────────────────────────────
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
    """Return the first existing column (as a Series) from candidate names."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _to_int_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return (
        s.map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavioural features from whatever metadata is present."""
    funny = _first_present(df, ["votes_funny"])
    helpful = _first_present(df, ["votes_helpful", "votes_up"])
    wscore = _first_present(df, ["weighted_vote_score"])
    free = _first_present(df, ["received_for_free"])
    purchase = _first_present(df, ["steam_purchase"])
    early = _first_present(df, ["written_during_early_access"])
    playtime = _first_present(
        df, ["author.playtime_forever", "playtime_forever",
             "author.playtime_at_review", "playtime_at_review"]
    )

    if funny is not None:
        f = _num(funny)
        df["feat_votes_funny_log"] = np.log1p(f)
        h = _num(helpful) if helpful is not None else 0
        # high funny share vs helpful is a classic joke/meme-review signal
        df["feat_funny_ratio"] = f / (f + h + 1)
    if helpful is not None:
        df["feat_votes_helpful_log"] = np.log1p(_num(helpful))
    if wscore is not None:
        df["feat_weighted_vote_score"] = _num(wscore)
    if free is not None:
        df["feat_received_for_free"] = _to_int_bool(free)
    if purchase is not None:
        df["feat_steam_purchase"] = _to_int_bool(purchase)
    if early is not None:
        df["feat_early_access"] = _to_int_bool(early)
    if playtime is not None:
        df["feat_playtime_log"] = np.log1p(_num(playtime))

    return df


def extract_bot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute linguistic features + any available behavioural metadata features."""
    text = df[config.TEXT_COL].astype(str)
    df["feat_length"] = text.apply(len)
    df["feat_ttr"] = text.apply(_type_token_ratio)
    df["feat_punct_density"] = text.apply(_punct_density)
    df["feat_exclamation"] = text.apply(_exclamation_ratio)
    df["feat_digit_ratio"] = text.apply(_digit_ratio)
    df["feat_vader_abs"] = df["vader_compound"].abs()
    df["feat_bigram_diversity"] = text.apply(_bigram_diversity)

    df = _add_metadata_features(df)
    return df


def run_bot_detection(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Fit Isolation Forest on linguistic + available metadata features and flag
    the top `contamination` fraction of reviews as suspicious.
    """
    model_features = FEATURE_COLS + [c for c in META_FEATURE_COLS if c in df.columns]
    n_meta = len(model_features) - len(FEATURE_COLS)
    print(f"[BOT] Using {len(model_features)} features "
          f"({len(FEATURE_COLS)} linguistic + {n_meta} behavioural metadata)")

    X = df[model_features].fillna(0).values
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    df["anomaly_score"] = iso.fit_predict(X)
    df["is_suspicious"] = (df["anomaly_score"] == -1).astype(int)

    flagged_pct = df["is_suspicious"].mean() * 100
    print(f"[BOT] {flagged_pct:.1f}% flagged as suspicious "
          f"({df['is_suspicious'].sum():,} reviews)")

    if "distilbert_divergence_confident" in df.columns:
        div_susp = df[df["is_suspicious"] == 1]["distilbert_divergence_confident"].mean()
        div_norm = df[df["is_suspicious"] == 0]["distilbert_divergence_confident"].mean()
        print(f"[BOT] Confident divergence — suspicious: {div_susp:.3f} | "
              f"normal: {div_norm:.3f} | uplift: {(div_susp-div_norm)*100:+.1f}pp")

    return df