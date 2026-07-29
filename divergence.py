"""
divergence.py — Compares computed sentiment (VADER, DistilBERT)
against Steam's binary recommendation label, at both genre and
topic level. This is the dissertation's central novel contribution.
"""

import pandas as pd

import config


def compute_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds divergence columns to the dataframe:
      - vader_divergence: mismatch OR neutral VADER label
      - distilbert_divergence: simple binary mismatch
      - distilbert_divergence_confident: mismatch AND confidence
        above DIVERGENCE_CONFIDENCE_THRESHOLD (filters out noisy,
        low-confidence disagreements)
    """
    df["vader_divergence"] = (
        (df["vader_pred"] != df[config.LABEL_COL]) | (df["vader_neutral"] == 1)
    ).astype(int)

    df["distilbert_divergence"] = (
        df["distilbert_pred"] != df[config.LABEL_COL]
    ).astype(int)

    df["distilbert_divergence_confident"] = (
        (df["distilbert_pred"] != df[config.LABEL_COL])
        & (df["distilbert_confidence"] >= config.DIVERGENCE_CONFIDENCE_THRESHOLD)
    ).astype(int)

    return df


def divergence_by_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table of divergence rates per genre."""
    table = df.groupby(config.GENRE_COL).agg(
        vader_div_rate=("vader_divergence", "mean"),
        distilbert_div_rate=("distilbert_divergence", "mean"),
        distilbert_div_confident=("distilbert_divergence_confident", "mean"),
        neutral_rate=("vader_neutral", "mean"),
        n_reviews=(config.TEXT_COL, "count"),
    ).reset_index()
    return table


def divergence_by_topic(df: pd.DataFrame, topic_col: str = "bertopic_topic") -> pd.Series:
    """Which topics are most associated with confident divergence."""
    return (
        df.groupby(topic_col)["distilbert_divergence_confident"]
        .mean()
        .sort_values(ascending=False)
    )


def divergence_genre_topic_pivot(df: pd.DataFrame, topic_col: str = "bertopic_topic"):
    """Pivot table for the genre × topic divergence heatmap."""
    return df.pivot_table(
        index=config.GENRE_COL,
        columns=topic_col,
        values="distilbert_divergence_confident",
        aggfunc="mean",
    )
