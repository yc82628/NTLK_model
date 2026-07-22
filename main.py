"""
main.py — Orchestrates the full Steam review NLP pipeline.

Run this after completing the setup steps in README.md:
    python check_gpu.py         # verify GPU first
    python download_data.py     # pull datasets from Kaggle
    python main.py               # run the full pipeline
"""

import time

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import bot_detection
import config
import divergence
import evaluation
import preprocessing
import sentiment
import topic_modelling
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    start_time = time.time()
    config.ensure_dirs()

    # ── Step 1: Load, merge genre, sample, preprocess ──────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — DATA LOADING & PREPROCESSING")
    print("=" * 60)
    df = preprocessing.load_merge_and_sample()
    df = preprocessing.preprocess(df)

    # ── Step 2: VADER with tuned threshold ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — VADER SENTIMENT ANALYSIS")
    print("=" * 60)
    val_for_tuning = df.sample(frac=0.1, random_state=config.RANDOM_STATE)
    best_threshold, sweep_results = sentiment.tune_vader_threshold(
        val_for_tuning, SentimentIntensityAnalyzer()
    )
    df = sentiment.run_vader(df, threshold=best_threshold)
    evaluation.save_vader_threshold_plot(sweep_results, best_threshold)

    # ── Step 3: Group-based train/test split ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — GROUP-BASED TRAIN/TEST SPLIT")
    print("=" * 60)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df[config.GAME_COL]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df[config.GAME_COL]) & set(test_df[config.GAME_COL])
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Games shared between train/test: {len(overlap)} (should be 0)")

    # ── Step 4: DistilBERT fine-tuning ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — DISTILBERT FINE-TUNING")
    print("=" * 60)
    trainer, tokenizer = sentiment.fine_tune_distilbert(train_df, test_df)

    df["distilbert_pred"], df["distilbert_confidence"] = (
        sentiment.distilbert_predict_with_confidence(trainer, df["bert_text"], tokenizer)
    )
    test_df["distilbert_pred"], test_df["distilbert_confidence"] = (
        sentiment.distilbert_predict_with_confidence(trainer, test_df["bert_text"], tokenizer)
    )
    # NOTE: test_df already carries the correct vader_pred/vader_neutral/vader_compound
    # from the Step 3 split (VADER ran on the full df in Step 2). The previous
    # `df.loc[test_df.index, ...]` line looked up the reset 0..N index as *labels*
    # and silently pulled the wrong rows, corrupting every VADER metric. Removed.

    # ── Step 5: Evaluate classifiers ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — CLASSIFIER EVALUATION")
    print("=" * 60)
    metrics = [
        evaluation.eval_classifier(test_df[config.LABEL_COL], test_df["vader_pred"], "VADER"),
        evaluation.eval_classifier(test_df[config.LABEL_COL], test_df["distilbert_pred"], "DistilBERT"),
    ]
    pd.DataFrame(metrics).to_csv(f"{config.TABLE_DIR}/sentiment_metrics.csv", index=False)

    evaluation.save_confusion_matrix(
        test_df[config.LABEL_COL], test_df["vader_pred"],
        "VADER Confusion Matrix", "vader_cm.png",
    )
    evaluation.save_confusion_matrix(
        test_df[config.LABEL_COL], test_df["distilbert_pred"],
        "DistilBERT Confusion Matrix", "distilbert_cm.png",
    )

    # ── Step 6: Topic modelling ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — LDA TOPIC MODELLING")
    print("=" * 60)
    lda_model, lda_vec, best_k, lda_coh, lda_words, lda_scores = topic_modelling.run_lda(df)
    pd.DataFrame(lda_words).to_csv(f"{config.TABLE_DIR}/lda_top_words.csv", index=False)
    evaluation.save_lda_k_selection_plot(lda_scores, best_k)

    print("\n" + "=" * 60)
    print("STEP 7 — BERTOPIC (WITH STABILITY CHECK)")
    print("=" * 60)
    topic_model, bert_coh, bert_words, bert_run_scores = (
        topic_modelling.run_bertopic_with_stability_check(df)
    )
    pd.DataFrame.from_dict(bert_words, orient="index").to_csv(
        f"{config.TABLE_DIR}/bertopic_top_words.csv"
    )
    evaluation.save_coherence_comparison(lda_coh, bert_coh, bert_run_scores)

    # ── Step 8: Divergence analysis ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — DIVERGENCE ANALYSIS")
    print("=" * 60)
    df = divergence.compute_divergence(df)
    div_table = divergence.divergence_by_genre(df)
    print(div_table.to_string(index=False))
    div_table.to_csv(f"{config.TABLE_DIR}/divergence_by_genre.csv", index=False)

    topic_div = divergence.divergence_by_topic(df)
    topic_div.to_csv(f"{config.TABLE_DIR}/divergence_by_topic.csv")

    pivot = divergence.divergence_genre_topic_pivot(df)
    evaluation.save_divergence_heatmap(
        pivot, "divergence_heatmap.png",
        "Confidence-Filtered Divergence Rate by Genre × Topic",
    )

    # ── Step 9: Bot / adversarial detection ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9 — BOT / ADVERSARIAL DETECTION")
    print("=" * 60)
    df = bot_detection.extract_bot_features(df)
    df = bot_detection.run_bot_detection(df)
    evaluation.save_bot_feature_plots(df)

    # ── Step 10: Save final annotated dataset ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 10 — SAVING RESULTS")
    print("=" * 60)
    df.to_csv(f"{config.OUT_DIR}/annotated_reviews.csv", index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"\nPipeline complete in {elapsed:.1f} minutes.")
    print(f"All results saved to '{config.OUT_DIR}/'")


if __name__ == "__main__":
    main()
