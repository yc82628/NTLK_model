"""
evaluation.py — Metrics computation and all figure generation
for Chapter 4 (Findings, Analysis and Discussion).
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless/local script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config


def eval_classifier(y_true, y_pred, name: str) -> dict:
    """Print a classification report and return a metrics dict."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"\n{name}")
    print(classification_report(y_true, y_pred, target_names=["Not Rec.", "Rec."]))
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1}


def save_confusion_matrix(y_true, y_pred, title: str, fname: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Rec.", "Rec."],
        yticklabels=["Not Rec.", "Rec."],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}")


def save_vader_threshold_plot(sweep_results, best_threshold):
    sweep_df = pd.DataFrame(sweep_results, columns=["threshold", "macro_f1"])
    plt.figure(figsize=(8, 4.5))
    plt.plot(sweep_df["threshold"], sweep_df["macro_f1"], marker="o", color="#3498db")
    plt.axvline(best_threshold, color="red", linestyle="--",
                label=f"Selected threshold = {best_threshold:.2f}")
    plt.axvline(0.05, color="gray", linestyle=":", label="VADER default = 0.05")
    plt.xlabel("VADER Compound Threshold")
    plt.ylabel("Macro F1-score")
    plt.title("VADER Threshold Tuning on Validation Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/vader_threshold_tuning.png", dpi=150)
    plt.close()
    print("[FIG] Saved vader_threshold_tuning.png")


def save_lda_k_selection_plot(scores: dict, best_k: int):
    plt.figure(figsize=(8, 4))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o", color="teal")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("C_v Coherence")
    plt.title("LDA K-Selection via Coherence Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/lda_k_selection.png", dpi=150)
    plt.close()
    print("[FIG] Saved lda_k_selection.png")


def save_coherence_comparison(lda_coh: float, bert_coh: float, bert_scores: list):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(["LDA", "BERTopic"], [lda_coh, bert_coh],
                color=["#3498db", "#9b59b6"], edgecolor="white")
    for i, v in enumerate([lda_coh, bert_coh]):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
    axes[0].set_title("C_v Coherence Score")
    axes[0].set_ylabel("Score")

    axes[1].bar(range(len(bert_scores)), bert_scores, color="#9b59b6", alpha=0.7)
    axes[1].axhline(np.mean(bert_scores), color="red", linestyle="--",
                     label=f"Mean={np.mean(bert_scores):.4f}")
    axes[1].set_title("BERTopic Stability Across Runs")
    axes[1].set_xlabel("Run Number")
    axes[1].set_ylabel("C_v Coherence")
    axes[1].legend()

    plt.suptitle("Topic Model Comparison: LDA vs BERTopic", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/coherence_comparison.png", dpi=150)
    plt.close()
    print("[FIG] Saved coherence_comparison.png")


def save_divergence_heatmap(pivot: pd.DataFrame, fname: str, title: str):
    fig_w = max(10, pivot.shape[1] // 2)
    plt.figure(figsize=(fig_w, 6))
    sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".2f",
                cbar_kws={"label": "Divergence Rate"})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}")


def save_bot_feature_plots(df: pd.DataFrame):
    from bot_detection import FEATURE_COLS

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        for status, grp in df.groupby("is_suspicious"):
            label = "Suspicious" if status == 1 else "Normal"
            grp[col].clip(upper=grp[col].quantile(0.99)).hist(
                ax=axes[i], bins=40, alpha=0.6, label=label, density=True
            )
        axes[i].set_title(col.replace("feat_", "").replace("_", " ").title())
        axes[i].legend(fontsize=8)
    axes[-1].set_visible(False)
    plt.suptitle("Bot Detection Feature Distributions")
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/bot_detection_features.png", dpi=150)
    plt.close()
    print("[FIG] Saved bot_detection_features.png")
