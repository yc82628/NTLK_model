"""
evaluation.py — Metrics computation and all figure generation for Chapter 4.

FIGURE DESIGN NOTE
------------------
Every chart is written for a non-technical reader: axes state what is measured
AND its unit or range, values are annotated directly on the plot, and jargon is
expanded in a subtitle. Where a statistic can mislead (a 100% rate computed
from two reviews), the figure suppresses it rather than displaying it.
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless/local script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

import config

# consistent, readable defaults for every figure
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _subtitle(fig, text: str):
    """Add a plain-English explanatory line under the title."""
    fig.text(0.5, 0.955, text, ha="center", fontsize=9,
             color="#444444", style="italic")


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
    """
    Confusion matrix annotated with both counts and row percentages, so a
    reader can see immediately how often each true class was got right.
    """
    cm = confusion_matrix(y_true, y_pred)
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", ax=ax, cbar_kws={"label": "Number of reviews"},
        xticklabels=["Predicted:\nNot Recommended", "Predicted:\nRecommended"],
        yticklabels=["Actual:\nNot Recommended", "Actual:\nRecommended"],
    )
    ax.set_xlabel("What the model predicted")
    ax.set_ylabel("What the reviewer actually chose (Steam thumb)")
    ax.set_title(title, pad=28)
    acc = accuracy_score(y_true, y_pred) * 100
    _subtitle(fig, f"Percentages are row-wise (share of each actual class). "
                   f"Overall accuracy: {acc:.1f}%")
    ax.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}")


def save_vader_threshold_plot(sweep_results, best_threshold):
    sweep_df = pd.DataFrame(sweep_results, columns=["threshold", "macro_f1"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(sweep_df["threshold"], sweep_df["macro_f1"], marker="o", color="#3498db",
            label="Agreement with Steam thumbs")
    ax.axvline(best_threshold, color="red", linestyle="--",
               label=f"Chosen cut-off = {best_threshold:.2f}")
    ax.axvline(0.05, color="gray", linestyle=":", label="VADER standard = 0.05")
    ax.set_xlabel("VADER sentiment cut-off score  (0 = neutral, higher = stricter positive)")
    ax.set_ylabel("Macro F1-score  (0 = worst, 1 = perfect)")
    ax.set_title("Choosing VADER's positive/negative cut-off", pad=28)
    _subtitle(fig, "Higher F1 means VADER's sentiment agreed more often with the "
                   "reviewer's own thumbs up/down.")
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{config.FIG_DIR}/vader_threshold_tuning.png", dpi=150)
    plt.close()
    print("[FIG] Saved vader_threshold_tuning.png")


def save_lda_k_selection_plot(scores: dict, best_k: int):
    ks, vals = list(scores.keys()), list(scores.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, vals, marker="o", color="teal")
    ax.axvline(best_k, color="red", linestyle="--",
               label=f"Selected: {best_k} topics (highest score)")
    ax.annotate(f"{scores[best_k]:.4f}", xy=(best_k, scores[best_k]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontweight="bold", color="red")
    ax.set_xlabel("Number of topics the model was asked to find (K)")
    ax.set_ylabel("Topic coherence, C_v  (0 = incoherent, 1 = highly coherent)")
    ax.set_title("How many topics best describe the reviews?", pad=28)
    _subtitle(fig, "Coherence measures whether each topic's top words genuinely "
                   "belong together. Note the curve is noisy — nearby K values "
                   "score similarly.")
    ax.set_xticks(ks)
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{config.FIG_DIR}/lda_k_selection.png", dpi=150)
    plt.close()
    print("[FIG] Saved lda_k_selection.png")


def save_coherence_comparison(lda_coh: float, bert_coh: float, bert_scores: list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    axes[0].bar(["LDA", "BERTopic"], [lda_coh, bert_coh],
                color=["#3498db", "#9b59b6"], edgecolor="white")
    for i, v in enumerate([lda_coh, bert_coh]):
        axes[0].text(i, v + 0.008, f"{v:.4f}", ha="center", fontweight="bold")
    axes[0].set_title("Topic quality: which model found clearer topics?")
    axes[0].set_ylabel("Topic coherence, C_v  (0 = incoherent, 1 = coherent)")
    axes[0].set_xlabel("Topic modelling method")
    axes[0].set_ylim(0, max(lda_coh, bert_coh) * 1.2)

    axes[1].bar(range(1, len(bert_scores) + 1), bert_scores,
                color="#9b59b6", alpha=0.75, edgecolor="white")
    axes[1].axhline(np.mean(bert_scores), color="red", linestyle="--",
                    label=f"Average = {np.mean(bert_scores):.4f} "
                          f"(spread ±{np.std(bert_scores):.4f})")
    for i, v in enumerate(bert_scores, start=1):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_title("Is BERTopic stable when repeated?")
    axes[1].set_xlabel("Repeat run number (different random seed)")
    axes[1].set_ylabel("Topic coherence, C_v")
    axes[1].set_xticks(range(1, len(bert_scores) + 1))
    axes[1].legend(fontsize=9)

    plt.suptitle("Topic Model Comparison: LDA vs BERTopic", fontweight="bold", y=1.0)
    _subtitle(fig, "Caution: BERTopic's words are scored against the LDA "
                   "vocabulary, which penalises it — see methodology.")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{config.FIG_DIR}/coherence_comparison.png", dpi=150)
    plt.close()
    print("[FIG] Saved coherence_comparison.png")


def save_divergence_heatmap_readable(
    df: pd.DataFrame, topic_col: str, fname: str, title: str,
    value_col: str = "distilbert_divergence_confident",
    top_n: int = None, min_cell: int = None,
):
    """
    Genre x topic divergence heatmap, made readable and honest:
      - drops the BERTopic outlier group (-1), which is not a topic
      - keeps only the highest-volume topics (config.HEATMAP_TOP_N_TOPICS)
      - blanks cells with too few reviews to support a percentage
        (config.HEATMAP_MIN_CELL_COUNT) instead of printing 0% / 100%
    """
    top_n = top_n or config.HEATMAP_TOP_N_TOPICS
    min_cell = min_cell or config.HEATMAP_MIN_CELL_COUNT

    d = df[df[topic_col] != -1]
    keep = d[topic_col].value_counts().nlargest(top_n).index.tolist()
    keep = sorted(keep)
    d = d[d[topic_col].isin(keep)]

    rates = d.pivot_table(index=config.GENRE_COL, columns=topic_col,
                          values=value_col, aggfunc="mean") * 100
    counts = d.pivot_table(index=config.GENRE_COL, columns=topic_col,
                           values=value_col, aggfunc="count")
    masked = rates.where(counts >= min_cell)

    n_hidden = int((counts < min_cell).sum().sum())
    fig_w = max(9, 0.75 * masked.shape[1] + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 0.75 * masked.shape[0] + 3.4))
    sns.heatmap(
        masked, cmap="Reds", annot=True, fmt=".1f", ax=ax,
        linewidths=0.5, linecolor="white", vmin=0,
        cbar_kws={"label": "Reviews where sentiment disagreed\nwith the thumb (%)"},
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel(f"Discussion topic number  (top {len(keep)} topics by review volume)")
    ax.set_ylabel("Game genre")
    ax.set_title(title, pad=30)
    _subtitle(fig, f"Each cell = % of reviews where the model's sentiment "
                   f"disagreed with the reviewer's thumb. Blank = fewer than "
                   f"{min_cell} reviews ({n_hidden} cells hidden).")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved {fname} (top {len(keep)} topics, "
          f"{n_hidden} low-count cells hidden)")


def save_divergence_by_genre_bar(div_table: pd.DataFrame, fname="divergence_by_genre.png"):
    """Plain bar chart of divergence rate per genre — the most readable summary."""
    d = div_table.sort_values("distilbert_div_confident", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.bar(d[config.GENRE_COL], d["distilbert_div_confident"] * 100,
                  color="#c0392b", alpha=0.85, edgecolor="white")
    for b, n in zip(bars, d["n_reviews"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{b.get_height():.1f}%\n(n={int(n):,})",
                ha="center", fontsize=9)
    ax.set_xlabel("Game genre")
    ax.set_ylabel("Reviews where sentiment disagreed with the thumb (%)")
    ax.set_title("Where does written sentiment disagree with the star rating?", pad=28)
    _subtitle(fig, "Only confident model predictions counted. n = reviews per genre.")
    ax.set_ylim(0, max(d["distilbert_div_confident"] * 100) * 1.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}")


def save_authenticity_flag_chart(flag_table: pd.DataFrame,
                                 fname="authenticity_flags.png"):
    """Horizontal bar chart of how many reviews carry each provenance flag."""
    d = flag_table[flag_table["available"]].sort_values("n_reviews")
    if d.empty:
        print("[FIG] Skipped authenticity_flags.png (no flags available)")
        return
    fig, ax = plt.subplots(figsize=(10, 0.85 * len(d) + 3))
    bars = ax.barh(d["description"], d["n_reviews"], color="#e67e22",
                   alpha=0.85, edgecolor="white")
    for b, pct in zip(bars, d["pct_of_corpus"]):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                f"{int(b.get_width()):,}  ({pct:.2f}%)", va="center", fontsize=9)
    ax.set_xlabel("Number of reviews carrying this flag")
    ax.set_ylabel("")
    ax.set_title("Reviews flagged for authenticity concerns", pad=30)
    _subtitle(fig, "These are observable facts about each review, not a "
                   "judgement that it is fake. Duplicated meme text is usually "
                   "posted by real players.")
    ax.set_xlim(0, d["n_reviews"].max() * 1.35)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}")


def save_bot_feature_plots(df: pd.DataFrame):
    """
    Distribution of each feature for flagged vs typical reviews, with
    plain-English feature names and labelled axes.
    """
    from bot_detection import (FEATURE_COLS, DUP_FEATURE_COLS,
                               META_FEATURE_COLS, FEATURE_LABELS)

    cols = [c for c in FEATURE_COLS + DUP_FEATURE_COLS + META_FEATURE_COLS
            if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        for status, grp in df.groupby("is_suspicious"):
            label = "Flagged as unusual" if status == 1 else "Typical review"
            vals = grp[col].clip(upper=grp[col].quantile(0.99))
            ax.hist(vals, bins=40, alpha=0.6, label=label, density=True)
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=10)
        ax.set_xlabel("Measured value")
        ax.set_ylabel("Relative frequency")
        ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("What makes a review look unusual?", fontweight="bold", y=1.0)
    _subtitle(fig, "Heavy overlap between the two groups means that feature "
                   "does NOT cleanly separate flagged from typical reviews.")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{config.FIG_DIR}/bot_detection_features.png", dpi=150)
    plt.close()
    print("[FIG] Saved bot_detection_features.png")


def save_roc_curves(y_true, score_dict: dict, fname="roc_curves.png"):
    """
    ROC curve comparing every scoring method on one axis.

    WHY THIS MATTERS FOR THIS PROJECT
    ---------------------------------
    Accuracy and F1 depend on where you put the decision cut-off. ROC-AUC does
    not: it measures how well a method RANKS reviews from "clearly negative" to
    "clearly positive", across every possible cut-off. That makes it the fair
    way to compare VADER against DistilBERT without the comparison hinging on
    the VADER threshold choice.

    Reading the number: 1.00 = perfect ranking, 0.50 = no better than a coin
    flip, below 0.50 = worse than chance.
    """
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    results = {}
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        results[name] = auc_val
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1,
            label="Random guessing  (AUC = 0.500)")
    ax.set_xlabel("False positive rate\n"
                  "(share of NOT-recommended reviews wrongly scored as positive)")
    ax.set_ylabel("True positive rate\n"
                  "(share of recommended reviews correctly scored as positive)")
    ax.set_title("ROC: how well does each method rank reviews?", pad=30)
    _subtitle(fig, "Higher and further toward the top-left is better. "
                   "AUC is independent of any chosen cut-off.")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}  " +
          " | ".join(f"{k} AUC={v:.3f}" for k, v in results.items()))
    return results


def save_pr_curves(y_true, score_dict: dict, fname="precision_recall_curves.png"):
    """
    Precision-Recall curve — more informative than ROC on a skewed dataset.

    Roughly 82% of Steam reviews are positive, so a method can look strong on
    ROC simply because negatives are rare. The PR curve focuses on how well the
    positive class is retrieved, and the dashed baseline shows what a model
    that guessed "recommended" every time would achieve.
    """
    import numpy as _np
    y_true = _np.asarray(y_true)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    results = {}
    for name, scores in score_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        results[name] = ap
        ax.plot(rec, prec, linewidth=2, label=f"{name}  (avg precision = {ap:.3f})")

    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Always predict 'recommended'  ({baseline:.3f})")
    ax.set_xlabel("Recall\n(share of all recommended reviews that were found)")
    ax.set_ylabel("Precision\n(share of positive predictions that were correct)")
    ax.set_title("Precision vs Recall for the positive class", pad=30)
    _subtitle(fig, "Curves above the dashed line beat the naive 'everything is "
                   "positive' baseline. Higher and further right is better.")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}  " +
          " | ".join(f"{k} AP={v:.3f}" for k, v in results.items()))
    return results


# kept for backward compatibility with any older call site
def save_divergence_heatmap(pivot: pd.DataFrame, fname: str, title: str):
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] // 2), 6))
    sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Divergence Rate"})
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{config.FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"[FIG] Saved {fname}")