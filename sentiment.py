"""
sentiment.py — VADER (with data-driven threshold tuning) and
DistilBERT (with class weighting and confidence scoring).

MEMORY NOTES
------------
- Tokenisation is lazy (datasets.map) and UNPADDED; a DataCollatorWithPadding
  pads each *batch* to its own longest sequence at train time. The old code
  padded every review to MAX_SEQ_LEN up front, holding a huge dense integer
  block in RAM before training even started.
- Prediction runs in mini-batches, so we never build one giant padded tensor
  over all 50k reviews at once.
- fp16 is enabled automatically on CUDA (halves GPU memory, speeds training).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── VADER ──────────────────────────────────────────────────────────────
def tune_vader_threshold(df_val: pd.DataFrame, analyzer: SentimentIntensityAnalyzer):
    """
    Sweep thresholds 0.00 → 0.30 and pick the one maximising macro-F1
    against the actual recommendation labels, rather than trusting
    VADER's generic 0.05 social-media default blindly.
    """
    compounds = df_val[config.TEXT_COL].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    y_true = df_val[config.LABEL_COL].values

    best_t, best_f1 = 0.05, -1
    results = []
    for t in np.arange(0.00, 0.31, 0.02):
        preds = (compounds >= t).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        results.append((t, f1))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"[VADER-TUNE] Best threshold = {best_t:.2f} "
          f"(macro-F1={best_f1:.4f}) vs default 0.05")
    return best_t, results


def run_vader(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df["vader_compound"] = df[config.TEXT_COL].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    df["vader_label"] = df["vader_compound"].apply(
        lambda c: "positive" if c >= threshold
        else ("negative" if c <= -threshold else "neutral")
    )
    df["vader_pred"] = (df["vader_label"] == "positive").astype(int)
    df["vader_neutral"] = (df["vader_label"] == "neutral").astype(int)
    return df


# ── DistilBERT ─────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to correct for
    Steam's typical positive-label skew."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def fine_tune_distilbert(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fine-tune distilbert-base-uncased with class weighting."""
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df[config.LABEL_COL].values,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"[WEIGHTS] Class weights (0=Not Rec., 1=Rec.): {class_weights}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)

    # [MEM] lazy, unpadded tokenisation into an Arrow-backed dataset
    def make_ds(texts, labels):
        ds = Dataset.from_dict({"text": list(texts), "labels": list(labels)})
        ds = ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=config.MAX_SEQ_LEN),
            batched=True,
            remove_columns=["text"],
        )
        return ds

    tr_ds = make_ds(train_df["bert_text"], train_df[config.LABEL_COL])
    val_ds = make_ds(test_df["bert_text"], test_df[config.LABEL_COL])

    collator = DataCollatorWithPadding(tokenizer)   # [MEM] per-batch padding

    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2
    )

    args = TrainingArguments(
        output_dir=f"{config.MODEL_DIR}/distilbert_steam",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        num_train_epochs=config.EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,                       # [MEM] keep only the best checkpoint
        fp16=torch.cuda.is_available(),           # [MEM] half precision on GPU only
        logging_steps=100,
        report_to="none",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {"f1_macro": f1, "accuracy": accuracy_score(labels, preds)}

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tr_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,   # transformers >=4.46 (was tokenizer=)
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )
    trainer.train()
    return trainer, tokenizer


@torch.no_grad()
def distilbert_predict_with_confidence(trainer, texts: pd.Series, tokenizer,
                                       batch_size: int = 64):
    """
    Return (predicted_label, confidence) via softmax probabilities, computed
    in mini-batches with per-batch padding to keep memory flat.
    """
    model = trainer.model
    model.eval()
    texts = list(texts)
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            truncation=True, padding=True, max_length=config.MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(model.device)
        probs = torch.softmax(model(**enc).logits, dim=1)
        confs.append(probs.max(dim=1).values.detach().cpu().numpy())
        preds.append(probs.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)
