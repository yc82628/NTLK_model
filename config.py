"""
config.py — All pipeline settings in one place.

Edit values here rather than hunting through other files.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
REVIEWS_PATH = os.path.join(DATA_DIR, "steam_reviews.csv")
GAMES_PATH   = os.path.join(DATA_DIR, "steam.csv")

OUT_DIR   = "./results"
FIG_DIR   = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")

# ── Column names in the raw data ─────────────────────────────────────────
TEXT_COL  = "review"
LABEL_COL = "recommended"
GENRE_COL = "genre"
GAME_COL  = "app_name"

# ── Sampling ───────────────────────────────────────────────────────────
SAMPLE_SIZE    = 50_000
MIN_REVIEW_LEN = 15          # tokens; filters out low-signal short reviews
RANDOM_STATE   = 42

# ── Memory-safe loading (the reviews CSV is ~8 GB / ~48M rows) ────────────
READ_CHUNK   = 200_000       # rows per chunk when streaming the reviews file
POOL_CAP     = 400_000       # max rows held in memory before the final sample
# Keep only English reviews. VADER and distilbert-base-uncased are English
# models, so non-English text adds noise (your low VADER F1 is a symptom).
# Recommended: True. Left False to preserve your original all-language results
# exactly — flip to True and re-run if you want the cleaner English-only corpus.
ENGLISH_ONLY = False

# ── DistilBERT ─────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
BATCH_SIZE    = 16           # reduce to 8 or 4 if you hit CUDA OOM errors
EPOCHS        = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LEN   = 512          # drop to 256 to roughly halve train/predict time
                             # (most Steam reviews are well under 256 tokens)

# ── LDA ────────────────────────────────────────────────────────────────
LDA_MIN_K        = 5
LDA_MAX_K        = 20
LDA_MAX_FEATURES = 5_000

# ── BERTopic ───────────────────────────────────────────────────────────
BERT_NGRAM        = (1, 2)
BERT_TOP_WORDS    = 15
BERTOPIC_N_RUNS   = 3         # repeated runs for stability check
BERTOPIC_RUN_SIZE = 10_000    # subsample size per stability run
EMBED_MODEL       = "all-MiniLM-L6-v2"  # embeddings, computed once and reused

# ── Divergence analysis ────────────────────────────────────────────────
DIVERGENCE_CONFIDENCE_THRESHOLD = 0.70

# ── Gaming-specific terms protected from stop-word removal ──────────────
GAMING_TERMS = {
    "not", "no", "never", "bug", "crash", "lag", "fps",
    "dlc", "patch", "mod", "pvp", "pve", "loot", "grind",
}


def ensure_dirs():
    """Create all output directories if they don't already exist."""
    for d in (DATA_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)
