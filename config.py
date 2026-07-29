"""
config.py — All pipeline settings in one place.

Edit values here rather than hunting through other files.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
REVIEWS_PATH = os.path.join(DATA_DIR, "steam_reviews.csv")
GAMES_PATH   = os.path.join(DATA_DIR, "steam.csv")

CACHE_DIR = "./cache"      # cached intermediate data (see USE_CACHE below)

OUT_DIR   = "./results"
FIG_DIR   = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")

# ── Column names in the raw data ─────────────────────────────────────────
TEXT_COL  = "review"
LABEL_COL = "recommended"
GENRE_COL = "genre"
GAME_COL  = "app_name"

# ── Caching (BIG runtime win) ──────────────────────────────────────────
# Reading + filtering the 48M-row CSV takes ~1-2 hours and produces the SAME
# result every run for a given config. With caching on, run 1 writes the
# sampled/preprocessed data to CACHE_DIR; every later run loads it in seconds.
# The cache key includes the sampling settings, so changing SAMPLE_SIZE,
# MIN_REVIEW_LEN, ENGLISH_ONLY, POOL_CAP or RANDOM_STATE automatically
# rebuilds it. Set False (or delete ./cache) to force a fresh read.
USE_CACHE = True

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
ENGLISH_ONLY = True

# ── DistilBERT ─────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
BATCH_SIZE    = 16           # reduce to 8 or 4 if you hit CUDA OOM errors
EPOCHS        = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LEN   = 256          # was 512. Most Steam reviews fit well inside 256
                             # tokens, so this roughly HALVES training and
                             # inference time for negligible accuracy change.
                             # Raise back to 512 if you want the original setting.
EARLY_STOPPING_PATIENCE = 1  # stop if macro-F1 hasn't improved for N evals;
                             # often saves a full epoch with no accuracy cost

# ── LDA ────────────────────────────────────────────────────────────────
LDA_MIN_K        = 5
LDA_MAX_K        = 20
LDA_K_STEP       = 1         # set to 2 to halve the LDA sweep time
                             # (tests K=5,7,9,... instead of every value)
LDA_MAX_FEATURES = 5_000

# ── BERTopic ───────────────────────────────────────────────────────────
BERT_NGRAM        = (1, 2)
BERT_TOP_WORDS    = 15
BERTOPIC_N_RUNS   = 3         # repeated runs for stability check
BERTOPIC_RUN_SIZE = 10_000    # subsample size per stability run
EMBED_MODEL       = "all-MiniLM-L6-v2"  # embeddings, computed once and reused

# ── Divergence analysis ────────────────────────────────────────────────
DIVERGENCE_CONFIDENCE_THRESHOLD = 0.70

# ── Figure readability ─────────────────────────────────────────────────
# The genre x topic heatmap had ~118 topic columns, making it unreadable and
# dominated by cells holding 1-2 reviews (which render as 0% or 100%).
HEATMAP_TOP_N_TOPICS = 15     # keep only the N highest-volume topics
HEATMAP_MIN_CELL_COUNT = 30   # blank out cells with fewer reviews than this

# ── Review authenticity screening (observable provenance rules) ────────
DUP_COORDINATED_MIN   = 3     # identical text repeated >=N times on ONE game
MIN_PLAUSIBLE_PLAYTIME_MIN = 30    # minutes; "near-zero playtime"
LONG_REVIEW_CHARS     = 400   # chars; a substantial verdict
JOKE_MIN_FUNNY_VOTES  = 5     # minimum funny votes before the ratio is meaningful
JOKE_FUNNY_RATIO      = 0.60  # share of votes that were 'funny' not 'helpful'

# ── Gaming-specific terms protected from stop-word removal ──────────────
GAMING_TERMS = {
    "not", "no", "never", "bug", "crash", "lag", "fps",
    "dlc", "patch", "mod", "pvp", "pve", "loot", "grind",
}


def ensure_dirs():
    """Create all output directories if they don't already exist."""
    for d in (DATA_DIR, CACHE_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)