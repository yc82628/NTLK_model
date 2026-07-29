# Steam NLP Pipeline — Local VS Code Setup

A pipeline comparing lexicon-based (VADER) and transformer-based (DistilBERT)
sentiment against Steam's binary recommendation label, with LDA and BERTopic
topic modelling, genre/topic divergence analysis, and review authenticity
screening.

---

## Quick start (already set up?)

```bash
venv\Scripts\activate        # Windows
python main.py
```
First run reads the full 8 GB CSV (~1–2 hours). **Later runs load from cache in
seconds** — see *Caching* below.

---

## What changed from the original version

**Crash and correctness fixes**
- **Memory-safe loading.** `preprocessing.py` streams the ~8 GB / 48M-row CSV in
  chunks and caps the working set at `POOL_CAP`, instead of loading it all at
  once (the original out-of-memory crash).
- **Fixed a silent VADER bug** in `main.py` that evaluated VADER against the
  wrong rows (a `df.loc[...]` lookup against a reset index). All VADER metrics
  are now correct.
- **Fixed the genre sampler**, which could request a larger sample than a genre
  contained (`ValueError: Cannot take a larger sample than population`).
- **Fixed a gensim crash** in BERTopic coherence — topic words are filtered to
  the Stream A dictionary before scoring (`unable to interpret topic...`).
- **Version pin:** `transformers>=4.46`. Earlier versions lack `eval_strategy`
  and `processing_class` and will raise `TypeError` during training setup.

**Speed**
- **Caching** of the sampled and preprocessed data (the single biggest win).
- `MAX_SEQ_LEN` 512 → **256**, roughly halving training and inference time.
- **fp16** mixed precision on GPU; **early stopping** when macro-F1 plateaus.
- BERTopic embeds the corpus **once** and reuses it across all runs.
- `calculate_probabilities=False` in BERTopic (was the main slow/OOM culprit).

**New analysis**
- **ROC-AUC and Precision-Recall curves** comparing VADER and DistilBERT
  without depending on any chosen cut-off.
- **Review authenticity screening** based on observable provenance
  (duplicate text, playtime mismatch, free copies) — see below.
- **Readable figures** with plain-English axis labels for non-technical readers.

**Config change to be aware of**
- `ENGLISH_ONLY` now defaults to **`True`** (was `False`). VADER and
  `distilbert-base-uncased` are English models, and the proposal describes an
  English corpus. This changes sample composition — state it in your
  methodology.

---

## Caching (important)

Reading and filtering 48M rows takes 1–2 hours and produces **the same result
every time** for a given config. So it is cached.

- Run 1: reads the CSV, writes `./cache/sample_<key>.pkl` and
  `./cache/preprocessed_<key>.pkl`
- Run 2+: loads those in seconds (look for `[CACHE] Loaded ...`)

The cache key is a hash of `SAMPLE_SIZE`, `MIN_REVIEW_LEN`, `ENGLISH_ONLY`,
`POOL_CAP` and `RANDOM_STATE`. **Change any of them and the cache rebuilds
automatically** — a stale cache cannot silently feed wrong data into results.

Force a fresh read by deleting `./cache/` or setting `USE_CACHE = False`.

---

## Prerequisites
- NVIDIA GPU with drivers installed (verify: `nvidia-smi`). CPU works but
  DistilBERT training takes **hours instead of minutes**.
- Python 3.10 or 3.11 (3.12 can break some ML libraries)
- VS Code with the Python extension
- A free Kaggle account
- ~15 GB free disk (3 GB download + 8 GB extracted + cache + model checkpoint)

---

## Step 1 — Create a virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → the one in `./venv`.

> Keep the venv **inside the project folder**. A venv on a system path can lead
> to editing one copy of a file while Python imports another.

---

## Step 2 — Install PyTorch with CUDA FIRST (before requirements.txt)

Check your CUDA version (top-right of `nvidia-smi` output), then get the exact
command from https://pytorch.org/get-started/locally/. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Verify before continuing:**
```bash
python check_gpu.py
```
Do not proceed until this prints `CUDA available: True`. If PyTorch can't see
the GPU, everything runs on CPU and is extremely slow.

> Never let a later `pip install` upgrade `torch` — it can silently replace your
> CUDA build with the CPU build. Re-run `check_gpu.py` if you install anything.

---

## Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```
Requires `transformers>=4.46` and `sentence-transformers`.

---

## Step 4 — Kaggle API credentials

1. https://www.kaggle.com/settings → "Create New Token" → downloads `kaggle.json`
2. Place it:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json` (then `chmod 600`)

---

## Step 5 — Download the datasets

```bash
python download_data.py
```
Pulls both Kaggle datasets into `./data/`.

---

## Step 6 — Run the pipeline

```bash
python main.py
```
All output goes to `./results/`.

**Rough timings** (RTX-class GPU, 50k sample): first run 1–2 h (dominated by the
CSV read), later runs ~20–40 min (DistilBERT + BERTopic). On CPU, add hours.

---

## Outputs

`results/figures/` — 11 figures
| File | Shows |
|---|---|
| `vader_threshold_tuning.png` | VADER cut-off sweep vs agreement with thumbs |
| `vader_cm.png`, `distilbert_cm.png` | Confusion matrices (counts + row %) |
| `roc_curves.png` | ROC + AUC, VADER vs DistilBERT (cut-off independent) |
| `precision_recall_curves.png` | PR curves + naive baseline (82% positive) |
| `lda_k_selection.png` | Coherence vs number of topics K |
| `coherence_comparison.png` | LDA vs BERTopic coherence + stability |
| `divergence_heatmap.png` | Divergence by genre × topic (top topics only) |
| `divergence_by_genre.png` | Divergence per genre with sample sizes |
| `authenticity_flags.png` | Reviews per provenance flag |
| `bot_detection_features.png` | Feature distributions, flagged vs typical |

`results/tables/` — 8 CSVs including `sentiment_metrics.csv`,
`ranking_metrics.csv` (ROC-AUC / average precision), `divergence_by_genre.csv`,
`authenticity_flags.csv`, `lda_top_words.csv`, `bertopic_top_words.csv`.

`results/annotated_reviews.csv` — every review with all model outputs, topic
assignments, divergence flags and authenticity flags. **This is the raw evidence
for any claim in Chapter 4.**

Re-running overwrites all of these under the same filenames. If a run crashes
partway, some files will be from this run and some from the last — rename
`results/` first if you want that guaranteed impossible.

---

## Review authenticity screening — read this before writing it up

This module does **not** identify bots. The dataset has no ground-truth labels
for inauthentic reviews, so nothing here is a validated bot classifier. It
reports two clearly separated things:

1. **Provenance flags** (rule-based, observable facts): duplicated text, a long
   review written at near-zero playtime, a free review copy, a review the
   community voted mostly "funny". Each is counted separately in
   `authenticity_flags.csv`.
2. **Behavioural anomaly score** (Isolation Forest). Note that
   `contamination=0.05` flags 5% **by construction** — that is not a discovery
   that 5% are fake. On Steam, "unusual" is often genuine copypasta or ASCII art.

Duplicate detection deliberately separates *copypasta across many games*
(usually a real human meme) from *text repeated on a single game* (the pattern
consistent with manipulation). Conflating them would badly overstate fraud.

**Defensible:** "X% of reviews carried at least one authenticity concern flag."
**Not defensible:** "The model identified X% of reviews as bot-generated."

---

## Key settings in `config.py`

| Setting | Default | Effect |
|---|---|---|
| `USE_CACHE` | `True` | Reuse cached data; the big runtime saver |
| `ENGLISH_ONLY` | `True` | Keep only English reviews |
| `SAMPLE_SIZE` | `50_000` | Reviews analysed, split evenly across genres |
| `POOL_CAP` | `400_000` | Bounds peak RAM during the CSV read |
| `MAX_SEQ_LEN` | `256` | Tokens per review; 512 is slower, ~same accuracy |
| `BATCH_SIZE` | `16` | Lower to 8/4 on CUDA out-of-memory |
| `EPOCHS` | `3` | With early stopping, may finish sooner |
| `LDA_K_STEP` | `1` | Set `2` to halve LDA sweep time |
| `HEATMAP_TOP_N_TOPICS` | `15` | Topics shown in the heatmap |
| `HEATMAP_MIN_CELL_COUNT` | `30` | Blank cells below this many reviews |

---

## Project Structure

```
steam_project/
├── venv/                    # virtual environment
├── data/                    # downloaded Kaggle datasets
├── cache/                   # cached sampled/preprocessed data (safe to delete)
├── results/
│   ├── figures/  tables/  models/
├── check_gpu.py             # run FIRST — verifies GPU setup
├── download_data.py         # pulls datasets from Kaggle API
├── config.py                # all settings in one place
├── preprocessing.py         # chunked loading, caching, Stream A/B cleaning
├── sentiment.py             # VADER + DistilBERT (+ positive-class probs)
├── topic_modelling.py       # LDA + BERTopic (shared embeddings)
├── divergence.py            # sentiment-vs-label divergence analysis
├── bot_detection.py         # authenticity screening (provenance + anomaly)
├── evaluation.py            # metrics, ROC/PR, all figures
├── main.py                  # orchestrates the full pipeline
└── requirements.txt
```

---

## Known issues / still to address

Documented honestly so they aren't discovered late:

- **VADER neutral band collapses.** If threshold tuning selects `0.00`, the
  neutral category becomes zero-width and no review is ever labelled neutral —
  so `vader_neutral` is all zeros and `vader_divergence` reduces to a plain
  mismatch. The methodology specifies a fixed 0.05 band; consider decoupling the
  binary cut-off from the neutral band.
- **Topic-model handoff.** The methodology says the higher-coherence model
  advances to divergence analysis, but the code uses `bertopic_topic`
  regardless. LDA has been scoring higher.
- **Coherence comparison is confounded.** BERTopic's words are scored against
  the Stream A dictionary, which penalises it. Do not claim LDA produces more
  coherent topics without addressing this.
- **Genre list drifts** from the six named in the methodology, because the
  sampler keeps whatever survives the join and size threshold.
- **Divergence is computed on the full corpus**, including reviews DistilBERT
  trained on, so it is slightly optimistic there.

---

## Troubleshooting

**`TypeError: ... unexpected keyword argument 'tokenizer'` / `'evaluation_strategy'`**
Your `transformers` is a version that renamed these. Run
`pip install --upgrade "transformers>=4.46"`. If the error persists *unchanged*
after editing, Python is importing a different copy of the file than the one you
edited — check with `findstr /n "tokenizer=" sentiment.py`.

**Training extremely slow / progress bar shows a huge ETA (hours)**
You are on CPU. Confirm with `python check_gpu.py`. The bar shows
`step/total [elapsed<remaining, it/s]`; on a GPU expect 5–12 it/s, on CPU ~0.03.
Fix the runtime, or set `EPOCHS = 2` and reduce `SAMPLE_SIZE` for a tolerable
CPU run.

**Empty "Epoch / Training Loss" table**
Normal. That table fills in one row per epoch; watch the tqdm bar underneath.

**`MISSING: classifier.weight` in the load report**
Normal and expected. You are attaching a fresh classification head to base
DistilBERT; those weights are meant to be newly initialised and then trained.

**CUDA out of memory during fine-tuning**
Reduce `BATCH_SIZE` to 8 or 4 in `config.py`.

**Out of memory while loading data**
Lower `POOL_CAP` (e.g. `200_000`) and/or `SAMPLE_SIZE`. The reader is chunked,
so `POOL_CAP` is what bounds peak RAM.

**`ValueError: Cannot take a larger sample than population`**
Fixed in this version. If it recurs, lower `SAMPLE_SIZE` or raise `POOL_CAP`.

**`ValueError: unable to interpret topic as either a list of tokens or ids`**
Fixed in this version (vocab filter in `topic_modelling.py`). If it recurs, the
file on disk is an older copy.

**BERTopic seems frozen**
Confirm `calculate_probabilities=False` in `topic_modelling.py`. The
`Embedding corpus once...` step can sit for several minutes on CPU before the
progress bar appears. Reduce `BERTOPIC_RUN_SIZE` if stability runs are slow.

**Kaggle API "403 Forbidden"**
Visit the dataset page in a browser once and click "Download" to accept the
terms, then the API will work.

**`ModuleNotFoundError`**
Confirm the venv is active (`(venv)` in your prompt), then re-run
`pip install -r requirements.txt`.
