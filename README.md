# Steam NLP Pipeline — Local VS Code Setup

## What changed in this (improved) version
- **Memory-safe data loading.** `preprocessing.py` now streams the ~8 GB reviews
  CSV in chunks and caps the working set at `POOL_CAP` rows, instead of loading
  the whole file at once (the original out-of-memory crash).
- **Fixed a silent VADER bug** in `main.py` that evaluated VADER against the
  wrong rows (a mislabelled `df.loc[...]` index lookup). VADER metrics are now
  correct.
- **DistilBERT** uses dynamic per-batch padding + lazy tokenisation + fp16 on
  GPU, and predicts in mini-batches — far less RAM/VRAM than padding everything
  to 512 up front.
- **BERTopic** runs with `calculate_probabilities=False` (the main slow/OOM
  culprit) and reuses a single set of sentence embeddings across all runs.
- **Version pin:** `transformers>=4.46` (earlier versions lack `eval_strategy`
  and `processing_class`).
- **New optional knob:** `ENGLISH_ONLY` in `config.py` (default `False`).
  Setting it `True` keeps only English reviews — recommended, since VADER and
  distilbert-uncased are English models. It changes your sample composition, so
  note it in your methodology if you enable it.

## Prerequisites
- NVIDIA GPU with drivers installed (verify: `nvidia-smi` in terminal)
- Python 3.10 or 3.11 (3.12 can have compatibility issues with some ML libraries)
- VS Code with the Python extension installed
- A Kaggle account (free) for dataset access

---

## Step 1 — Create a virtual environment

Open a terminal in VS Code (`` Ctrl+` `` / `` Cmd+` ``) in your project folder:

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

In VS Code, press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `./venv`.

---

## Step 2 — Install PyTorch with CUDA support (do this BEFORE requirements.txt)

Check your CUDA version first:
```bash
nvidia-smi
```
Look at the top-right of the output for "CUDA Version: XX.X".

Go to https://pytorch.org/get-started/locally/ and it will generate the exact command for you.
Example for CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Verify it worked before continuing:**
```bash
python check_gpu.py
```
This should print your GPU name and confirm CUDA is available. Do not proceed until this passes — if PyTorch can't see your GPU, everything downstream will silently run on CPU and be extremely slow.

---

## Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Set up Kaggle API credentials

1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" under the API section — downloads `kaggle.json`
3. Place it in the correct location for your OS:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`
4. On macOS/Linux, restrict its permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Step 5 — Download the datasets

```bash
python download_data.py
```
This pulls both Kaggle datasets into `./data/` automatically.

---

## Step 6 — Run the pipeline

```bash
python main.py
```

All output (figures, tables, models, annotated CSV) will be saved to `./results/`.

---

## Project Structure

```
steam_project/
├── venv/                      # virtual environment (not committed to git)
├── data/                      # downloaded Kaggle datasets
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
├── check_gpu.py                # run first — verifies GPU setup
├── download_data.py            # pulls datasets from Kaggle API
├── config.py                   # all settings in one place
├── preprocessing.py            # Stream A / Stream B text cleaning
├── sentiment.py                # VADER + DistilBERT
├── topic_modelling.py          # LDA + BERTopic
├── divergence.py               # sentiment-label divergence analysis
├── bot_detection.py            # adversarial/bot review detection
├── evaluation.py               # metrics, plots, confusion matrices
├── main.py                     # orchestrates the full pipeline
└── requirements.txt
```

---

## Troubleshooting

**CUDA out of memory during DistilBERT fine-tuning:**
Open `config.py` and reduce `BATCH_SIZE` from 16 to 8 or 4.

**`ModuleNotFoundError` for any package:**
Make sure your venv is activated (you should see `(venv)` in your terminal prompt) and re-run `pip install -r requirements.txt`.

**Kaggle API "403 Forbidden" error:**
You likely haven't accepted the dataset's terms on the Kaggle website. Visit the dataset page in your browser once and click "Download" — this registers your acceptance, then the API will work.

**Training is still slow despite having a GPU:**
Run `python check_gpu.py` again. If VRAM is under 4GB, reduce `BATCH_SIZE` in `config.py`.

**Training is extremely slow (hours) / progress bar shows a huge ETA:**
You are almost certainly on CPU. Check that `check_gpu.py` prints `CUDA available: True`. On CPU, either run on a Colab/Kaggle T4 GPU, or set `MAX_SEQ_LEN = 256` and `EPOCHS = 2` in `config.py` to make a CPU run tolerable.

**Runs out of memory while loading data:**
Lower `POOL_CAP` (e.g. 200_000) and/or `SAMPLE_SIZE` in `config.py`. The reader is chunked, so the pool cap is what bounds peak RAM.

**BERTopic seems frozen on the final run:**
Make sure `calculate_probabilities` is `False` in `topic_modelling.py` (it is, in this version). Reduce `BERTOPIC_RUN_SIZE` if the stability runs are slow.
