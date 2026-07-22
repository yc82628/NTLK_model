"""
topic_modelling.py — LDA (with automatic K-selection via coherence)
and BERTopic (with a multi-seed stability check).
"""

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
 
import config
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
 
def _coherence_cv(top_words: dict, clean_texts: list) -> float:
    """
    Compute c_v coherence for a set of BERTopic topics, guarding against words
    that are absent from the Stream A dictionary (which otherwise crashes gensim).
    """
    gensim_dict = Dictionary(clean_texts)
    vocab = set(gensim_dict.token2id)
 
    words_list = [
        [w for w in top_words[t] if w in vocab]
        for t in sorted(top_words)
    ]
    # drop near-empty topics so coherence is computed on meaningful word sets
    words_list = [ws for ws in words_list if len(ws) >= 2]
 
    if not words_list:
        return 0.0
    return CoherenceModel(
        topics=words_list, texts=clean_texts,
        dictionary=gensim_dict, coherence="c_v",
    ).get_coherence()
 
 
def run_lda(df: pd.DataFrame):
    """
    Fit LDA on Stream A text. Automatically selects K by testing
    LDA_MIN_K through LDA_MAX_K and picking the highest C_v coherence.
    """
    print("Vectorising for LDA...")
    vec = CountVectorizer(
        max_df=0.95, min_df=5, max_features=config.LDA_MAX_FEATURES
    )
    dtm = vec.fit_transform(df["clean_text"])
    vocab = vec.get_feature_names_out()
 
    texts = [t.split() for t in df["clean_text"]]
    gensim_dict = Dictionary(texts)
 
    best_score, best_k, best_model = -1.0, config.LDA_MIN_K, None
    best_words, best_dist = None, None
    scores = {}
 
    for k in range(config.LDA_MIN_K, config.LDA_MAX_K + 1):
        lda = LatentDirichletAllocation(
            n_components=k, random_state=config.RANDOM_STATE, n_jobs=-1
        )
        dist = lda.fit_transform(dtm)
        topic_words = [
            [vocab[i] for i in comp.argsort()[:-11:-1]]
            for comp in lda.components_
        ]
        cm = CoherenceModel(
            topics=topic_words, texts=texts, dictionary=gensim_dict, coherence="c_v"
        )
        score = cm.get_coherence()
        scores[k] = score
        print(f"  K={k:2d}  C_v={score:.4f}")
 
        if score > best_score:
            best_score, best_k, best_model = score, k, lda
            best_words, best_dist = topic_words, dist
 
    df["lda_topic"] = best_dist.argmax(axis=1)
    print(f"[LDA] Best K={best_k}  C_v={best_score:.4f}")
    return best_model, vec, best_k, best_score, best_words, scores
 
 
def _run_bertopic_once(docs, embeddings, clean_texts, seed):
    """Single BERTopic run using precomputed embeddings for a given seed."""
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=50, metric="euclidean", prediction_data=True
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        n_gram_range=config.BERT_NGRAM,
        top_n_words=config.BERT_TOP_WORDS,
        calculate_probabilities=False,   # [MEM] the big BERTopic time/memory sink
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
 
    topic_info = topic_model.get_topic_info()
    top_words = {
        row["Topic"]: [w for w, _ in topic_model.get_topic(row["Topic"])[:10]]
        for _, row in topic_info.iterrows()
        if row["Topic"] != -1
    }
 
    coherence = _coherence_cv(top_words, clean_texts)
    outlier_pct = (np.array(topics) == -1).mean() * 100
    # confidence = HDBSCAN membership strength (cheap, replaces the dense matrix)
    strength = getattr(topic_model.hdbscan_model, "probabilities_", None)
    return topic_model, topics, strength, coherence, top_words, outlier_pct
 
 
def run_bertopic_with_stability_check(df: pd.DataFrame):
    """
    Embed the corpus once, run BERTopic on several seeded subsamples to report
    coherence stability (mean ± std), then one final run on the full corpus.
    """
    print(f"[BERTopic] Embedding corpus once with {config.EMBED_MODEL}...")
    embedder = SentenceTransformer(config.EMBED_MODEL, device=DEVICE)
    docs_all = df["bert_text"].tolist()
    clean_all = [t.split() for t in df["clean_text"]]
    all_emb = embedder.encode(
        docs_all, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
 
    print(f"[BERTopic] Running {config.BERTOPIC_N_RUNS}x stability check...")
    sub_size = min(config.BERTOPIC_RUN_SIZE, len(df))
    run_scores = []
    for i in range(config.BERTOPIC_N_RUNS):
        seed = config.RANDOM_STATE + i
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df), size=sub_size, replace=False)
        sub_docs = [docs_all[j] for j in idx]
        sub_clean = [clean_all[j] for j in idx]
        _, _, _, coh, _, outlier_pct = _run_bertopic_once(
            sub_docs, all_emb[idx], sub_clean, seed
        )
        run_scores.append(coh)
        print(f"  Run {i+1} (seed={seed}): C_v={coh:.4f}, outliers={outlier_pct:.1f}%")
 
    print(f"[BERTopic] Stability — mean={np.mean(run_scores):.4f}  "
          f"std={np.std(run_scores):.4f}")
 
    print("[BERTopic] Running final full-corpus model...")
    topic_model, topics, strength, coherence, top_words, outlier_pct = _run_bertopic_once(
        docs_all, all_emb, clean_all, config.RANDOM_STATE
    )
    df["bertopic_topic"] = topics
    df["bertopic_confidence"] = strength if strength is not None else 0.0
    print(f"[BERTopic] Final: {len(top_words)} topics, C_v={coherence:.4f}, "
          f"{outlier_pct:.1f}% outliers")
 
    return topic_model, coherence, top_words, run_scores

