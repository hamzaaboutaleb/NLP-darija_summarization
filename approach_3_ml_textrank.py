"""
Approach 3 — ML Classical Summarization (TextRank + scikit-learn)
==================================================================
Strategy:
  - Represent each sentence as a TF-IDF vector
  - Build a cosine-similarity graph between sentences
  - Run PageRank (TextRank algorithm) on the graph to score sentences
  - Return the highest-ranked sentences as the summary

Why TextRank?
  TextRank is unsupervised (no labels needed), language-agnostic, and
  produces coherent extractive summaries by modeling sentence importance
  through graph centrality rather than raw term frequency alone.

Dependencies:
  pip install scikit-learn networkx pandas numpy nltk
"""

import re
import string
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
TEXT_COL = "text"
SUMMARY_COL = "summary"   # optional — used for evaluation only
NUM_SENTENCES = 3         # sentences to include in summary
DAMPING = 0.85            # PageRank damping factor
MAX_ITER = 100            # PageRank iterations


# ---------------------------------------------------------------------------
# Darija stop-words (Latin script) — extend from your data
# ---------------------------------------------------------------------------

DARIJA_STOPWORDS = [
    "f", "fi", "w", "wa", "had", "li", "lli", "ta", "ila",
    "mn", "min", "3la", "3nd", "b", "bi", "ma", "la", "kan", "kayn",
    "dyal", "dial", "ntuma", "nta", "ana", "hna", "huma", "wla",
    "walakin", "aw", "m3a", "m3", "hit", "ash", "ach", "kifach",
    "bzzaf", "shwiya", "ghir", "bas", "mashi", "wach", "hal",
]


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_sentence(sentence: str) -> str:
    """Lowercase and remove punctuation for vectorization."""
    sentence = sentence.lower()
    sentence = re.sub(r"[" + re.escape(string.punctuation) + "]+", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK with a period-split fallback."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        sentences = [s.strip() for s in re.split(r"[.!?؟]", text) if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def build_similarity_matrix(sentences: list[str]) -> np.ndarray:
    """
    Compute pairwise cosine similarity between TF-IDF sentence vectors.

    Returns
    -------
    np.ndarray of shape (n, n)
    """
    cleaned = [clean_sentence(s) for s in sentences]

    vectorizer = TfidfVectorizer(
        stop_words=DARIJA_STOPWORDS,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        # Fallback: identity matrix (all sentences equally important)
        n = len(sentences)
        return np.eye(n)

    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Zero out self-similarity to avoid self-loops in the graph
    np.fill_diagonal(sim_matrix, 0.0)
    return sim_matrix


# ---------------------------------------------------------------------------
# TextRank summarizer
# ---------------------------------------------------------------------------

class TextRankSummarizer:
    """
    Graph-based extractive summarizer using the TextRank algorithm.

    Parameters
    ----------
    num_sentences : int
        Number of top-ranked sentences to include in the summary.
    damping : float
        PageRank damping factor (typically 0.85).
    max_iter : int
        Maximum PageRank iterations.
    """

    def __init__(
        self,
        num_sentences: int = NUM_SENTENCES,
        damping: float = DAMPING,
        max_iter: int = MAX_ITER,
    ):
        self.num_sentences = num_sentences
        self.damping = damping
        self.max_iter = max_iter

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)

        if len(sentences) <= self.num_sentences:
            return text  # Already short enough

        # Build similarity matrix
        sim_matrix = build_similarity_matrix(sentences)

        # Build weighted directed graph
        graph = nx.from_numpy_array(sim_matrix)

        # Compute PageRank scores
        try:
            scores = nx.pagerank(
                graph,
                alpha=self.damping,
                max_iter=self.max_iter,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality if PageRank doesn't converge
            scores = nx.degree_centrality(graph)

        # Select top sentences (preserve original order)
        ranked = sorted(scores, key=scores.get, reverse=True)
        top_indices = sorted(ranked[: self.num_sentences])
        return " ".join(sentences[i] for i in top_indices)

    def summarize_with_scores(self, text: str) -> tuple[str, dict]:
        """Returns (summary_text, {sentence: score}) for inspection."""
        sentences = split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return text, {}

        sim_matrix = build_similarity_matrix(sentences)
        graph = nx.from_numpy_array(sim_matrix)
        try:
            scores = nx.pagerank(graph, alpha=self.damping, max_iter=self.max_iter)
        except nx.PowerIterationFailedConvergence:
            scores = nx.degree_centrality(graph)

        ranked = sorted(scores, key=scores.get, reverse=True)
        top_indices = sorted(ranked[: self.num_sentences])
        summary = " ".join(sentences[i] for i in top_indices)
        sentence_scores = {sentences[i]: round(scores[i], 6) for i in range(len(sentences))}
        return summary, sentence_scores


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=[TEXT_COL])
    print(f"Loaded {len(df)} samples from '{csv_path}'.")
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def token_f1(predicted: str, reference: str) -> float:
    """Token-level F1 overlap between predicted and reference summary."""
    pred_tokens = set(predicted.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    intersection = pred_tokens & ref_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def batch_evaluate(df: pd.DataFrame, summarizer: TextRankSummarizer) -> pd.DataFrame:
    """Run summarizer on the entire DataFrame and return results."""
    records = []
    for _, row in df.iterrows():
        text = row[TEXT_COL]
        predicted = summarizer.summarize(text)
        entry = {"original": text, "predicted_summary": predicted}
        if SUMMARY_COL in df.columns and pd.notna(row.get(SUMMARY_COL)):
            entry["reference_summary"] = row[SUMMARY_COL]
            entry["token_f1"] = round(token_f1(predicted, row[SUMMARY_COL]), 4)
        records.append(entry)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Load data ----
    df = load_dataset(CSV_PATH)

    # ---- Build summarizer ----
    summarizer = TextRankSummarizer(num_sentences=NUM_SENTENCES)

    # ---- Demo on first 5 samples with score inspection ----
    for idx, row in df.head(5).iterrows():
        text = row[TEXT_COL]
        summary, scores = summarizer.summarize_with_scores(text)
        print(f"\n=== Sample {idx} ===")
        print(f"ORIGINAL  ({len(split_sentences(text))} sentences):")
        print(f"  {text[:200]}…")
        print(f"SUMMARY   ({NUM_SENTENCES} sentences):")
        print(f"  {summary}")
        if SUMMARY_COL in df.columns and pd.notna(row.get(SUMMARY_COL)):
            f1 = token_f1(summary, row[SUMMARY_COL])
            print(f"TOKEN F1  : {f1:.4f}")
        # Print top sentence scores
        top_scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("TOP PAGERANK SCORES:")
        for sent, score in top_scored:
            print(f"  [{score:.5f}] {sent[:80]}…")

    # ---- Batch evaluation ----
    print("\nRunning batch evaluation…")
    results_df = batch_evaluate(df, summarizer)
    results_df.to_csv("results_approach3.csv", index=False)
    print(f"Saved {len(results_df)} results to results_approach3.csv")

    # ---- Print average F1 if available ----
    if "token_f1" in results_df.columns:
        avg_f1 = results_df["token_f1"].mean()
        print(f"Average Token F1: {avg_f1:.4f}")
