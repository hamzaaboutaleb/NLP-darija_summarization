"""
Approach 1 — NLP Extractive Summarization (TF-IDF + Sentence Scoring)
=======================================================================
Strategy:
  - Tokenize Darija text into sentences
  - Build a TF-IDF matrix over sentences
  - Score each sentence by the sum of its TF-IDF weights
  - Return the top-N highest-scoring sentences as the summary

Why extractive?
  Extractive methods work well with Darija because they do NOT require
  the model to generate new text (which would need a Darija language model).
  They simply select the most informative sentences already in the document.

Dependencies:
  pip install nltk scikit-learn pandas
"""

import re
import string
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# ---------------------------------------------------------------------------
# Darija-aware helpers
# ---------------------------------------------------------------------------

# A minimal set of common Darija / Moroccan Arabic stop-words (Latin script)
# Extend this list with tokens from your own dataset as needed.
DARIJA_STOPWORDS = {
    "f", "fi", "w", "wa", "had", "had", "li", "lli", "ta", "ila",
    "mn", "min", "3la", "3nd", "b", "bi", "ma", "la", "kan", "kayn",
    "dyal", "dial", "ntuma", "nta", "ana", "hna", "huma", "wla",
    "walakin", "aw", "aw", "m3a", "m3", "hit", "ash", "ach", "kifach",
    "bzzaf", "shwiya", "ghir", "bas", "mashi", "wach", "hal",
}


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    NLTK's sent_tokenize works reasonably well for Latin-script Darija.
    For Arabic-script Darija add a rule-based splitter on '.' '!' '؟'.
    """
    sentences = sent_tokenize(text)
    # Fallback: split on period if tokenizer returns only one chunk
    if len(sentences) <= 1:
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# TF-IDF Extractor
# ---------------------------------------------------------------------------

class TFIDFSummarizer:
    """
    Extractive summarizer using TF-IDF sentence scoring.

    Parameters
    ----------
    num_sentences : int
        Number of sentences to include in the summary.
    max_features : int
        Vocabulary size cap for the TF-IDF vectorizer.
    """

    def __init__(self, num_sentences: int = 3, max_features: int = 500):
        self.num_sentences = num_sentences
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(DARIJA_STOPWORDS),
            ngram_range=(1, 2),
        )

    def summarize(self, text: str) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return text  # Already short enough

        cleaned = [clean_text(s) for s in sentences]

        # Build TF-IDF matrix  (sentences × features)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(cleaned)
        except ValueError:
            # Vocabulary too small → return first N sentences
            return " ".join(sentences[: self.num_sentences])

        # Score = sum of TF-IDF weights per sentence
        scores = tfidf_matrix.sum(axis=1).A1  # shape (n_sentences,)

        # Pick top-N indices, preserve original order
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                : self.num_sentences
            ]
        )
        return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str, text_column: str = "text") -> pd.DataFrame:
    """
    Load the Darija dataset from a CSV file.

    Expected CSV columns
    --------------------
    text     : raw Darija text to summarize
    summary  : (optional) reference summary for evaluation
    """
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available: {list(df.columns)}"
        )
    df = df.dropna(subset=[text_column])
    print(f"Loaded {len(df)} samples from '{csv_path}'.")
    return df


# ---------------------------------------------------------------------------
# Evaluation helpers (ROUGE-like)
# ---------------------------------------------------------------------------

def simple_overlap_score(predicted: str, reference: str) -> float:
    """
    Token-level F1 overlap (a lightweight alternative to rouge when the
    full rouge library is unavailable).
    """
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


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    CSV_PATH = "dataset.csv"          # ← change to your CSV path
    TEXT_COL = "text"                 # ← column with Darija text
    SUMMARY_COL = "summary"          # ← column with reference summary (optional)
    NUM_SENTENCES = 3

    # ---- Load data ----
    df = load_dataset(CSV_PATH, text_column=TEXT_COL)

    # ---- Build summarizer ----
    summarizer = TFIDFSummarizer(num_sentences=NUM_SENTENCES)

    # ---- Run on first 5 samples ----
    results = []
    for idx, row in df.head(5).iterrows():
        original = row[TEXT_COL]
        predicted = summarizer.summarize(original)
        entry = {"original": original, "predicted_summary": predicted}

        if SUMMARY_COL in df.columns and pd.notna(row.get(SUMMARY_COL)):
            score = simple_overlap_score(predicted, row[SUMMARY_COL])
            entry["overlap_f1"] = round(score, 4)

        results.append(entry)
        print(f"\n--- Sample {idx} ---")
        print(f"ORIGINAL  : {original[:120]}...")
        print(f"SUMMARY   : {predicted}")
        if "overlap_f1" in entry:
            print(f"OVERLAP F1: {entry['overlap_f1']}")

    # ---- Save results ----
    out_df = pd.DataFrame(results)
    out_df.to_csv("results_approach1.csv", index=False)
    print("\nResults saved to results_approach1.csv")
