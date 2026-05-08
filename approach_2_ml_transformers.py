"""
Approach 2 — ML Abstractive Summarization (HuggingFace Transformers)
======================================================================
Strategy:
  - Use a pre-trained multilingual seq2seq model (mT5 or mBART)
    that supports Arabic / Darija script
  - Fine-tune on the Darija dataset with a standard seq2seq objective
  - At inference, generate new summary text (abstractive)

Why this approach?
  Transformer-based abstractive models can paraphrase and compress text
  meaningfully. Multilingual checkpoints already have some knowledge of
  Arabic morphology, which transfers to Darija.

Recommended base models (pick one):
  - "csebuetnlp/mT5_multilingual_XLSum"  (fine-tuned on news in 45 langs)
  - "facebook/mbart-large-cc25"           (multilingual BART)
  - "Helsinki-NLP/opus-mt-ar-en"         (Arabic → English, for inspection)

Dependencies:
  pip install transformers datasets torch sentencepiece pandas rouge-score
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline,
)
from datasets import Dataset as HFDataset

# ---------------------------------------------------------------------------
# Config — edit these to match your setup
# ---------------------------------------------------------------------------

BASE_MODEL = "csebuetnlp/mT5_multilingual_XLSum"  # or swap to mBART
CSV_PATH = "dataset.csv"
TEXT_COL = "text"
SUMMARY_COL = "summary"       # reference summary column
OUTPUT_DIR = "./model_approach2"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# mT5 uses a language code prefix — set to Arabic
LANG_PREFIX = "ar"            # change to "ary" if your model supports Moroccan Arabic code


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=[TEXT_COL, SUMMARY_COL])
    print(f"Loaded {len(df)} samples.")
    return df


def dataframe_to_hf_dataset(df: pd.DataFrame) -> dict:
    """Split into train/validation and convert to HuggingFace Dataset format."""
    split = int(0.9 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)
    return {
        "train": HFDataset.from_pandas(train_df[[TEXT_COL, SUMMARY_COL]]),
        "validation": HFDataset.from_pandas(val_df[[TEXT_COL, SUMMARY_COL]]),
    }


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def preprocess_function(examples, tokenizer):
    # Prepend language prefix expected by mT5 XLSum
    inputs = [f"{LANG_PREFIX}: " + t for t in examples[TEXT_COL]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=examples[SUMMARY_COL],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    # Replace padding token id in labels with -100 so loss ignores them
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(csv_path: str = CSV_PATH):
    print(f"Using device: {DEVICE}")

    # Load data
    df = load_dataframe(csv_path)
    splits = dataframe_to_hf_dataset(df)

    # Tokenizer & model
    tokenizer = get_tokenizer(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)

    # Tokenize datasets
    tokenized = {
        split: ds.map(
            lambda ex: preprocess_function(ex, tokenizer),
            batched=True,
            remove_columns=ds.column_names,
        )
        for split, ds in splits.items()
    }

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=(DEVICE == "cuda"),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training…")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to '{OUTPUT_DIR}'.")
    return trainer


# ---------------------------------------------------------------------------
# Inference (after training or using the base checkpoint directly)
# ---------------------------------------------------------------------------

def build_summarizer(model_dir: str = OUTPUT_DIR):
    """
    Load a fine-tuned (or base) model and return a summarization pipeline.
    Falls back to BASE_MODEL if the fine-tuned directory doesn't exist yet.
    """
    source = model_dir if os.path.isdir(model_dir) else BASE_MODEL
    print(f"Loading model from: {source}")
    summarizer = pipeline(
        "summarization",
        model=source,
        tokenizer=source,
        device=0 if DEVICE == "cuda" else -1,
    )
    return summarizer


def summarize_text(text: str, summarizer=None) -> str:
    if summarizer is None:
        summarizer = build_summarizer()
    input_text = f"{LANG_PREFIX}: " + text
    result = summarizer(
        input_text,
        max_length=MAX_TARGET_LEN,
        min_length=20,
        do_sample=False,
    )
    return result[0]["summary_text"]


# ---------------------------------------------------------------------------
# Quick ROUGE evaluation
# ---------------------------------------------------------------------------

def evaluate_rouge(predictions: list[str], references: list[str]) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        totals = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for k in totals:
                totals[k] += scores[k].fmeasure
        n = len(predictions)
        return {k: round(v / n, 4) for k, v in totals.items()}
    except ImportError:
        print("Install rouge-score for ROUGE metrics: pip install rouge-score")
        return {}


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Step 1: Train (comment out if already trained) ----
    train(CSV_PATH)

    # ---- Step 2: Load summarizer ----
    summarizer = build_summarizer(OUTPUT_DIR)

    # ---- Step 3: Run on a few samples ----
    df = load_dataframe(CSV_PATH)
    sample = df.head(5)

    predictions, references = [], []
    for idx, row in sample.iterrows():
        pred = summarize_text(row[TEXT_COL], summarizer)
        predictions.append(pred)
        references.append(row[SUMMARY_COL])
        print(f"\n--- Sample {idx} ---")
        print(f"ORIGINAL  : {row[TEXT_COL][:120]}…")
        print(f"PREDICTED : {pred}")
        print(f"REFERENCE : {row[SUMMARY_COL]}")

    # ---- Step 4: ROUGE scores ----
    rouge = evaluate_rouge(predictions, references)
    if rouge:
        print("\nROUGE scores:", rouge)

    # ---- Step 5: Save results ----
    sample = sample.copy()
    sample["predicted_summary"] = predictions
    sample.to_csv("results_approach2.csv", index=False)
    print("\nResults saved to results_approach2.csv")
