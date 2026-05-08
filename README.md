# 🇲🇦 Darija Text Summarization

Automatic text summarization for **Moroccan Darija** — three independent approaches ranging from classical NLP to deep learning, all trained/evaluated on the same CSV dataset.

---

##  Repository Structure

```
darija-summarization/
│
├── approach_1_nlp_extractive.py   # TF-IDF sentence scoring (NLP)
├── approach_2_ml_transformers.py  # HuggingFace seq2seq fine-tuning (Deep ML)
├── approach_3_ml_textrank.py      # TextRank graph algorithm (Classical ML)
│
├── dataset.csv                    # Your Darija dataset (add this)
├── requirements.txt               # All dependencies
└── README.md
```

---

##  Dataset Format

Place your dataset at the root as `dataset.csv` with at least these two columns:

| Column    | Description                                  |
|-----------|----------------------------------------------|
| `text`    | Full Darija text to summarize                |
| `summary` | Reference summary *(optional, for eval only)*|

> The scripts default to `text` and `summary` column names. You can change these via the `TEXT_COL` / `SUMMARY_COL` constants at the top of each file.

---

## 🔬 Approach Comparison

| | Approach 1 | Approach 2 | Approach 3 |
|---|---|---|---|
| **File** | `approach_1_nlp_extractive.py` | `approach_2_ml_transformers.py` | `approach_3_ml_textrank.py` |
| **Type** | Extractive | **Abstractive** | Extractive |
| **Algorithm** | TF-IDF sentence scoring | mT5 / mBART fine-tuning | TextRank (PageRank on sentences) |
| **Needs labels?** |  No |  Yes |  No |
| **Needs GPU?** |  No |  Recommended |  No |
| **Output style** | Picks original sentences | Generates new text | Picks original sentences |
| **Speed** | Very fast |  Slow (training) |  Fast |
| **Best for** | Quick baseline | Highest quality | Balanced accuracy/speed |

---

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or install per approach:

```bash
# Approach 1
pip install nltk scikit-learn pandas

# Approach 2
pip install transformers datasets torch sentencepiece pandas rouge-score

# Approach 3
pip install scikit-learn networkx pandas numpy nltk
```

### 2. Run an approach

```bash
# Approach 1 — TF-IDF Extractive
python approach_1_nlp_extractive.py

# Approach 2 — Transformer (trains then evaluates)
python approach_2_ml_transformers.py

# Approach 3 — TextRank
python approach_3_ml_textrank.py
```

Each script prints sample summaries to the console and saves results to a CSV:

- `results_approach1.csv`
- `results_approach2.csv`
- `results_approach3.csv`

---

##  Approach Details

### Approach 1 — TF-IDF Extractive (`approach_1_nlp_extractive.py`)

**How it works:**
1. Split the document into sentences
2. Vectorize sentences using TF-IDF (with Darija stop-words removed)
3. Score each sentence by the sum of its TF-IDF weights
4. Return the top-N highest-scoring sentences in their original order

**Key parameters** (top of file):
```python
NUM_SENTENCES = 3       # sentences in output summary
MAX_FEATURES  = 500     # TF-IDF vocabulary size
```

**Evaluation metric:** Token-level F1 overlap against reference summaries.

---

### Approach 2 — HuggingFace Transformers (`approach_2_ml_transformers.py`)

**How it works:**
1. Load a multilingual seq2seq checkpoint (`csebuetnlp/mT5_multilingual_XLSum`)
2. Fine-tune on your Darija (text, summary) pairs using `Seq2SeqTrainer`
3. At inference, generate abstractive summaries with beam search

**Key parameters:**
```python
BASE_MODEL     = "csebuetnlp/mT5_multilingual_XLSum"
EPOCHS         = 3
BATCH_SIZE     = 4
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128
```

**Switching the base model:**
```python
# For mBART instead of mT5:
BASE_MODEL = "facebook/mbart-large-cc25"
```

**Evaluation metric:** ROUGE-1, ROUGE-2, ROUGE-L (requires `rouge-score`).

>  Training requires ~4 GB GPU memory with `BATCH_SIZE=4`. Reduce batch size or use `fp16=False` on CPU.

---

### Approach 3 — TextRank (`approach_3_ml_textrank.py`)

**How it works:**
1. Represent sentences as TF-IDF vectors
2. Build a fully connected graph where edge weights = cosine similarity
3. Run PageRank on the graph — central sentences get higher scores
4. Return the top-N ranked sentences

**Key parameters:**
```python
NUM_SENTENCES = 3      # sentences in output summary
DAMPING       = 0.85   # PageRank damping factor
MAX_ITER      = 100    # PageRank convergence iterations
```

**Evaluation metric:** Token-level F1 overlap.

---

##  Darija Language Notes

- All three approaches support **Latin-script Darija** out of the box
- For **Arabic-script Darija**, Approach 2 (mT5) handles it natively; Approaches 1 & 3 work but benefit from Arabic-aware tokenization
- Extend `DARIJA_STOPWORDS` in Approaches 1 & 3 with tokens from your own vocabulary for better results
- Approach 2's language prefix is set to `"ar"` (Arabic) — change to `"ary"` if your model supports the ISO Moroccan Arabic code

---

##  Evaluation Output

After running any script, a results CSV is generated:

| Column | Description |
|---|---|
| `original` | Input Darija text |
| `predicted_summary` | Generated summary |
| `reference_summary` | Ground-truth summary *(if available)* |
| `token_f1` / `rouge*` | Evaluation score *(if reference available)* |

---

## 🛠 requirements.txt

```
nltk>=3.8
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
networkx>=3.1
transformers>=4.38
datasets>=2.16
torch>=2.1
sentencepiece>=0.1.99
rouge-score>=0.1.2
```



##  Contributing

Pull requests are welcome! Ideas for improvement:
- Add Arabic-script Darija tokenizer
- Integrate `arabert` or `DarijaBERT` as a Approach 2 backbone
- Add a Streamlit demo UI
- Benchmark all three approaches on a shared test split
