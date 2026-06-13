# TextClassification — Emotion Detection with DistilBERT

Fine-tune DistilBERT to classify Twitter messages into six emotional states: **sadness, joy, love, anger, fear, surprise**, using the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## What's inside

`emotion_classification.ipynb` walks through:

1. **Data exploration** — class distribution, tweet length analysis, UMAP visualisation of learned embeddings
2. **Tokenization** — `distilbert-base-uncased` tokenizer with padding/truncation to 128 tokens
3. **Baseline** — freeze DistilBERT, extract `[CLS]` embeddings, train a logistic regression classifier (no GPU required)
4. **Fine-tuning** — attach a classification head and train the full model end-to-end via the HuggingFace `Trainer` API
5. **Inference** — run the fine-tuned model on arbitrary text via a `pipeline`

## Dependencies

| Package | Version used |
|---|---|
| `torch` | 2.12 |
| `transformers` | 5.10 |
| `datasets` | 3.6 |
| `evaluate` | latest |
| `scikit-learn` | 1.7 |
| `umap-learn` | 0.5 |
| `numpy`, `pandas`, `matplotlib` | standard |

Install all at once:

```bash
pip install torch transformers datasets evaluate scikit-learn umap-learn numpy pandas matplotlib
```

## Usage

```bash
cd TextClassification
jupyter notebook emotion_classification.ipynb
```

Run cells top-to-bottom. The dataset and model weights are downloaded automatically from the HuggingFace Hub on first run. Fine-tuning checkpoints are saved to `./results/distilbert-emotion/`.
