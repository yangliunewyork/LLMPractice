# TextClassification — Emotion Detection with DistilBERT

Fine-tune DistilBERT to classify Twitter messages into six emotional states: **sadness, joy, love, anger, fear, surprise**, using the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## Design decisions

### Why a Transformer encoder?

Text classification is an *understanding* task: given a sentence, assign a label. This requires reading the full sequence **bidirectionally** — each token must attend to what comes before *and* after it to build a rich contextual representation. That is the defining property of **Transformer encoders**.

- **Decoder-only models** (e.g. GPT) process tokens left-to-right and are optimised for generation — wrong fit for classification.
- **Encoder-decoder models** (e.g. T5) add a cross-attention decoder for sequence-to-sequence tasks — unnecessary complexity here.
- A **pure encoder** is the right inductive bias: it reads the full input, produces one hidden state per token, and we reduce those to a single sentence vector for the classifier.

### Why DistilBERT specifically?

BERT is the canonical Transformer encoder for NLP classification, pre-trained with masked language modelling (bidirectional) on large corpora. **DistilBERT** is a knowledge-distilled version that:

- Is **40 % smaller** (66 M vs 110 M parameters) and **60 % faster** at inference
- Retains **97 % of BERT's GLUE score**
- Fits comfortably on a laptop CPU, making it practical for experimentation

The `distilbert-base-uncased` checkpoint lowercases all input before tokenization, which reduces vocabulary size with negligible accuracy cost for emotion/sentiment tasks where casing carries little signal.

### Baseline vs. fine-tuning

The notebook explores both strategies deliberately:

| Strategy | What trains | Accuracy | Purpose |
|---|---|---|---|
| Frozen encoder + logistic regression | LR head only | ~63 % | Shows the quality of pre-trained representations without any task-specific gradient signal |
| Full fine-tuning | Entire DistilBERT + head | ~92 % | Shows the gain from adapting the encoder weights to the target distribution |

The gap (~29 pp) illustrates why fine-tuning matters even when the pre-trained embeddings are already meaningful.

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
