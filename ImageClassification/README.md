# ImageClassification — MNIST Digit Recognition with a CNN

Classify handwritten digits (0–9) from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using a Convolutional Neural Network (CNN) built in PyTorch.

## What's inside

`mnist_cnn.ipynb` walks through:

1. **CNN explainer** — how convolution, pooling, and weight sharing work; annotated architecture diagram
2. **Data exploration** — sample image grid and class distribution across all 10 digits
3. **Model definition** — two conv blocks (32→64 filters) with dropout, followed by fully-connected layers (~400K parameters)
4. **Training** — 5 epochs with Adam and CrossEntropyLoss; loss and accuracy curves
5. **Evaluation** — test accuracy, per-class precision/recall/F1, and a confusion matrix
6. **Filter visualisation** — plots the 32 learned 3×3 filters from the first conv layer
7. **Failure analysis** — inspects misclassified examples to understand where the model struggles

## Dependencies

| Package | Version used |
|---|---|
| `torch` | 2.12 |
| `torchvision` | 0.27 |
| `scikit-learn` | 1.7 |
| `numpy`, `matplotlib` | standard |

Install all at once:

```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

## Usage

```bash
cd ImageClassification
jupyter notebook mnist_cnn.ipynb
```

Run cells top-to-bottom. The MNIST dataset (~11 MB) is downloaded automatically to `./data/` on first run.
