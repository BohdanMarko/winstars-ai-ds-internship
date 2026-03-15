# Task 1 - MNIST Image Classification

Three MNIST classifiers built behind a unified OOP interface. All models accept the same inputs and return the same outputs regardless of the algorithm selected.

## Models

| Key | Class | Approach |
|-----|-------|----------|
| `rf` | `RandomForestMnistClassifier` | Ensemble of decision trees (sklearn) |
| `ffnn` | `FeedForwardNNMnistClassifier` | Fully-connected neural network (PyTorch) |
| `cnn` | `CNNMnistClassifier` | Convolutional neural network (PyTorch) |

## Project Structure

```
task1/
├── interface.py            # MnistClassifierInterface (ABC)
├── mnist_classifier.py     # Unified wrapper - selects classifier by key
├── classifiers/
│   ├── random_forest.py    # RF via sklearn
│   ├── feed_forward_nn.py  # MLP via PyTorch
│   └── cnn.py              # CNN via PyTorch
├── tests/
│   └── test_classifiers.py
├── demo.ipynb              # Training, evaluation, comparison
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10
- Dependencies listed in `requirements.txt`

## Setup

The virtual environment is shared across both tasks and lives at the repo root.

**Create the environment (once):**
```bash
py -3.10 -m venv venv
```

**Activate:**
```bash
venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -r task1/requirements.txt
```

**Register the venv as a Jupyter kernel (for notebooks):**
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python 3.10 (venv)"
```
Then in VS Code or Jupyter, select the **"Python 3.10 (venv)"** kernel when opening `demo.ipynb`.

## Usage

```python
from mnist_classifier import MnistClassifier

clf = MnistClassifier("cnn")       # or "rf", "ffnn"
clf.train(X_train, y_train)
preds = clf.predict(X_test)
```

Hyperparameters are forwarded via `**kwargs`:
```python
clf = MnistClassifier("rf", n_estimators=200, max_depth=20)
clf = MnistClassifier("ffnn", epochs=10, dropout=0.5)
clf = MnistClassifier("cnn", epochs=20, learning_rate=5e-4)
```
