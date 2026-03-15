import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from interface import MnistClassifierInterface


class _FeedForwardNet(nn.Module):
    """Fully-connected (MLP) network: 784 -> 512 -> 256 -> 128 -> 10.

    Every neuron in a layer is connected to every neuron in the next layer
    (hence "fully-connected"). The network sees each image as a flat vector
    of 784 pixels - it has no notion of spatial structure.

    Architecture narrows progressively (512 -> 256 -> 128), forcing the
    network to compress raw pixel information into increasingly abstract
    features before the 10-class output.

    Each hidden block: Linear -> BatchNorm1d -> ReLU -> Dropout
      - Linear:      learnable weighted sum (y = Wx + b)
      - BatchNorm1d: normalises activations per batch -> stabilises training
      - ReLU:        max(0, x) — non-linearity for complex decision boundaries
      - Dropout:     randomly zeroes neurons during training -> reduces overfitting
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForwardNNMnistClassifier(MnistClassifierInterface):
    """MNIST classifier using a Feed-Forward Neural Network (MLP).

    Treats each image as a flat 784-d vector — simple and fast, but ignores
    spatial pixel relationships (unlike CNN). Normalises [0, 255] -> [0, 1]
    internally.

    Training: mini-batch gradient descent with Adam optimiser and
    CrossEntropyLoss (expects raw logits, applies softmax internally).
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.3,
        random_state: int = 67,
        device: str | None = None,
    ):
        """
        Args:
            epochs: Full passes over the training set.
            batch_size: Samples per gradient update.
            learning_rate: Adam step size.
            dropout: Fraction of neurons randomly zeroed during training.
            random_state: Seed for weight initialisation reproducibility.
            device: "cuda", "mps", or "cpu" (auto-detected if None).
        """
        torch.manual_seed(random_state)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._net = _FeedForwardNet(dropout=dropout).to(self._device)
        self._is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Normalise pixels to [0,1], then run Adam optimisation for self._epochs.

        Uses weight_decay=1e-4 (L2 regularization) to penalise large weights.
        """
        X = torch.FloatTensor(X_train / 255.0)
        y = torch.LongTensor(y_train)
        loader = DataLoader(TensorDataset(X, y), batch_size=self._batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self._net.train()
        for epoch in range(self._epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._net(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(X_batch)

            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{self._epochs} - loss: {avg_loss:.4f}")

        self._is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Forward pass in eval mode (BatchNorm uses running stats, Dropout off).

        Returns argmax of raw logits -> predicted digit per sample.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() before predict().")

        X = torch.FloatTensor(X_test / 255.0)
        loader = DataLoader(X, batch_size=self._batch_size, shuffle=False)

        self._net.eval()
        preds = []
        with torch.no_grad():
            for X_batch in loader:
                logits = self._net(X_batch.to(self._device))
                preds.append(logits.argmax(dim=1).cpu().numpy())

        return np.concatenate(preds)
