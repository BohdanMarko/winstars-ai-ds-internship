import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from interface import MnistClassifierInterface


class _ConvNet(nn.Module):
    """Convolutional Neural Network for 28x28 grayscale images.

    Unlike the MLP (which sees a flat pixel vector), a CNN preserves 2D
    spatial structure and learns local patterns via sliding filters:
      - Conv2d: slides small learned filters (3x3) over the image to detect
        local features (edges, curves, corners)
      - ReLU: non-linearity after each convolution
      - MaxPool2d(2): halves spatial dimensions, keeping only the strongest
        activations — adds translation invariance

    Three conv blocks stack increasingly abstract features:
      Block 1 (32 filters): low-level edges and strokes
      Block 2 (64 filters): combinations -> shapes, loops
      Block 3 (128 filters): high-level digit parts

    Then a fully-connected head classifies the extracted feature map:
      Flatten (N,1152) -> FC (N,256) -> ReLU -> Dropout -> FC (N,10)

    Spatial dimensions: (N,1,28,28) -> (N,32,14,14) -> (N,64,7,7) -> (N,128,3,3)
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNMnistClassifier(MnistClassifierInterface):
    """MNIST classifier using a Convolutional Neural Network.

    The strongest of the three classifiers - exploits 2D spatial structure
    that RF and FFNN ignore. Reshapes flat (N, 784) -> (N, 1, 28, 28)
    and normalises [0, 255] -> [0, 1] internally.

    Training: Adam + CosineAnnealingLR scheduler (smoothly decays LR from
    initial value to ~0 over all epochs, helping the model converge to a
    sharper minimum compared to a fixed LR).
    """

    def __init__(
        self,
        epochs: int = 15,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.4,
        random_state: int = 67,
        device: str | None = None,
    ):
        """
        Args:
            epochs: Training passes. Fewer than FFNN (15 vs 20) because
                convolutions converge faster on image data.
            batch_size: Samples per gradient step.
            learning_rate: Initial Adam LR (decayed by cosine scheduler).
            dropout: Applied in the FC head before final classification layer.
            random_state: Seed for reproducibility.
            device: Compute device (auto-detected if None).
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
        self._net = _ConvNet(dropout=dropout).to(self._device)
        self._is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Reshape to (N,1,28,28), normalise, train with Adam + cosine LR decay."""
        X = torch.FloatTensor(X_train.reshape(-1, 1, 28, 28) / 255.0)
        y = torch.LongTensor(y_train)
        loader = DataLoader(TensorDataset(X, y), batch_size=self._batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self._epochs):
            self._net.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._net(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(X_batch)
            scheduler.step()
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{self._epochs} - loss: {avg_loss:.4f}")

        self._is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Reshape to 4D, forward pass in eval mode, argmax -> digit labels."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() before predict().")

        X = torch.FloatTensor(X_test.reshape(-1, 1, 28, 28) / 255.0)
        loader = DataLoader(X, batch_size=self._batch_size, shuffle=False)

        self._net.eval()
        preds = []
        with torch.no_grad():
            for X_batch in loader:
                logits = self._net(X_batch.to(self._device))
                preds.append(logits.argmax(dim=1).cpu().numpy())

        return np.concatenate(preds)
