from abc import ABC, abstractmethod
import numpy as np


class MnistClassifierInterface(ABC):
    """Abstract interface that all MNIST classifiers must implement."""
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the classifier on the provided training data.

        Args:
            X_train (np.ndarray): Flattened images, shape (N, 784), values in [0, 255].
            y_train (np.ndarray): Integer class labels, shape (N,), values in {0, ..., 9}.
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test data.

        Args:
            X_test (np.ndarray): Flattened images, shape (N, 784), values in [0, 255].

        Returns:
            np.ndarray: Predicted integer class labels, shape (N,), values in {0, ..., 9}.
        """
        pass