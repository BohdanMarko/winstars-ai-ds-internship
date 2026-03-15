import numpy as np
from classifiers.random_forest import RandomForestMnistClassifier
from classifiers.feed_forward_nn import FeedForwardNNMnistClassifier
from classifiers.cnn import CNNMnistClassifier
from interface import MnistClassifierInterface


CLASSIFIERS: dict[str, type[MnistClassifierInterface]] = {
    "rf":   RandomForestMnistClassifier,
    "ffnn": FeedForwardNNMnistClassifier,
    "cnn":  CNNMnistClassifier
}


class MnistClassifier:
    """Facade that delegates to one of the registered classifiers.

    Selects the concrete classifier by key ("rf", "ffnn", "cnn") and
    forwards all hyperparameters via **kwargs.
    """

    def __init__(self, algorithm: str, **kwargs):
        if algorithm not in CLASSIFIERS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Supported algorithms: {list(CLASSIFIERS.keys())}"
            )
        self._classifier: MnistClassifierInterface = CLASSIFIERS[algorithm](**kwargs)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._classifier.train(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X_test)