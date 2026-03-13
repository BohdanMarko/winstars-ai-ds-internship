import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """MNIST classifier using a Random Forest."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 67,
        n_jobs: int = -1,
    ):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)
        self._is_trained = True
        print("RandomForest training completed.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() before predict().")
        return self._model.predict(X_test)
