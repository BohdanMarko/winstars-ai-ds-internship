import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """MNIST classifier using an ensemble of decision trees (Random Forest).

    Each tree is trained on a random bootstrap sample of the data and considers
    a random subset of features at each split, which decorrelates trees and
    reduces overfitting. Final prediction is a majority vote across all trees.

    Works directly on flat 784-d pixel vectors - no normalization or reshaping
    needed, since decision trees split on raw feature values.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 67,
        n_jobs: int = -1,
    ):
        """
        Args:
            n_estimators: Number of trees in the forest. More trees = more
                stable predictions but slower training.
            max_depth: Maximum depth per tree. None = nodes expand until all
                leaves are pure (or contain < min_samples_split samples).
            min_samples_split: Minimum samples required to split an internal
                node. Higher values regularize the tree.
            random_state: Seed for reproducible bootstrap sampling and feature
                selection.
            n_jobs: Parallel workers for fitting/predicting (-1 = all cores).
        """
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit all trees on bootstrap samples of X_train via sklearn."""
        self._model.fit(X_train, y_train)
        self._is_trained = True
        print("RandomForest training completed.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Majority vote across all trees -> predicted digit per sample."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() before predict().")
        return self._model.predict(X_test)
