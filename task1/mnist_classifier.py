from classifiers import RandomForestMnistClassifier
from task1.classifiers.feed_forward_nn import FeedForwardNNMnistClassifier


CLASSIFIERS: dict[str, type] = {
    "rf":  RandomForestMnistClassifier,
    "ffnn": FeedForwardNNMnistClassifier,
}


class MnistClassifier:
    def __init__(self, algorithm: str, **kwargs):
        if algorithm not in CLASSIFIERS:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported algorithms: {list(CLASSIFIERS.keys())}")
        self._classifier = CLASSIFIERS[algorithm](**kwargs)
        
    def train(self, X_train, y_train):
        self._classifier.train(X_train, y_train)
        
    def predict(self, X_test):
        return self._classifier.predict(X_test)