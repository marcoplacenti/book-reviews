import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 5000,
        hidden_sizes: list = [512, 256],
        num_classes: int = 3,
        dropout_rate: float = 0.3,
    ):
        super(SentimentClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TfidfFeatureExtractor:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            strip_accents="ascii",
        )
        self.is_fitted = False

    def fit(self, texts: list):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("TfidfFeatureExtractor must be fitted before transform")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
