"""
Sentiment Analysis Training Package

This package contains modules for training a sentiment classification model
on Amazon book reviews.
"""

from .data_preprocessing import DataPreprocessor
from .dataset import SentimentDataset, create_data_loaders
from .model import SentimentClassifier
from .trainer import Trainer, CrossValidator, save_model, load_model


__version__ = "0.1.0"
__all__ = [
    "DataPreprocessor",
    "SentimentDataset",
    "create_data_loaders",
    "SentimentClassifier",
    "Trainer",
    "CrossValidator",
    "save_model",
    "load_model",
]
