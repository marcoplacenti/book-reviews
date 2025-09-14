import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict
from .model import TfidfFeatureExtractor


class SentimentDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "labels": self.labels[idx]}


def create_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 5000,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Basic function to create the data loaders for training

    Args:
        train_df (pd.DataFrame): _description_
        test_df (pd.DataFrame): _description_
        max_features (int, optional): _description_. Defaults to 5000.
        batch_size (int, optional): _description_. Defaults to 32.
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        Dict[str, DataLoader]: _description_
    """
    feature_extractor = TfidfFeatureExtractor(max_features=max_features)

    train_texts = train_df["sentence"].tolist()
    test_texts = test_df["sentence"].tolist()

    train_features = feature_extractor.fit_transform(train_texts)
    test_features = feature_extractor.transform(test_texts)

    train_labels = train_df["sentiment"].values
    test_labels = test_df["sentiment"].values

    train_dataset = SentimentDataset(train_features, train_labels)
    test_dataset = SentimentDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {
        "train": train_loader,
        "test": test_loader,
        "feature_extractor": feature_extractor,
    }
