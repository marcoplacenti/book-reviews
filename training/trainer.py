import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from .model import SentimentClassifier, TfidfFeatureExtractor
from .dataset import create_data_loaders
from shared.logger_config import get_logger


class Trainer:
    def __init__(self, model: SentimentClassifier, device: str = None):
        self.model = model
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.logger = get_logger("_trainer")

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(
        self, data_loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Logic to train a single epoch

        Args:
            data_loader (DataLoader): data to train
            optimizer (torch.optim.Optimizer): optimizer

        Returns:
            Tuple[float, float]: loss and accuracy
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(data_loader, desc="Training")

        for batch in progress_bar:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            logits = self.model(features)
            loss = nn.CrossEntropyLoss()(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"}
            )

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Logic to test the model

        Args:
            data_loader (DataLoader): data to test/evaluate

        Returns:
            Tuple[float, float, Dict]: loss, accuracy, and dict with other metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating")

            for batch in progress_bar:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(features)
                loss = nn.CrossEntropyLoss()(logits, labels)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_predictions,
            "true_labels": all_labels,
        }

        return avg_loss, accuracy, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> Dict:
        """Main training logic

        Args:
            train_loader (DataLoader): data to train
            val_loader (DataLoader): data to test/evaluate
            epochs (int, optional): number of epochs. Defaults to 10.
            learning_rate (float, optional): lr. Defaults to 0.001.

        Returns:
            Dict: a dict with the best accuracy score plus all the "historical" metrics
        """

        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")

            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc, val_metrics = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            self.logger.info(f"Val F1: {val_metrics['f1']:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {"best_val_accuracy": best_val_acc, "history": self.history}


class CrossValidator:
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.kfold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        self.logger = get_logger("cross_validator")

    def cross_validate(
        self,
        df: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_features: int = 5000,
    ) -> Dict:
        X = df[["sentence"]].reset_index(drop=True)
        y = df["sentiment"].reset_index(drop=True)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X, y)):
            self.logger.info(f"FOLD {fold + 1}/{self.n_folds}")
            self.logger.info("=" * 60)

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            data_loaders = create_data_loaders(
                train_df, val_df, max_features, batch_size
            )

            model = SentimentClassifier(input_size=max_features)
            trainer = Trainer(model)

            result = trainer.train(
                data_loaders["train"],
                data_loaders["test"],
                epochs=epochs,
                learning_rate=learning_rate,
            )

            _, _, metrics = trainer.evaluate(data_loaders["test"])

            fold_result = {
                "fold": fold + 1,
                "val_accuracy": result["best_val_accuracy"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
            }

            fold_results.append(fold_result)

            self.logger.info(f"Fold {fold + 1} Results:")
            self.logger.info(f"Accuracy: {fold_result['val_accuracy']:.4f}")
            self.logger.info(f"F1: {fold_result['val_f1']:.4f}")

        cv_results = self.summarize_cv_results(fold_results)
        return cv_results

    def summarize_cv_results(self, fold_results: List[Dict]) -> Dict:
        metrics = ["val_accuracy", "val_f1", "val_precision", "val_recall"]
        summary = {}

        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)

        summary["fold_results"] = fold_results

        self.logger.info("=" * 60)
        self.logger.info("CROSS-VALIDATION SUMMARY")
        self.logger.info("=" * 60)

        for metric in metrics:
            mean_val = summary[f"{metric}_mean"]
            std_val = summary[f"{metric}_std"]
            self.logger.info(
                f"{metric.replace('val_', '').title()}: {mean_val:.4f} (+/- {std_val:.4f})"
            )

        return summary


def save_model(
    model: SentimentClassifier,
    feature_extractor: TfidfFeatureExtractor,
    save_dir: str,
    metrics: Dict = None,
):
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "_model.pth")
    torch.save(model.state_dict(), model_path)

    feature_path = os.path.join(save_dir, "feature_extractor.pkl")
    feature_extractor.save(feature_path)

    config = {
        "input_size": model.input_size,
        "num_classes": model.num_classes,
        "hidden_sizes": model.hidden_sizes,
        "dropout_rate": model.dropout_rate,
        "max_features": feature_extractor.max_features,
        "saved_at": datetime.now().isoformat(),
    }

    if metrics:
        config["metrics"] = metrics

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger = get_logger("model_saver")
    logger.info(f"Model saved to {save_dir}")


def load_model(
    save_dir: str,
) -> Tuple[SentimentClassifier, TfidfFeatureExtractor, Dict]:
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model = SentimentClassifier(
        input_size=config["input_size"],
        hidden_sizes=config.get("hidden_sizes", [512, 256]),
        num_classes=config["num_classes"],
        dropout_rate=config.get("dropout_rate", 0.3),
    )

    model_path = os.path.join(save_dir, "_model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    feature_extractor = TfidfFeatureExtractor(max_features=config["max_features"])
    feature_path = os.path.join(save_dir, "feature_extractor.pkl")
    feature_extractor.load(feature_path)

    return model, feature_extractor, config
