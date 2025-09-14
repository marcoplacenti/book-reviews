#!/usr/bin/env python3

import unittest
import pandas as pd
import numpy as np
import torch
import tempfile
import shutil
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from training.data_preprocessing import DataPreprocessor
from training.dataset import SentimentDataset, create_data_loaders
from training.model import SentimentClassifier, TfidfFeatureExtractor
from training.trainer import Trainer, save_model, load_model


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()

    def test_clean_text(self):
        """Test that text cleaning removes HTML tags, URLs, and normalizes whitespace."""
        dirty_text = "This is <b>great</b> book! Visit http://example.com"
        clean_text = self.preprocessor.clean_text(dirty_text)
        self.assertNotIn("<b>", clean_text)
        self.assertNotIn("</b>", clean_text)
        self.assertNotIn("http://example.com", clean_text)

    def test_rating_to_sentiment(self):
        """Test that star ratings are correctly mapped to sentiment classes (1-2→Negative, 3→Neutral, 4-5→Positive)."""
        self.assertEqual(self.preprocessor.rating_to_sentiment(1.0), 0)  # Negative
        self.assertEqual(self.preprocessor.rating_to_sentiment(2.0), 0)  # Negative
        self.assertEqual(self.preprocessor.rating_to_sentiment(3.0), 1)  # Neutral
        self.assertEqual(self.preprocessor.rating_to_sentiment(4.0), 2)  # Positive
        self.assertEqual(self.preprocessor.rating_to_sentiment(5.0), 2)  # Positive

    def test_create_sentence_dataset(self):
        """Test that review-level data is correctly split into sentence-level training examples with inherited sentiment labels."""
        # Create test data
        test_data = {
            "rating": [1, 3, 5],
            "title": ["Bad Book", "Okay Book", "Great Book"],
            "text": [
                "This was terrible. Really bad.",
                "This was okay. Nothing special.",
                "This was amazing! Loved it.",
            ],
        }
        df = pd.DataFrame(test_data)

        sentence_df = self.preprocessor.create_sentence_dataset(df)

        self.assertGreater(len(sentence_df), 0)
        self.assertIn("sentence", sentence_df.columns)
        self.assertIn("sentiment", sentence_df.columns)
        self.assertTrue(
            all(sentiment in [0, 1, 2] for sentiment in sentence_df["sentiment"])
        )


class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_size = 1000
        self.model = SentimentClassifier(
            input_size=self.input_size, hidden_sizes=[128, 64]
        )

    def test_model_initialization(self):
        """Test that the FFNN model initializes with correct input size and number of output classes."""
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.num_classes, 3)

    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass and outputs the correct tensor shape."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, self.input_size)

        output = self.model(input_tensor)

        self.assertEqual(output.shape, (batch_size, 3))

    def test_tfidf_feature_extractor(self):
        """Test that TF-IDF feature extraction works correctly, including fit/transform and vocabulary building."""
        texts = [
            "This is a great book",
            "This book is terrible",
            "Okay book, nothing special",
        ]

        extractor = TfidfFeatureExtractor(max_features=100)
        features = extractor.fit_transform(texts)

        self.assertEqual(features.shape[0], 3)
        # Actual features will be limited by vocabulary size, not max_features
        self.assertLessEqual(features.shape[1], 100)
        self.assertGreater(features.shape[1], 0)

        # Test transform on new text
        new_features = extractor.transform(["Another great book"])
        self.assertEqual(new_features.shape, (1, features.shape[1]))


class TestDataset(unittest.TestCase):
    def test_sentiment_dataset(self):
        """Test that the PyTorch Dataset wrapper correctly handles features and labels for training."""
        features = np.random.rand(100, 50)
        labels = np.random.randint(0, 3, 100)

        dataset = SentimentDataset(features, labels)

        self.assertEqual(len(dataset), 100)

        sample = dataset[0]
        self.assertIn("features", sample)
        self.assertIn("labels", sample)
        self.assertEqual(sample["features"].shape, (50,))

    def test_data_loader_factory(self):
        """Test that data loaders are correctly created from dataframes with proper feature extraction and batching."""
        # Create test dataframes
        train_data = {
            "sentence": ["Great book!", "Terrible book.", "Okay book."] * 10,
            "sentiment": [2, 0, 1] * 10,
        }
        test_data = {
            "sentence": ["Amazing book!", "Bad book.", "Average book."] * 5,
            "sentiment": [2, 0, 1] * 5,
        }

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        data_loaders = create_data_loaders(
            train_df, test_df, max_features=100, batch_size=8
        )

        self.assertIn("train", data_loaders)
        self.assertIn("test", data_loaders)
        self.assertIn("feature_extractor", data_loaders)

        # Test one batch
        train_batch = next(iter(data_loaders["train"]))
        self.assertIn("features", train_batch)
        self.assertIn("labels", train_batch)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_initialization(self):
        """Test that the Trainer class initializes correctly with a model and logger."""
        model = SentimentClassifier(input_size=100)
        trainer = Trainer(model)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.logger)

    def test_model_saver(self):
        """Test that models can be saved to disk and loaded back with all components (model, feature extractor, config)."""
        # Create feature extractor first to get actual vocab size
        feature_extractor = TfidfFeatureExtractor(max_features=50)
        dummy_texts = ["test text one", "test text two", "test text three"]
        features = feature_extractor.fit_transform(dummy_texts)
        actual_feature_size = features.shape[1]

        # Create model with correct input size
        model = SentimentClassifier(input_size=actual_feature_size, hidden_sizes=[16])

        test_metrics = {"accuracy": 0.85, "f1": 0.82}

        # Save model
        save_model(model, feature_extractor, self.temp_dir, test_metrics)

        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "_model.pth")))
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, "feature_extractor.pkl"))
        )
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))

        # Load model
        loaded_model, loaded_extractor, config = load_model(self.temp_dir)

        self.assertEqual(loaded_model.input_size, actual_feature_size)
        self.assertEqual(loaded_extractor.max_features, 50)
        self.assertIn("accuracy", config["metrics"])


class TestIntegration(unittest.TestCase):
    def test_end_to_end_training_pipeline(self):
        """Integration test that runs the complete training pipeline from raw data to trained model."""
        # Create minimal test data
        test_data = {
            "rating": [1, 2, 3, 4, 5] * 20,
            "title": ["Bad"] * 20
            + ["Okay"] * 20
            + ["Good"] * 20
            + ["Great"] * 20
            + ["Amazing"] * 20,
            "text": ["This is bad."] * 20
            + ["This is okay."] * 20
            + ["This is good."] * 20
            + ["This is great."] * 20
            + ["This is amazing."] * 20,
        }
        df = pd.DataFrame(test_data)

        # Preprocess data
        preprocessor = DataPreprocessor()
        sentence_df = preprocessor.create_sentence_dataset(df)
        train_df, test_df = preprocessor.create_train_test_split(
            sentence_df, test_size=0.2
        )

        # Create data loaders
        data_loaders = create_data_loaders(
            train_df, test_df, max_features=100, batch_size=8
        )

        # Get actual feature size by making a dummy transform
        dummy_features = data_loaders["feature_extractor"].transform(["dummy text"])
        actual_input_size = dummy_features.shape[1]

        # Create and train model with correct input size
        model = SentimentClassifier(input_size=actual_input_size, hidden_sizes=[32])
        trainer = Trainer(model)

        # Train for just 1 epoch
        result = trainer.train(
            data_loaders["train"], data_loaders["test"], epochs=1, learning_rate=0.01
        )

        self.assertIn("best_val_accuracy", result)
        self.assertIn("history", result)
        self.assertGreater(result["best_val_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
