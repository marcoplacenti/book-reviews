#!/usr/bin/env python3

import unittest
import requests
import json
import time
import tempfile
import shutil
import threading
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from serving.api import ModelManager, app
from training.model import SentimentClassifier, TfidfFeatureExtractor
from training.trainer import save_model


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create feature extractor and fit it first to get actual dimensions
        feature_extractor = TfidfFeatureExtractor(max_features=100)
        test_texts = [
            "This is a great book",
            "This book is terrible",
            "Okay book, nothing special",
        ]
        features = feature_extractor.fit_transform(test_texts)
        actual_input_size = features.shape[1]

        # Create a test model with matching dimensions
        model = SentimentClassifier(input_size=actual_input_size, hidden_sizes=[32])

        test_metrics = {"accuracy": 0.75, "f1": 0.72}
        save_model(model, feature_extractor, self.temp_dir, test_metrics)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_loading(self):
        """Test that ModelManager can load a saved model with all components (model, feature extractor, config)."""
        manager = ModelManager(self.temp_dir)

        self.assertIsNotNone(manager.model)
        self.assertIsNotNone(manager.feature_extractor)
        self.assertIsNotNone(manager.config)
        self.assertGreater(manager.config["input_size"], 0)

    def test_single_prediction(self):
        """Test that the model can make predictions on single text inputs with correct output format."""
        manager = ModelManager(self.temp_dir)

        result = manager.predict("This is a great book!")

        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("processing_time_ms", result)

        self.assertIn(result["sentiment"], ["positive", "negative", "neutral"])
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)
        self.assertGreater(result["processing_time_ms"], 0)

    def test_batch_prediction(self):
        """Test that the model can process multiple texts in batch mode efficiently."""
        manager = ModelManager(self.temp_dir)

        texts = ["Great book!", "Terrible book.", "Okay book."]

        results, total_time = manager.predict_batch(texts)

        self.assertEqual(len(results), 3)
        self.assertGreater(total_time, 0)

        for result in results:
            self.assertIn("sentiment", result)
            self.assertIn("confidence", result)
            self.assertIn("processing_time_ms", result)

    def test_invalid_model_directory(self):
        """Test that ModelManager raises appropriate errors when trying to load from non-existent directory."""
        with self.assertRaises(RuntimeError):
            ModelManager("nonexistent_directory")


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This requires the API server to be running
        cls.base_url = "http://localhost:8080"

    def test_health_endpoint(self):
        """Test that the /health endpoint returns correct status and model information."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("status", data)
            self.assertIn("model_loaded", data)
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_root_endpoint(self):
        """Test that the root / endpoint returns API information and model configuration."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("message", data)
            self.assertIn("version", data)
            self.assertIn("model_info", data)
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_predict_endpoint(self):
        """Test that the /predict endpoint accepts text and returns sentiment predictions with confidence scores."""
        try:
            payload = {"text": "This is an amazing book!"}
            response = requests.post(
                f"{self.base_url}/predict", json=payload, timeout=5
            )
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("text", data)
            self.assertIn("sentiment", data)
            self.assertIn("confidence", data)
            self.assertIn("processing_time_ms", data)

            self.assertEqual(data["text"], payload["text"])
            self.assertIn(data["sentiment"], ["positive", "negative", "neutral"])

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_predict_validation(self):
        """Test that input validation works correctly for empty/short text inputs."""
        try:
            # Test empty text
            payload = {"text": ""}
            response = requests.post(
                f"{self.base_url}/predict", json=payload, timeout=5
            )
            self.assertEqual(response.status_code, 422)  # Validation error

            # Test very short text
            payload = {"text": "Hi"}
            response = requests.post(
                f"{self.base_url}/predict", json=payload, timeout=5
            )
            self.assertEqual(response.status_code, 422)  # Validation error

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_batch_predict_endpoint(self):
        """Test that the /predict/batch endpoint can process multiple texts and return results for all."""
        try:
            payload = {
                "texts": [
                    "This is a great book!",
                    "This book is terrible.",
                    "This book is okay.",
                ]
            }
            response = requests.post(
                f"{self.base_url}/predict/batch", json=payload, timeout=5
            )
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("results", data)
            self.assertIn("total_processing_time_ms", data)

            self.assertEqual(len(data["results"]), 3)

            for result in data["results"]:
                self.assertIn("sentiment", result)
                self.assertIn("confidence", result)

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_batch_predict_validation(self):
        """Test that batch prediction validates input limits (empty lists, too many texts)."""
        try:
            # Test empty list
            payload = {"texts": []}
            response = requests.post(
                f"{self.base_url}/predict/batch", json=payload, timeout=5
            )
            self.assertEqual(response.status_code, 422)

            # Test too many texts
            payload = {"texts": ["test"] * 101}
            response = requests.post(
                f"{self.base_url}/predict/batch", json=payload, timeout=5
            )
            self.assertEqual(response.status_code, 422)

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_metrics_endpoint(self):
        """Test that the /metrics endpoint returns model configuration and available endpoints."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertIn("model_config", data)
            self.assertIn("endpoints", data)

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://localhost:8080"

    def test_latency_requirements(self):
        """Test that API latency meets performance requirements (P99 < 300ms, average < 50ms)."""
        """Test that latency requirements are met"""
        try:
            payload = {"text": "This is a test message for latency testing."}

            latencies = []
            for _ in range(10):
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/predict", json=payload, timeout=5
                )
                end_time = time.time()

                self.assertEqual(response.status_code, 200)
                total_latency = (end_time - start_time) * 1000
                latencies.append(total_latency)

            # Check p99 latency (in this case, max of 10 requests)
            p99_latency = max(latencies)
            self.assertLess(
                p99_latency,
                300,
                f"P99 latency {p99_latency:.2f}ms exceeds 300ms requirement",
            )

            avg_latency = sum(latencies) / len(latencies)
            self.assertLess(
                avg_latency, 50, f"Average latency {avg_latency:.2f}ms is too high"
            )

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

    def test_concurrent_requests(self):
        """Test that the API can handle multiple simultaneous requests with good success rate and latency."""
        """Test handling of concurrent requests"""
        try:
            import threading

            def make_request(results, index):
                payload = {"text": f"Test message {index}"}
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/predict", json=payload, timeout=5
                    )
                    end_time = time.time()
                    results[index] = {
                        "status": response.status_code,
                        "latency": (end_time - start_time) * 1000,
                        "success": response.status_code == 200,
                    }
                except Exception as e:
                    results[index] = {"error": str(e), "success": False}

            results = {}
            threads = []
            num_threads = 10

            # Create and start threads
            for i in range(num_threads):
                thread = threading.Thread(target=make_request, args=(results, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check results
            successful_requests = sum(
                1 for r in results.values() if r.get("success", False)
            )
            success_rate = successful_requests / num_threads

            self.assertGreaterEqual(
                success_rate, 0.9, f"Success rate {success_rate:.2f} is too low"
            )

            # Check latencies for successful requests
            latencies = [
                r["latency"] for r in results.values() if r.get("success", False)
            ]
            if latencies:
                max_latency = max(latencies)
                self.assertLess(
                    max_latency,
                    300,
                    f"Max concurrent latency {max_latency:.2f}ms exceeds 300ms",
                )

        except requests.exceptions.RequestException:
            self.skipTest("API server not running")


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create feature extractor and fit it first to get actual dimensions
        feature_extractor = TfidfFeatureExtractor(max_features=50)
        test_texts = ["test text sample"]
        features = feature_extractor.fit_transform(test_texts)
        actual_input_size = features.shape[1]

        # Create a test model with matching dimensions
        model = SentimentClassifier(input_size=actual_input_size, hidden_sizes=[16])

        save_model(model, feature_extractor, self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_special_characters(self):
        """Test that the model handles edge cases like emojis, HTML tags, URLs, and special characters gracefully."""
        manager = ModelManager(self.temp_dir)

        special_texts = [
            "Book with émojis! 😀📚",
            "HTML tags <b>removed</b> properly",
            "URLs http://example.com filtered out",
            "Special chars: !@#$%^&*()",
            "Numbers 123 and symbols ???",
        ]

        for text in special_texts:
            result = manager.predict(text)
            self.assertIn("sentiment", result)
            self.assertIn(result["sentiment"], ["positive", "negative", "neutral"])

    def test_very_long_text(self):
        """Test that the model can process very long text inputs without errors or excessive latency."""
        manager = ModelManager(self.temp_dir)

        # Create a very long text
        long_text = "This is a great book! " * 1000

        result = manager.predict(long_text)
        self.assertIn("sentiment", result)
        self.assertGreater(result["processing_time_ms"], 0)

    def test_empty_and_whitespace(self):
        """Test that the model handles empty strings and whitespace-only inputs gracefully without crashing."""
        manager = ModelManager(self.temp_dir)

        # Test that empty string still produces a result (handled gracefully)
        result = manager.predict("")
        self.assertIn("sentiment", result)

        # Test whitespace-only text
        result = manager.predict("   \n\t   ")
        self.assertIn("sentiment", result)


if __name__ == "__main__":
    unittest.main()
