# pylint: disable=logging-fstring-interpolation
import argparse
import sys
import torch
from pathlib import Path

from .data_preprocessing import DataPreprocessor
from .dataset import create_data_loaders
from .model import SentimentClassifier
from .trainer import Trainer, CrossValidator, save_model
from shared.logger_config import setup_logger

sys.path.append(str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="Train FFNN sentiment classification model"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Books_10k.jsonl",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features",
    )

    parser.add_argument(
        "--hidden_sizes",
        nargs="+",
        type=int,
        default=[512, 256],
        help="Hidden layer sizes",
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )

    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")

    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Perform cross-validation instead of single train/test split",
    )

    parser.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )

    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    parser.add_argument(
        "--log_file", type=str, default=None, help="Optional log file path"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("train_", level=args.log_level, log_file=args.log_file)

    logger.info(" FFNN Sentiment Classification Training")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Max features: {args.max_features}")
    logger.info(f"Hidden sizes: {args.hidden_sizes}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Dropout rate: {args.dropout_rate}")
    logger.info(f"Cross-validation: {args.cross_validate}")

    if args.cross_validate:
        logger.info(f"CV folds: {args.cv_folds}")

    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 60)

    logger.info("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(args.data_path)
    sentence_df = preprocessor.create_sentence_dataset(df)

    if args.cross_validate:
        logger.info(f"Performing {args.cv_folds}-fold cross-validation...")

        # Currently useless
        cv = CrossValidator(n_folds=args.cv_folds, random_state=args.random_state)
        cv_results = cv.cross_validate(
            sentence_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_features=args.max_features,
        )

        logger.info("Training final model on full dataset...")
        train_df, test_df = preprocessor.create_train_test_split(
            sentence_df, test_size=0.2, random_state=args.random_state
        )

        data_loaders = create_data_loaders(
            train_df, test_df, args.max_features, args.batch_size
        )

        final_model = SentimentClassifier(
            input_size=args.max_features,
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout_rate,
        )
        final_trainer = Trainer(final_model)

        training_result = final_trainer.train(
            data_loaders["train"],
            data_loaders["test"],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

        _, _, final_metrics = final_trainer.evaluate(data_loaders["test"])

        model_metrics = {
            "cross_validation": cv_results,
            "final_test": {
                "accuracy": final_metrics["accuracy"],
                "f1": final_metrics["f1"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
            },
        }

        save_model(
            final_model, data_loaders["feature_extractor"], args.save_dir, model_metrics
        )

        logger.info("Final Results:")
        logger.info(
            f"CV Accuracy: {cv_results['val_accuracy_mean']:.4f} (+/- {cv_results['val_accuracy_std']:.4f})"
        )
        logger.info(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {final_metrics['f1']:.4f}")
    else:
        logger.info("Creating train/test split...")
        train_df, test_df = preprocessor.create_train_test_split(
            sentence_df, test_size=0.2, random_state=args.random_state
        )

        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(
            train_df, test_df, args.max_features, args.batch_size
        )

        logger.info("Creating model...")
        model = SentimentClassifier(
            input_size=args.max_features,
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout_rate,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")

        logger.info("Training model...")
        trainer = Trainer(model)
        training_result = trainer.train(
            data_loaders["train"],
            data_loaders["test"],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

        logger.info("Final evaluation...")
        _, _, final_metrics = trainer.evaluate(data_loaders["test"])

        logger.info("Final Results:")
        logger.info(
            f"Best Validation Accuracy: {training_result['best_val_accuracy']:.4f}"
        )
        logger.info(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {final_metrics['f1']:.4f}")
        logger.info(f"Test Precision: {final_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {final_metrics['recall']:.4f}")

        model_metrics = {
            "test_accuracy": final_metrics["accuracy"],
            "test_f1": final_metrics["f1"],
            "test_precision": final_metrics["precision"],
            "test_recall": final_metrics["recall"],
            "training_history": training_result["history"],
        }

        save_model(
            model, data_loaders["feature_extractor"], args.save_dir, model_metrics
        )

    logger.info(f"Training completed! Model saved to '{args.save_dir}'")


if __name__ == "__main__":
    main()
