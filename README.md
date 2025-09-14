# End-to-end ML Engineering Project

A sentiment analysis system for Amazon book reviews using a simple Feed-Forward Neural Network (FFNN) and TF-IDF for feature extraction.

## Requirements

1. Train a classifier that categorizes review sentences into positive, neutral and negative classes. If a review consists of multiple sentences, you can assume each sentence in the review gets its review's star rating. We are less critical about the model you chose. So do not spend too much time on that
2. Deploy the trained model locally so that it can serve predictions with p99 latency of 300ms.
3. Not needed for this case, but ensure your solution can be easily deployed to a cloud environment.
4.	Ensure appropriate logging and monitoring is in place.
5.	Submit your solution via Github.

Things to keep in mind:

- Unit testing
- Readability (Code comments, etc) 
- Thought Process
- Design Choices 
- Performance Considerations
- Productionisation
- Extensibility and maintainability of a system


## Repo Structure

```
data/                # Data to train and test the model
training/            # Code relative to the training pipeline
   model.py          # Model and Feature Extractor
   trainer.py        # Training logic and Cross Validation
   dataset.py        # Data loading and preprocessing
   train.py          # Training entrypoint
serving/             # Code relative to the model serving
   api.py            # FastAPI application
deployment/          # Docker deployment assets
   Dockerfile
   docker-compose.yml   # Docker configuration
tests/               # Tests
shared/              # Shared utilities
models/              # Saved models
```

## Quick Start

### Local Development

To run the application locally, simply execute the following command in the terminal

```bash
   sh run_locally.sh
```

This will:

- Setup the virtual environment
- Run the test for the training stage
- Train and save the model locally
- Build a local docker contrainer
- Test the model serving stage
- Deploy the container on Google Cloud Run (Currently disabled)


## API Endpoints

Note: This uses my own Google Cloud Run endpoint. Either use your own or use localhost for local deployments.

### Single Prediction
```bash
curl -X POST https://book-reviews-sentiment-analysis-6uygtlew7q-ma.a.run.app/predict \
-H "Content-Type: application/json" \
-d '{"text": "This book was absolutely amazing! I loved every single page. Wow!!"}
```

Response:
```json
{
  "text": "This book was absolutely amazing! I loved every single page. Wow!!",
  "sentiment": "positive",
  "confidence": 0.8945,
  "processing_time_ms": 1.23
}
```

### Batch Prediction
```bash
curl -X POST https://book-reviews-sentiment-analysis-6uygtlew7q-ma.a.run.app/predict/batch \
-H "Content-Type: application/json" \
-d '{"texts": ["This book was absolutely amazing! I loved every single page. Wow!!", "This book was very bad."]}

```

### Health Check
```bash
GET /health
```

### Metrics
```bash
GET /metrics
```


## Model Training

### Basic Training
```bash
python -m training.train --epochs 1 --batch_size 64
```

### Cross-Validation (Currently it just does an empty run for CV but it's not really used.)
```bash
python -m training.train --cross_validate --cv_folds 3 --epochs 1
```

### Custom Architecture
For providing custom architecture
```bash
python -m training.train \
  --hidden_sizes 256 128 64 \
  --dropout_rate 0.2 \
  --max_features 10000
```

## Configuration

Key parameters:
- **max_features**: TF-IDF vocabulary size (default: 5000)
- **hidden_sizes**: FFNN layer sizes (default: [512, 256])
- **dropout_rate**: Regularization (default: 0.3)
- **learning_rate**: Adam optimizer (default: 0.001)

## Monitoring

Logging and health checks:
- Structured logging for all requests
- Health endpoint for uptime monitoring
- Performance metrics in logs
- Model performance metrics

## Model Details

- **Architecture**: Feed-Forward Neural Network
- **Input**: TF-IDF vectors (5000 dimensions)
- **Output**: 3 classes (Negative, Neutral, Positive)
- **Training**: Adam optimizer, Cross-Entropy loss
- **Features**: Text cleaning, stopword removal, n-grams
