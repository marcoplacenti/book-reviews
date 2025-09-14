from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import torch
import time
import os
from typing import Dict, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import load_model
from shared.logger_config import get_logger

app = FastAPI(
    title="Sentiment Analysis API",
    description="Fast sentiment classification for book reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger("api")


class SentimentRequest(BaseModel):
    text: str

    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v.strip()) < 5:
            raise ValueError("Text must be at least 5 characters long")
        return v.strip()


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time_ms: float


class BatchSentimentRequest(BaseModel):
    texts: List[str]

    @validator("texts")
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 texts allowed per batch")
        for text in v:
            if not text or not text.strip():
                raise ValueError("Each text must be non-empty")
            if len(text.strip()) < 5:
                raise ValueError("Each text must be at least 5 characters long")
        return [text.strip() for text in v]


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time_ms: float


class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model = None
        self.feature_extractor = None
        self.config = None
        self.model_dir = model_dir
        self.sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory '{self.model_dir}' not found")

            logger.info(f"Loading model from {self.model_dir}")
            self.model, self.feature_extractor, self.config = load_model(
                self.model_dir
            )
            self.model.eval()
            logger.info(
                f"Model loaded successfully. Input size: {self.config['input_size']}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def predict(self, text: str) -> Dict:
        start_time = time.time()

        try:
            features = self.feature_extractor.transform([text])
            features_tensor = torch.FloatTensor(features)

            with torch.no_grad():
                logits = self.model(features_tensor)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()

            processing_time = (time.time() - start_time) * 1000

            return {
                "sentiment": self.sentiment_labels[predicted_class],
                "confidence": round(confidence, 4),
                "processing_time_ms": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Prediction failed for text: {text[:50]}... Error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        start_time = time.time()

        try:
            features = self.feature_extractor.transform(texts)
            features_tensor = torch.FloatTensor(features)

            with torch.no_grad():
                logits = self.model(features_tensor)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]

            total_processing_time = (time.time() - start_time) * 1000
            per_text_time = total_processing_time / len(texts)

            results = []
            for i, text in enumerate(texts):
                results.append(
                    {
                        "sentiment": self.sentiment_labels[predicted_classes[i].item()],
                        "confidence": round(confidences[i].item(), 4),
                        "processing_time_ms": round(per_text_time, 2),
                    }
                )

            return results, total_processing_time

        except Exception as e:
            logger.error(f"Batch prediction failed. Error: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    logger.info("Sentiment Analysis API started")
    logger.info(
        f"Model loaded with {model_manager.config.get('input_size', 'unknown')} input features"
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Sentiment Analysis API shutting down")


@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "model_info": {
            "input_size": model_manager.config.get("input_size"),
            "num_classes": model_manager.config.get("num_classes"),
            "max_features": model_manager.config.get("max_features"),
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "timestamp": time.time(),
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        result = model_manager.predict(request.text)

        logger.info(
            f"Prediction completed in {result['processing_time_ms']:.2f}ms - "
            f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})"
        )

        return SentimentResponse(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
        )

    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchSentimentRequest):
    try:
        results, total_time = model_manager.predict_batch(request.texts)

        response_results = []
        for i, (text, result) in enumerate(zip(request.texts, results)):
            response_results.append(
                SentimentResponse(
                    text=text,
                    sentiment=result["sentiment"],
                    confidence=result["confidence"],
                    processing_time_ms=result["processing_time_ms"],
                )
            )

        logger.info(
            f"Batch prediction completed for {len(request.texts)} texts in {total_time:.2f}ms "
            f"(avg: {total_time/len(request.texts):.2f}ms per text)"
        )

        return BatchSentimentResponse(
            results=response_results, total_processing_time_ms=round(total_time, 2)
        )

    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    return {
        "model_config": model_manager.config,
        "endpoints": {
            "/predict": "Single text prediction",
            "/predict/batch": "Batch text prediction (max 100)",
            "/health": "Health check",
            "/metrics": "Model metrics and configuration",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
