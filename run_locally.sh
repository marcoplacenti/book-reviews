python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -m pytest tests/test_training.py -v

python -m training.train --epochs 1 --batch_size 16

cd deployment
docker compose build sentiment-api
docker compose up -d sentiment-api
sleep 30
python -m pytest tests/test_serving.py -v

curl -X POST http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This book was absolutely amazing! I loved every single page. Wow!!"}'

docker compose down

# gcloud run deploy book-reviews-sentiment-analysis \
#   --image europe-north2-docker.pkg.dev/book-reviews-472014/book-reviews-docker-repository/book-reviews-app:latest \
#   --platform managed \
#   --region europe-north2 \
#   --allow-unauthenticated