import json
import re
import pandas as pd
import nltk
from typing import Tuple
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from shared.logger_config import get_logger


class DataPreprocessor:
    def __init__(self):
        self.logger = get_logger("data_preprocessing")
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def load_data(self, file_path: str) -> pd.DataFrame:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        df = pd.DataFrame(data)
        self.logger.info(f"Loaded {len(df)} reviews")
        return df

    def clean_text(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ''

        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s\.\!\?\,]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def rating_to_sentiment(self, rating: float) -> int:
        if rating <= 2.0:
            return 0  # Negative
        elif rating == 3.0:
            return 1  # Neutral
        else:
            return 2  # Positive

    def create_sentence_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        sentence_data = []

        for _, row in df.iterrows():
            rating = row['rating']
            title = str(row['title']) if pd.notna(row['title']) else ''
            text = str(row['text']) if pd.notna(row['text']) else ''

            full_text = f"{title}. {text}".strip()
            cleaned_text = self.clean_text(full_text)

            if len(cleaned_text) < 10:
                continue

            sentences = sent_tokenize(cleaned_text)
            sentiment = self.rating_to_sentiment(rating)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 10:
                    sentence_data.append({
                        'sentence': sentence,
                        'sentiment': sentiment,
                        'rating': rating
                    })

        sentence_df = pd.DataFrame(sentence_data)
        self.logger.info(f"Created {len(sentence_df)} sentence examples")

        sentiment_counts = sentence_df['sentiment'].value_counts().sort_index()
        self.logger.info(f"Sentiment distribution: Negative: {sentiment_counts.get(0, 0)}, "
                        f"Neutral: {sentiment_counts.get(1, 0)}, Positive: {sentiment_counts.get(2, 0)}")

        return sentence_df

    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = df[['sentence']]
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )

        train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

        self.logger.info(f"Train set: {len(train_df)} samples")
        self.logger.info(f"Test set: {len(test_df)} samples")

        return train_df, test_df
