from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "complaints.csv"
MODEL_DIR = ROOT / "models"


def build_pipeline():
    """Create a readable text-classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=6000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_models():
    """Train and save separate models for category and priority prediction."""
    data = pd.read_csv(DATA_FILE)
    data = data.dropna(subset=["complaint", "category", "priority"])
    data["complaint"] = data["complaint"].astype(str).str.strip()
    data = data[data["complaint"].str.len() > 0]

    MODEL_DIR.mkdir(exist_ok=True)

    category_model = build_pipeline()
    priority_model = build_pipeline()

    category_model.fit(data["complaint"], data["category"])
    priority_model.fit(data["complaint"], data["priority"])

    joblib.dump(category_model, MODEL_DIR / "category_model.joblib")
    joblib.dump(priority_model, MODEL_DIR / "priority_model.joblib")

    print(f"Trained both models using {len(data)} complaint examples.")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    train_models()
