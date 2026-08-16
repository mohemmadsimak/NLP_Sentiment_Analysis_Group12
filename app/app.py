"""
app.py — Sentiment Analysis of Customer Reviews (Web Application)
CCS3356 NLP Group Assignment — Group 12 "Logic Legends"

Serves the group's best-performing model (Naive Bayes with balanced sample
weights + TF-IDF, trained by Aasim — see reports/model_comparison.md for why
this model was selected over the higher-accuracy-but-majority-class-only
Random Forest / SVM / LSTM / GRU / SimpleRNN alternatives).

Workflow (matches Section 5, Q10 of the validation submission):
    User Inputs Customer Review
        -> Text Preprocessing (same cleaning + lemmatization as training)
        -> TF-IDF Feature Extraction (fitted vectorizer, loaded not re-fit)
        -> Naive Bayes Prediction
        -> Sentiment Classification + confidence scores
        -> Display Result to User
"""

import os
import re
import joblib
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# One-time NLTK data (safe to call every start — no-ops if already present)
# ---------------------------------------------------------------------------
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Load the trained model + vectorizer (produced by notebooks/aasim_naivebayes_rnn.ipynb)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "aasim_naive_bayes.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "aasim_tfidf_vectorizer.joblib")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = Flask(__name__)


def clean_text(text: str) -> str:
    """Identical cleaning step used during training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_lemmatize(text: str) -> str:
    """Identical tokenization + stop-word removal + lemmatization used during training."""
    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def predict_sentiment(review_text: str):
    cleaned = clean_text(review_text)
    processed = tokenize_lemmatize(cleaned)

    if not processed.strip():
        return None

    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    classes = model.classes_
    confidence = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probabilities)}

    return {
        "sentiment": prediction,
        "confidence": confidence,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    review_text = (data.get("review_text") or "").strip()

    if not review_text:
        return jsonify({"error": "Please enter a review before submitting."}), 400

    result = predict_sentiment(review_text)
    if result is None:
        return jsonify({"error": "Review has no usable content after preprocessing."}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
