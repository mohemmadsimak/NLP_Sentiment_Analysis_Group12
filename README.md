# NLP_Group_12 — Sentiment Analysis of Customer Reviews Using ML and DL

**Module:** CCS3356 – Natural Language Processing, Sri Lanka Technology Campus
**Group:** Logic Legends (Group_12)

## Group Members
| # | Student ID | Name | ML Model | DL Model |
|---|---|---|---|---|
| 1 | CIT-24-01-0241 | B Premnath | Random Forest | LSTM |
| 2 | CIT-24-01-0010 | M S M Simak | SVM | GRU |
| 3 | CIT-24-01-0298 | A S M Aasim | Naive_Bayes | SimpleRNN |

## Problem Statement
Automatically classify the sentiment (**Positive / Neutral / Negative**) of customer
product reviews, removing the need to read and triage feedback manually. See
`reports/Group_12_Project_Validation_Submission.pdf` for the full problem rationale,
target users, and value proposition.

## Dataset
- **Name:** Datafiniti Amazon Consumer Reviews of Amazon Products
- **Source:** https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazonproducts
- **File:** `data/1429_1.csv` (34,626 reviews after cleaning/deduplication)
- **Label construction:** `reviews.rating` (1–5 stars) → sentiment:
  - 1–2 stars → **Negative**
  - 3 stars → **Neutral**
  - 4–5 stars → **Positive**
- **Known challenge:** the dataset is heavily imbalanced (~93% Positive, ~4% Neutral,
  ~2% Negative) since most reviewers only leave feedback when highly satisfied. All
  models below use class-balancing strategies to compensate — see `reports/model_comparison.md`.

## Setup Instructions
```bash
git clone <repo-url>
cd NLP_Group_12
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
The notebooks call `nltk.download(...)` for `stopwords`, `wordnet`, `omw-1.4`,
`punkt`, and `punkt_tab` on first run — an internet connection is required once.

## How to Run the Project

### Option A — Plain Python scripts (fastest, no Jupyter needed)
Each notebook has an identical, ready-to-run `.py` version in `src/`. These run
noticeably faster than the notebooks (no kernel/browser overhead) and save every plot
as a PNG under `reports/figures/<script_name>/` instead of displaying it inline:

```bash
cd src
python premnath_random_forest_lstm.py   # Random Forest + LSTM
python simak_svm_gru.py                 # SVM + GRU
python aasim_naivebayes_rnn.py          # Naive Bayes + SimpleRNN
```
Each takes roughly 1–2 minutes on CPU. Trained models are saved to `models/` and
figures to `reports/figures/` exactly as when run from the notebook.

### Option B — Jupyter notebooks (for step-by-step / viva walkthroughs)
```bash
cd notebooks
jupyter notebook premnath_random_forest_lstm.ipynb   # Random Forest + LSTM
jupyter notebook simak_svm_gru.ipynb                 # SVM + GRU
jupyter notebook aasim_naivebayes_rnn.ipynb          # Naive Bayes + SimpleRNN
```
Or run headlessly end-to-end:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
Both options import `src/data_utils.py`, which loads and cleans the raw CSV and
provides an identical stratified 80/20 train/test split so every member's models are
evaluated on the same holdout reviews.

## Repository Structure
```
project-root/
│
├── data/              # Raw dataset (1429_1.csv)
├── notebooks/          # One notebook per member (preprocessing → model → evaluation)
├── src/                 # Shared data_utils.py + plain-Python .py version of each notebook
├── models/             # Saved trained models (.joblib / .keras / .pkl)
├── app/                 # Final integrated web application (Flask)
│   ├── app.py
│   ├── templates/index.html
│   └── static/style.css
├── reports/             # Validation submission PDF + model comparison report
├── screenshots/         # Git repo / branch screenshots for submission
├── videos/               # Progress / demo video
├── requirements.txt
├── README.md
└── .gitignore
```

## Final Integrated Application
`app/` is a small Flask web app implementing the workflow from Section 5, Q10 of the
validation submission (user input → preprocessing → TF-IDF → model → sentiment label +
confidence scores). It loads the already-trained, saved **Naive Bayes (balanced)**
model — the group's recommended model per `reports/model_comparison.md` — directly from
`models/`, so no retraining is needed to run it.

```bash
cd app
python app.py
```
Then open **http://127.0.0.1:5000** in a browser, paste a review, and click
"Analyze Sentiment". The page returns the predicted label (Positive/Neutral/Negative)
plus a confidence bar for each class.

## Model Summary
| Member | ML Model | ML Feature Extraction | DL Model | Text Normalization |
|---|---|---|---|---|
| Premnath | Random Forest | TF-IDF (1–2 grams) | LSTM | Lemmatization |
| Simak | SVM (LinearSVC) | Bag-of-Words (CountVectorizer) | GRU | Porter Stemming |
| Aasim | Naive Bayes | TF-IDF (1–2 grams) | SimpleRNN | Lemmatization |

## Results Summary
See `reports/model_comparison.md` for the full write-up. Headline numbers (macro-averaged
to account for class imbalance):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.915 | 0.402 |
| SVM | 0.935 | 0.382 |
| Naive Bayes (balanced) | 0.770 | **0.443** |
| LSTM | 0.933 | 0.322 |
| GRU | 0.933 | 0.322 |
| SimpleRNN | 0.933 | 0.322 |

**Naive Bayes with balanced sample weighting gave the best macro F1**, i.e. the most
even performance across Positive, Neutral, and Negative classes, at the cost of overall
accuracy. The three DL models converged to majority-class (Positive) predictions given
the severe class imbalance and limited training (4 epochs, no pretrained embeddings) —
a finding discussed further in `reports/model_comparison.md` and tied to the bias risks
already flagged in Section 7 of the validation submission.

## Ethics & Responsible AI
See Section 7 of `reports/Group_12_Project_Validation_Submission.pdf` and
`reports/model_comparison.md` for the discussion of dataset bias, harmful/misleading
outputs, mitigation strategies, and limitations (sarcasm, context-dependent meaning,
short reviews, unseen vocabulary).
