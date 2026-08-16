"""
Auto-generated plain-Python version of premnath_random_forest_lstm.ipynb
Run with:  python premnath_random_forest_lstm.py
(Same code, same order as the notebook -- no Jupyter required. Figures that
the notebook displayed inline are saved as PNG files under
../reports/figures/premnath_random_forest_lstm/ instead of popping up a window.)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- required for headless / script execution

import os as _os
_FIG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "reports", "figures", "premnath_random_forest_lstm")
_os.makedirs(_FIG_DIR, exist_ok=True)
_fig_counter = {"n": 0}

def _save_and_close_current_figure():
    """Replacement for plt.show() in a non-interactive script."""
    import matplotlib.pyplot as _plt
    _fig_counter["n"] += 1
    path = _os.path.join(_FIG_DIR, f"fig_{_fig_counter['n']:02d}.png")
    _plt.savefig(path, bbox_inches="tight", dpi=110)
    _plt.close()
    print(f"[saved figure] {path}")


# # CCS3356 – Sentiment Analysis of Customer Reviews
# ## Member 01 – B Premnath (CIT-24-01-0241)
# **ML Model:** Random Forest Classifier (TF-IDF features)
# **DL Model:** LSTM (Long Short-Term Memory)
# 
# **Pipeline:** Data Collection → Data Cleaning → Tokenization → Stop Word Removal → Lemmatization → TF-IDF Feature Extraction → Model Training → Evaluation
# 

import sys, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_dataset, get_split, RANDOM_STATE

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

sns.set_style('whitegrid')
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ## 1. Data Collection

df = load_dataset('../data/1429_1.csv')
print('Shape:', df.shape)
print(df.head())


# ### Class Distribution

ax = df['sentiment'].value_counts().plot(kind='bar', color=['#2ecc71','#95a5a6','#e74c3c'])
plt.title('Sentiment Class Distribution (raw)')
plt.ylabel('Number of reviews')
_save_and_close_current_figure()
print(df['sentiment'].value_counts(normalize=True))


# ## 2. Data Cleaning
# Removing URLs, HTML tags, punctuation, digits, and extra whitespace; lowercasing all text.

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['review_text'].apply(clean_text)
print(df[['review_text', 'clean_text']].head(3))


# ## 3. Tokenization, Stop Word Removal & Lemmatization

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize_lemmatize(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    return tokens

t0 = time.time()
df['tokens'] = df['clean_text'].apply(tokenize_lemmatize)
df['processed_text'] = df['tokens'].apply(lambda toks: ' '.join(toks))
print(f'Preprocessing done in {time.time()-t0:.1f}s')
print(df[['processed_text']].head(3))


# ## 4. Word Frequency (EDA)

from collections import Counter
all_words = [w for toks in df['tokens'] for w in toks]
freq = Counter(all_words).most_common(20)
words, counts = zip(*freq)
plt.figure(figsize=(10,5))
sns.barplot(x=list(counts), y=list(words), color='#3498db')
plt.title('Top 20 Most Frequent Words (after cleaning)')
plt.xlabel('Frequency')
_save_and_close_current_figure()


# ## 5. Train/Test Split
# An 80/20 stratified split shared logic (via `data_utils.get_split`) so the ML and DL models
# below are evaluated on the same holdout reviews.

X_train, X_test, y_train, y_test = get_split(df, text_col='processed_text', label_col='sentiment')
print(X_train.shape, X_test.shape)
print(y_train.value_counts())


# ## 6. TF-IDF Feature Extraction

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print('TF-IDF & matrix shape:', X_train_tfidf.shape)


# ## 7. ML Model – Random Forest Classifier
# Random Forest is robust, handles large datasets effectively, reduces overfitting through
# ensemble learning, and performs well on text classification tasks.
# `class_weight='balanced'` is used to counter the strong Positive-class imbalance noted
# in the validation submission.

rf = RandomForestClassifier(
    n_estimators=200, max_depth=40, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)
t0 = time.time()
rf.fit(X_train_tfidf, y_train)
print(f'Random Forest trained in {time.time()-t0:.1f}s')


y_pred_rf = rf.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf, labels=['Negative','Neutral','Positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])
plt.title('Random Forest – Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
_save_and_close_current_figure()


# ## 8. DL Model – LSTM (Long Short-Term Memory)
# LSTM networks are specifically designed for sequence data and can capture contextual
# relationships within text, making them highly effective for sentiment analysis.

MAX_WORDS = 10000
MAX_LEN = 100

label_map = {'Negative':0, 'Neutral':1, 'Positive':2}
y_train_int = y_train.map(label_map).values
y_test_int = y_test.map(label_map).values
y_train_cat = to_categorical(y_train_int, num_classes=3)
y_test_cat = to_categorical(y_test_int, num_classes=3)

keras_tok = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
keras_tok.fit_on_texts(X_train)

X_train_seq = pad_sequences(keras_tok.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_seq = pad_sequences(keras_tok.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')
print(X_train_seq.shape, X_test_seq.shape)


lstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.summary()


from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train_int)
# Dampen extreme weights (sqrt) so gradient-based training doesn't over-correct
cw = np.sqrt(cw)
class_weight_dict = {i: w for i, w in enumerate(cw)}
print(class_weight_dict)

history = lstm_model.fit(
    X_train_seq, y_train_cat,
    validation_split=0.1,
    epochs=4, batch_size=128,
    class_weight=class_weight_dict,
    verbose=2
)


y_pred_lstm_prob = lstm_model.predict(X_test_seq, verbose=0)
y_pred_lstm = np.argmax(y_pred_lstm_prob, axis=1)
inv_label_map = {v:k for k,v in label_map.items()}
y_pred_lstm_labels = [inv_label_map[i] for i in y_pred_lstm]

print('Accuracy:', accuracy_score(y_test, y_pred_lstm_labels))
print(classification_report(y_test, y_pred_lstm_labels))

cm2 = confusion_matrix(y_test, y_pred_lstm_labels, labels=['Negative','Neutral','Positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])
plt.title('LSTM – Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
_save_and_close_current_figure()


# ## 9. Model Comparison – Random Forest vs LSTM

def macro_scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return accuracy_score(y_true, y_pred), p, r, f1

acc_rf, p_rf, r_rf, f1_rf = macro_scores(y_test, y_pred_rf)
acc_lstm, p_lstm, r_lstm, f1_lstm = macro_scores(y_test, y_pred_lstm_labels)

comparison = pd.DataFrame({
    'Model': ['Random Forest (ML)', 'LSTM (DL)'],
    'Accuracy': [acc_rf, acc_lstm],
    'Macro Precision': [p_rf, p_lstm],
    'Macro Recall': [r_rf, r_lstm],
    'Macro F1': [f1_rf, f1_lstm],
})
print(comparison)


comparison.set_index('Model')[['Accuracy','Macro F1']].plot(kind='bar', figsize=(7,4))
plt.title('Member 1 – Random Forest vs LSTM')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
_save_and_close_current_figure()


# ## 10. Save Models

import joblib, os
os.makedirs('../models', exist_ok=True)
joblib.dump(rf, '../models/premnath_random_forest.joblib')
joblib.dump(tfidf, '../models/premnath_tfidf_vectorizer.joblib')
lstm_model.save('../models/premnath_lstm_model.keras')
import pickle
with open('../models/premnath_keras_tokenizer.pkl', 'wb') as f:
    pickle.dump(keras_tok, f)
print('Saved: random_forest, tfidf_vectorizer, lstm_model, keras_tokenizer')


# ## Contribution Summary (Premnath)
# - Data preprocessing (cleaning, tokenization, stop-word removal, lemmatization)
# - TF-IDF feature extraction
# - Random Forest implementation
# - LSTM implementation
# - Performance evaluation
# - Documentation support
# 
