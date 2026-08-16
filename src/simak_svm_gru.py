"""
Auto-generated plain-Python version of simak_svm_gru.ipynb
Run with:  python simak_svm_gru.py
(Same code, same order as the notebook -- no Jupyter required. Figures that
the notebook displayed inline are saved as PNG files under
../reports/figures/simak_svm_gru/ instead of popping up a window.)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- required for headless / script execution

import os as _os
_FIG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "reports", "figures", "simak_svm_gru")
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
# ## Member 02 – M S M Simak (CIT-24-01-0010)
# **ML Model:** Support Vector Machine (SVM)
# **DL Model:** GRU (Gated Recurrent Unit)
# 
# **Pipeline:** Data Cleaning → Lowercase Conversion → Tokenization → Stop Word Removal → Stemming → Vectorization → Model Training → Evaluation
# 

import sys, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_dataset, get_split, RANDOM_STATE

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_recall_fscore_support)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

sns.set_style('whitegrid')
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ## 1. Data Collection

df = load_dataset('../data/1429_1.csv')
print('Shape:', df.shape)
print(df.head())


# ## 2. Data Cleaning & Lowercase Conversion
# Removes noise, missing values, special characters, and irrelevant information; converts
# all text to lowercase to ensure consistency and prevent duplicate word representations.

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['review_text'].apply(clean_text)
print(df[['review_text', 'clean_text']].head(3))


# ## 3. Tokenization, Stop Word Removal & Stemming
# Stemming (Porter Stemmer) reduces words to their root forms, allowing similar words to
# be treated as a single feature — different from Member 1's lemmatization approach.

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize_stem(text):
    tokens = text.split()
    tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    return tokens

t0 = time.time()
df['tokens'] = df['clean_text'].apply(tokenize_stem)
df['processed_text'] = df['tokens'].apply(lambda toks: ' '.join(toks))
print(f'Preprocessing done in {time.time()-t0:.1f}s')
print(df[['processed_text']].head(3))


# ### Class Distribution (EDA)

df['sentiment'].value_counts().plot(kind='bar', color=['#2ecc71','#95a5a6','#e74c3c'])
plt.title('Sentiment Class Distribution')
plt.ylabel('Number of reviews')
_save_and_close_current_figure()


# ## 4. Train/Test Split

X_train, X_test, y_train, y_test = get_split(df, text_col='processed_text', label_col='sentiment')
print(X_train.shape, X_test.shape)


# ## 5. Vectorization (Bag-of-Words / CountVectorizer)
# Transforms textual data into numerical representations that machine learning algorithms
# can understand. A Bag-of-Words count representation is used here (distinct from
# Member 1's TF-IDF weighting) to give the group a comparison across feature-extraction
# strategies as well as models.

vectorizer = CountVectorizer(max_features=8000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print('Vectorized matrix shape:', X_train_vec.shape)


# ## 6. ML Model – Support Vector Machine (SVM)
# SVM performs exceptionally well in high-dimensional text classification problems and has
# proven effectiveness in sentiment analysis applications. `LinearSVC` is used for
# efficiency on a high-dimensional sparse feature matrix, wrapped with
# `CalibratedClassifierCV` so we can also report probability-based metrics.

svm = CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000))
t0 = time.time()
svm.fit(X_train_vec, y_train)
print(f'SVM trained in {time.time()-t0:.1f}s')


y_pred_svm = svm.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

cm = confusion_matrix(y_test, y_pred_svm, labels=['Negative','Neutral','Positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])
plt.title('SVM – Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
_save_and_close_current_figure()


# ## 7. DL Model – GRU (Gated Recurrent Unit)
# GRU offers performance similar to LSTM while requiring fewer parameters and shorter
# training times, thanks to its simplified gating mechanism (reset and update gates only).

MAX_WORDS = 10000
MAX_LEN = 100

label_map = {'Negative':0, 'Neutral':1, 'Positive':2}
inv_label_map = {v:k for k,v in label_map.items()}
y_train_int = y_train.map(label_map).values
y_test_int = y_test.map(label_map).values
y_train_cat = to_categorical(y_train_int, num_classes=3)
y_test_cat = to_categorical(y_test_int, num_classes=3)

keras_tok = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
keras_tok.fit_on_texts(X_train)
X_train_seq = pad_sequences(keras_tok.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_seq = pad_sequences(keras_tok.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')
print(X_train_seq.shape, X_test_seq.shape)


gru_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    GRU(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
gru_model.summary()


cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train_int)
# Dampen extreme weights (sqrt) so gradient-based training doesn't over-correct
cw = np.sqrt(cw)
class_weight_dict = {i: w for i, w in enumerate(cw)}

history = gru_model.fit(
    X_train_seq, y_train_cat,
    validation_split=0.1,
    epochs=4, batch_size=128,
    class_weight=class_weight_dict,
    verbose=2
)


y_pred_gru_prob = gru_model.predict(X_test_seq, verbose=0)
y_pred_gru = np.argmax(y_pred_gru_prob, axis=1)
y_pred_gru_labels = [inv_label_map[i] for i in y_pred_gru]

print('Accuracy:', accuracy_score(y_test, y_pred_gru_labels))
print(classification_report(y_test, y_pred_gru_labels))

cm2 = confusion_matrix(y_test, y_pred_gru_labels, labels=['Negative','Neutral','Positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])
plt.title('GRU – Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
_save_and_close_current_figure()


# ## 8. Model Comparison – SVM vs GRU

def macro_scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return accuracy_score(y_true, y_pred), p, r, f1

acc_svm, p_svm, r_svm, f1_svm = macro_scores(y_test, y_pred_svm)
acc_gru, p_gru, r_gru, f1_gru = macro_scores(y_test, y_pred_gru_labels)

comparison = pd.DataFrame({
    'Model': ['SVM (ML)', 'GRU (DL)'],
    'Accuracy': [acc_svm, acc_gru],
    'Macro Precision': [p_svm, p_gru],
    'Macro Recall': [r_svm, r_gru],
    'Macro F1': [f1_svm, f1_gru],
})
print(comparison)


comparison.set_index('Model')[['Accuracy','Macro F1']].plot(kind='bar', figsize=(7,4))
plt.title('Member 2 – SVM vs GRU')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
_save_and_close_current_figure()


# ## 9. Save Models

import joblib, os, pickle
os.makedirs('../models', exist_ok=True)
joblib.dump(svm, '../models/simak_svm.joblib')
joblib.dump(vectorizer, '../models/simak_count_vectorizer.joblib')
gru_model.save('../models/simak_gru_model.keras')
with open('../models/simak_keras_tokenizer.pkl', 'wb') as f:
    pickle.dump(keras_tok, f)
print('Saved: svm, count_vectorizer, gru_model, keras_tokenizer')


# ## Contribution Summary (Simak)
# - Data cleaning
# - SVM implementation
# - GRU model development
# - Experimental analysis
# - Report writing
# 
