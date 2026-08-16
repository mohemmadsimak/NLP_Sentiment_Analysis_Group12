"""
Auto-generated plain-Python version of aasim_naivebayes_rnn.ipynb
Run with:  python aasim_naivebayes_rnn.py
(Same code, same order as the notebook -- no Jupyter required. Figures that
the notebook displayed inline are saved as PNG files under
../reports/figures/aasim_naivebayes_rnn/ instead of popping up a window.)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- required for headless / script execution

import os as _os
_FIG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "reports", "figures", "aasim_naivebayes_rnn")
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
# ## Member 03 – A S M Aasim (CIT-24-01-0298)
# **ML Model:** Naive Bayes
# **DL Model:** RNN (SimpleRNN) for Text Classification
# 
# **Pipeline:** Dataset Exploration → Text Cleaning → Tokenization → Lemmatization → Word Embedding → Model Training → Evaluation → Visualization
# 

import sys, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from data_utils import load_dataset, get_split, RANDOM_STATE

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_recall_fscore_support)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

sns.set_style('whitegrid')
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ## 1. Dataset Exploration
# Analyzing the dataset's size, structure, class distribution, and data quality before
# preprocessing and modeling.

df = load_dataset('../data/1429_1.csv')
print('Shape:', df.shape)
print('\nMissing values per column:')
print(df.isna().sum())
print('\nRating distribution:')
print(df['rating'].value_counts().sort_index())
print(df.head())


fig, axes = plt.subplots(1, 2, figsize=(12,4))
df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='#3498db')
axes[0].set_title('Star Rating Distribution')
axes[0].set_xlabel('Rating (1-5)')

df['sentiment'].value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71','#95a5a6','#e74c3c'])
axes[1].set_title('Sentiment Class Distribution (target)')
plt.tight_layout()
_save_and_close_current_figure()
print(df['sentiment'].value_counts(normalize=True).round(3))


# **Insight:** Ratings are heavily skewed towards 4-5 stars, so the mapped Positive class
# dominates the dataset (~93%). This class imbalance — flagged in Section 6 of the project
# validation submission — is addressed below with `class_weight='balanced'` during model
# training and by reporting macro-averaged metrics (not just accuracy).

# ## 2. Text Cleaning
# Removing special characters, punctuation, URLs, numbers, and extra spaces.

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['review_text'].apply(clean_text)
print(df[['review_text','clean_text']].head(3))


# ## 3. Tokenization & Lemmatization

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


# ## 4. Word Frequency Analysis & Visualization (EDA)

from collections import Counter

pos_words = [w for toks in df.loc[df.sentiment=='Positive','tokens'] for w in toks]
neg_words = [w for toks in df.loc[df.sentiment=='Negative','tokens'] for w in toks]

fig, axes = plt.subplots(1, 2, figsize=(14,5))
WordCloud(width=500, height=350, background_color='white', colormap='Greens').generate(' '.join(pos_words[:200000])).to_image()
axes[0].imshow(WordCloud(width=500, height=350, background_color='white', colormap='Greens').generate(' '.join(pos_words[:200000])))
axes[0].axis('off'); axes[0].set_title('Positive Reviews – Word Cloud')

axes[1].imshow(WordCloud(width=500, height=350, background_color='white', colormap='Reds').generate(' '.join(neg_words[:200000])))
axes[1].axis('off'); axes[1].set_title('Negative Reviews – Word Cloud')
plt.tight_layout()
_save_and_close_current_figure()


review_lengths = df['tokens'].apply(len)
plt.figure(figsize=(8,4))
sns.histplot(review_lengths, bins=50, color='#9b59b6')
plt.title('Distribution of Review Length (tokens after cleaning)')
plt.xlabel('Number of tokens')
plt.xlim(0, 150)
_save_and_close_current_figure()
print(review_lengths.describe())


# ## 5. Train/Test Split

X_train, X_test, y_train, y_test = get_split(df, text_col='processed_text', label_col='sentiment')
print(X_train.shape, X_test.shape)


# ## 6. ML Model – Naive Bayes
# Naive Bayes is computationally efficient, simple to implement, and widely used as a
# baseline model for text classification tasks. TF-IDF features are used since
# `MultinomialNB` requires non-negative count-like features.

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb_model = MultinomialNB()
t0 = time.time()
nb_model.fit(X_train_tfidf, y_train)
print(f'Naive Bayes trained in {time.time()-t0:.1f}s')


y_pred_nb = nb_model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

cm = confusion_matrix(y_test, y_pred_nb, labels=['Negative','Neutral','Positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])
plt.title('Naive Bayes – Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
_save_and_close_current_figure()


# **Note:** Plain Naive Bayes (no class balancing) tends to collapse toward the majority
# Positive class on this imbalanced dataset. A `sample_weight`-balanced variant is
# trained below for a fairer comparison against the other members' balanced models.

from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight('balanced', y_train)
nb_balanced = MultinomialNB()
nb_balanced.fit(X_train_tfidf, y_train, sample_weight=sw)
y_pred_nb_bal = nb_balanced.predict(X_test_tfidf)
print('Balanced NB Accuracy:', accuracy_score(y_test, y_pred_nb_bal))
print(classification_report(y_test, y_pred_nb_bal))


# ## 7. Word Embedding + DL Model – SimpleRNN
# Word embeddings convert text into dense numerical vectors that capture semantic and
# contextual relationships between words, learned here inside the network's `Embedding`
# layer. RNNs can identify important local sequential patterns and phrases in text.

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


rnn_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rnn_model.summary()


cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train_int)
# Dampen extreme weights (sqrt) so gradient-based training doesn't over-correct
cw = np.sqrt(cw)
class_weight_dict = {i: w for i, w in enumerate(cw)}

history = rnn_model.fit(
    X_train_seq, y_train_cat,
    validation_split=0.1,
    epochs=4, batch_size=128,
    class_weight=class_weight_dict,
    verbose=2
)


plt.figure(figsize=(7,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('SimpleRNN Training Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
_save_and_close_current_figure()


y_pred_rnn_prob = rnn_model.predict(X_test_seq, verbose=0)
y_pred_rnn = np.argmax(y_pred_rnn_prob, axis=1)
y_pred_rnn_labels = [inv_label_map[i] for i in y_pred_rnn]

print('Accuracy:', accuracy_score(y_test, y_pred_rnn_labels))
print(classification_report(y_test, y_pred_rnn_labels))

cm2 = confusion_matrix(y_test, y_pred_rnn_labels, labels=['Negative','Neutral','Positive'])
plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'])
plt.title('SimpleRNN – Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
_save_and_close_current_figure()


# ## 8. Model Comparison – Naive Bayes (balanced) vs SimpleRNN

def macro_scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return accuracy_score(y_true, y_pred), p, r, f1

acc_nb, p_nb, r_nb, f1_nb = macro_scores(y_test, y_pred_nb_bal)
acc_rnn, p_rnn, r_rnn, f1_rnn = macro_scores(y_test, y_pred_rnn_labels)

comparison = pd.DataFrame({
    'Model': ['Naive Bayes (ML, balanced)', 'SimpleRNN (DL)'],
    'Accuracy': [acc_nb, acc_rnn],
    'Macro Precision': [p_nb, p_rnn],
    'Macro Recall': [r_nb, r_rnn],
    'Macro F1': [f1_nb, f1_rnn],
})
print(comparison)


comparison.set_index('Model')[['Accuracy','Macro F1']].plot(kind='bar', figsize=(7,4))
plt.title('Member 3 – Naive Bayes vs SimpleRNN')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
_save_and_close_current_figure()


# ## 9. Save Models

import joblib, os, pickle
os.makedirs('../models', exist_ok=True)
joblib.dump(nb_balanced, '../models/aasim_naive_bayes.joblib')
joblib.dump(tfidf, '../models/aasim_tfidf_vectorizer.joblib')
rnn_model.save('../models/aasim_rnn_model.keras')
with open('../models/aasim_keras_tokenizer.pkl', 'wb') as f:
    pickle.dump(keras_tok, f)
print('Saved: naive_bayes, tfidf_vectorizer, rnn_model, keras_tokenizer')


# ## Contribution Summary (Aasim)
# - Dataset exploration
# - Naive Bayes implementation
# - RNN development
# - Result visualization
# - Presentation preparation
# 
