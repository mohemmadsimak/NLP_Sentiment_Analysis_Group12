# Section 4 — Model Comparison Plan: Results

**Group 12 — Logic Legends** | CCS3356 Natural Language Processing
Dataset: 34,626 Amazon customer reviews (after cleaning/deduplication) | 80/20 stratified split | Test set: 6,926 reviews (162 Negative, 300 Neutral, 6,464 Positive)

## 1. All Six Models — Head-to-Head

| Member | Model | Type | Feature Extraction | Accuracy | Macro Precision | Macro Recall | Macro F1 | Train Time |
|---|---|---|---|---|---|---|---|---|
| Premnath | Random Forest | ML | TF-IDF (1–2gram) | 0.915 | 0.504 | 0.386 | 0.402 | 11.5s |
| Simak | SVM (LinearSVC) | ML | Bag-of-Words | 0.935 | 0.812 | 0.366 | 0.382 | 51.3s |
| Aasim | Naive Bayes (unbalanced) | ML | TF-IDF (1–2gram) | 0.933 | 0.310 | 0.333 | 0.321 | <0.1s |
| Aasim | Naive Bayes (balanced) | ML | TF-IDF (1–2gram) | 0.770 | 0.423 | **0.626** | **0.443** | <0.1s |
| Premnath | LSTM | DL | Embedding (learned) | 0.933 | 0.311 | 0.333 | 0.322 | ~2 min |
| Simak | GRU | DL | Embedding (learned) | 0.933 | 0.311 | 0.333 | 0.322 | ~2 min |
| Aasim | SimpleRNN | DL | Embedding (learned) | 0.933 | 0.311 | 0.333 | 0.322 | ~2 min |

*(Accuracy alone is misleading here — always predicting "Positive" for every review
already scores ~93.3% accuracy given the class distribution, which is exactly what the
0.933/0.311-macro-F1 rows above reduce to. Macro-averaged metrics, which weight all
three classes equally regardless of size, are the metrics the group is treating as
decisive — matching the justification given in Q8 of the validation submission.)*

## 2. Best-Performing Model: Naive Bayes (balanced sample weights)

**Balanced Naive Bayes is the group's recommended model for the final application**,
because it is the only model that meaningfully identifies Negative and Neutral reviews
instead of defaulting to the majority class:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.16 | 0.60 | 0.25 |
| Neutral | 0.13 | 0.49 | 0.21 |
| Positive | 0.98 | 0.79 | 0.87 |

Trade-off: overall accuracy drops to 77% because the model now over-predicts the
minority classes on some genuinely-positive reviews (precision on Negative/Neutral is
low), but it recovers 60% of true Negative reviews and 49% of true Neutral reviews —
exactly the reviews a business would most want surfaced, since these are the ones
signalling problems. Random Forest and SVM keep headline accuracy higher (91–94%) but
recall almost no Negative/Neutral reviews (6–13% recall) — they are, in effect, "always
predict Positive" classifiers with a thin veneer of minority-class detection.

## 3. Why the Deep Learning Models Underperformed

All three sequence models (LSTM, GRU, SimpleRNN) converged to predicting **Positive for
every review** in the test set (Negative/Neutral recall = 0.00). This is a genuine,
reportable result, not a bug, and stems from three compounding factors:

1. **Severe class imbalance** — Positive reviews outnumber Negative reviews ~40:1 in
   this dataset (see Section 6 of the validation submission, "Class Imbalance").
   Gradient-based training with cross-entropy loss finds the global loss minimum by
   defaulting to the majority class unless imbalance is corrected carefully.
2. **Class weighting was capped, not removed.** An initial run using scikit-learn's
   raw `compute_class_weight('balanced')` values (≈14× for Negative, ≈8× for Neutral)
   caused the opposite failure — the networks over-corrected and collapsed to ~4%
   accuracy, predicting almost exclusively Negative/Neutral. Square-root-dampened
   weights (≈3.8× / 2.8× / 0.6×) fixed that collapse but were still not enough, in
   just 4 training epochs with no pretrained embeddings, to teach the network the
   subtler lexical cues (e.g. "stopped working after one week") that distinguish
   Negative reviews from Positive ones — the exact failure mode anticipated in Q14 of
   the validation submission.
3. **Learned (not pretrained) embeddings + short training.** Each DL model trains its
   `Embedding` layer from scratch on this dataset alone, for only 4 epochs. TF-IDF/
   Bag-of-Words fed into Naive Bayes and SVM instead give the ML models direct access
   to distinctive minority-class vocabulary from the first training pass.

**Conclusion:** on this specific, heavily-imbalanced dataset with limited training
budget, the simpler ML models — especially Naive Bayes with sample-weight balancing —
outperformed the deep learning models on the classes that matter most for the intended
use case (surfacing dissatisfied customers). This nuance is included in the final
report/presentation rather than only quoting the misleadingly-high accuracy figures.

## 4. Recommendation for the Final Integrated Application

- **Deploy:** Naive Bayes (balanced) + TF-IDF, for its substantially better minority-
  class recall.
- **Future improvement path (documented as a limitation, not implemented in this
  version):** pretrained word embeddings (GloVe/Word2Vec) or a BERT-based encoder,
  oversampling (SMOTE) or undersampling the Positive class, and longer DL training,
  would likely close the gap for the sequence models — noted in Section 7, Q16 of the
  validation submission ("Highly domain-specific terminology... New vocabulary not
  present during training").
