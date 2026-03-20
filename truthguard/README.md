# 🛡️ TruthGuard — Fake News Detector

Detects fake news articles by benchmarking three NLP architectures: a **Dense embedding model**, a **Bidirectional LSTM**, and Google's **Universal Sentence Encoder**. Evaluates each model on Accuracy, Precision, Recall, and F1-Score.

---

## 🎯 Objective

Classify news articles as real or fake by comparing lightweight and powerful NLP architectures, providing a clear performance benchmark across four evaluation metrics.

---

## 📁 Project Structure

```
truthguard_by_text/
├── truthguard_by_text.ipynb   # Main notebook
├── True.csv                   # Real news articles
└── Fake.csv                   # Fake news articles
```

---

## 🔬 Approach

### 1. Data Preparation

Two separate CSV files are merged and labeled:

```python
df_true['label'] = 1   # Real news
df_fake['label'] = 0   # Fake news
df = pd.concat([df_true, df_fake], ignore_index=True)
```

**Class balancing:** The dataset is balanced by downsampling the majority class.

```python
fake_news_balanced = fake_news.sample(n=len(true_news), random_state=42)
df = pd.concat([true_news, fake_news_balanced]).reset_index(drop=True)
```

**Stopword removal and vectorization:**
```python
text_vec = TextVectorization(
    max_tokens=10000,
    standardize='strip_punctuation',
    output_mode='int',
    output_sequence_length=average_text_len
)
text_vec.adapt(X_train_np)  # Vocabulary learned from training data only
```

### 2. Model 1 — Dense with Embeddings

```
Input → TextVectorization → Embedding(10000, 128) → GlobalAveragePooling1D
      → Dense(32, relu) → Dense(1, sigmoid)
```

### 3. Model 2 — Bidirectional LSTM

```
Input → TextVectorization → Embedding(10000, 128)
      → BiLSTM(64, return_sequences=True) → BiLSTM(64)
      → Flatten → Dropout(0.1) → Dense(32, relu) → Dense(1, sigmoid)
```

### 4. Model 3 — Universal Sentence Encoder (Transfer Learning)

```
Input (string) → USE (Google, frozen, 512-dim)
              → Dense(64, relu) → Dropout(0.2) → Dense(1, sigmoid)
```

All three models share the same `compile_and_fit` helper and are evaluated with `get_metrics`.

---

## 💡 Key Concepts

### Why `max_tokens=10000`?
News articles have a large vocabulary. Capping at 10,000 tokens keeps the embedding matrix manageable while covering the most informative words. Rare words (typically noise) are discarded.

### Flatten vs GlobalAveragePooling1D
- **Model 2 uses `Flatten`** after BiLSTM: preserves positional detail, more parameters
- **Model 1 uses `GAP`**: averages across all timesteps, more regularized

Both approaches are compared to show their effect on final performance.

### USE Architecture Difference
Models 1 and 2 require a `TextVectorization` + `Embedding` pipeline. Model 3 skips both — the USE directly encodes raw strings into 512-dimensional vectors using a Transformer with Attention Mechanism internally.

---

## 📊 Evaluation

```python
results = {
    'Dense Embedding'  : get_metrics(model1, X_test_np, y_test_np),
    'Bi-LSTM'          : get_metrics(model2, X_test_np, y_test_np),
    'Transfer Learning': get_metrics(model3, X_test_np, y_test_np),
}
results_df = pd.DataFrame(results).transpose()
print(results_df)
```

| Metric | Dense | BiLSTM | USE |
|--------|-------|--------|-----|
| Accuracy | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1-Score | — | — | — |

*Run the notebook to populate results.*

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
tensorflow-hub
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```
