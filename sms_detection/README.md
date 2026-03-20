# 📱 SMS Spam Detector

Detects spam in SMS messages by benchmarking **three progressively powerful architectures**: a lightweight Dense model, a Bidirectional LSTM, and Google's **Universal Sentence Encoder** (Transfer Learning). Results are compared across Accuracy, Precision, Recall, and F1-Score.

---

## 🎯 Objective

Build and compare three NLP models for binary SMS classification (spam vs. ham), evaluating the trade-off between model complexity and performance.

---

## 📁 Project Structure

```
sms_detection/
├── sms_detection.ipynb   # Main notebook
└── spam.csv              # Dataset (encoding: latin-1)
```

---

## 🔬 Models

All three models share the same vectorization layer (`TextVectorization`) and are built with the Keras **Functional API**, allowing the vectorizer to be reused across models.

### Model 1 — Dense with Embeddings
```
Input (string) → TextVectorization → Embedding(128) → GlobalAveragePooling1D → Dense(32, relu) → Dense(1, sigmoid)
```
Lightweight and fast. No sequential memory — treats text as a bag of embedded words.

### Model 2 — Bidirectional LSTM
```
Input → TextVectorization → Embedding(128) → BiLSTM(64, return_sequences=True) → BiLSTM(64) → Flatten → Dropout(0.1) → Dense(32, relu) → Dense(1, sigmoid)
```
Captures word order and long-range context by reading the sequence both forwards and backwards.

### Model 3 — Universal Sentence Encoder (Transfer Learning)
```
Input (string) → USE (Google, 512-dim, frozen) → Dense(64, relu) → Dropout(0.2) → Dense(1, sigmoid)
```
Uses a pre-trained Google model. No Embedding or LSTM needed — the USE already encodes full sentences into 512-dimensional vectors.

---

## 💡 Key Concepts

### TextVectorization
```python
text_vec = TextVectorization(
    max_tokens=total_words_length,
    output_mode='int',
    output_sequence_length=avg_words_len  # Auto-padding
)
text_vec.adapt(X_train_np)  # Learns vocabulary from training data only
```

### GlobalAveragePooling1D vs Flatten
- **GAP** (Model 1): Averages all token embeddings → fewer parameters, less overfitting
- **Flatten** (Model 2): Keeps all values after BiLSTM → more parameters, captures more detail

### Bidirectional LSTM
Reads the sentence left→right and right→left. Useful because the end of a message often contextualizes its beginning (e.g., a suspicious call to action at the end of a seemingly normal message).

### Transfer Learning (`trainable=False`)
The USE weights are frozen — only the Dense layers on top are trained. This preserves the semantic knowledge the model learned from billions of sentences.

---

## 📊 Evaluation

```python
results = {
    'Dense Embedding'  : get_metrics(model_1, X_test_np, y_test_np),
    'Bi-LSTM'          : get_metrics(model_2, X_test_np, y_test_np),
    'Transfer Learning': get_metrics(model_3, X_test_np, y_test_np),
}
results_df = pd.DataFrame(results).transpose()
results_df.plot(kind='bar')
```

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
tensorflow-hub
pandas
numpy
scikit-learn
```
