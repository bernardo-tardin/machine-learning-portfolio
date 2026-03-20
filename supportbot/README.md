# 🤖 Support Ticket Classifier

Automatically categorizes customer support tickets into one of four types: **Refund request**, **Technical issue**, **Cancellation request**, or **Product inquiry**. Compares a classical **Naive Bayes** baseline against a deep learning **Stacked Bidirectional LSTM**.

---

## 🎯 Objective

Route support tickets to the correct team automatically, benchmarking a fast probabilistic model against a contextual deep learning model on a multiclass classification problem.

---

## 📁 Project Structure

```
supportbot/
├── supportbot.ipynb                  # Main notebook
└── customer_support_tickets.csv      # Dataset
```

---

## 🔬 Approach

### 1. Data Preparation

**Feature engineering:** Subject and description are concatenated into a single text field.

```python
filtered_df['text'] = filtered_df['ticket_sub'].fillna('') + " " + filtered_df['ticket_text']
```

**Regex cleaning:**
```python
def clean(text):
    text = text.lower()
    text = re.sub(r'\{.*?\}', '', text)   # Remove {template placeholders}
    text = re.sub(r'\n', ' ', text)        # Flatten newlines
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Class balancing:** The dataset has 5 categories with unequal sizes. All categories are downsampled to match the smallest class (`Billing inquiry`).

```python
balanced_refund = refund.sample(n=len(billing), random_state=42)
# ... repeated for each category
new_df = pd.concat([balanced_refund, balanced_technical,
                    balanced_cancellation, balanced_product])
```

> ℹ️ `Billing inquiry` is excluded from the final model to keep the problem balanced and tractable.

### 2. Model A — Multinomial Naive Bayes (Baseline)

```python
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_NB)
X_test_vec  = vectorizer.transform(X_test_NB)  # No refit — prevents leakage

model = MultinomialNB()
model.fit(X_train_vec, y_train_NB)
```

### 3. Model B — Stacked Bidirectional LSTM (Deep Learning)

```python
# Labels encoded as integers for sparse_categorical_crossentropy
new_df['label_enc'] = new_df['label'].map({
    'Refund request': 0, 'Technical issue': 1,
    'Cancellation request': 2, 'Product inquiry': 3
})

# Architecture
x = text_vec(input_layer)
x = layers.Embedding(input_dim=vocab_size, output_dim=128)(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)  # Level 1
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)  # Level 2
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(4, activation='softmax')(x)   # 4 classes
```

**Softmax** distributes output as probabilities across all 4 classes, summing to 1.0.

---

## 💡 Key Concepts

### Stacked BiLSTM
Two Bidirectional LSTM layers process the sequence hierarchically: the first captures low-level patterns (word pairs, phrases); the second captures higher-level patterns (sentence meaning, topic). `return_sequences=True` is required on both to pass the full sequence between layers.

### `GlobalAveragePooling1D` after BiLSTM
Averages all timestep vectors into a single fixed-size vector. More parameter-efficient than `Flatten`, reducing overfitting risk on longer sequences.

### Multiclass vs Binary
- Binary: `Dense(1, activation='sigmoid')` + `binary_crossentropy`
- Multiclass: `Dense(N, activation='softmax')` + `sparse_categorical_crossentropy`

```python
# Multiclass prediction
y_preds = model.predict(X_test_DL_np)
y_preds = np.argmax(y_preds, axis=1)  # [0.1, 0.7, 0.1, 0.1] → 1
```

---

## 📊 Evaluation

Both models use a **4×4 confusion matrix** to reveal which ticket types are confused with each other — the most actionable insight for improving the classifier.

```python
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
```

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```
