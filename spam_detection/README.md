# 📧 Email Spam Detector

Classifies emails as spam or ham using an **LSTM-based deep learning model**. Features full text preprocessing (punctuation removal, stopword filtering), class balancing via undersampling, WordCloud visualization, and training safety via callbacks.

---

## 🎯 Objective

Build a production-aware spam classifier for email data, applying NLP preprocessing best practices and using callbacks to prevent overfitting during training.

---

## 📁 Project Structure

```
spam_detection/
├── spam_detection.ipynb   # Main notebook
├── emails.csv             # Dataset
└── requirements.txt       # Generated with pip freeze
```

---

## 🔬 Approach

### 1. Data Preparation

**Class balancing:** The dataset contains more ham than spam. Undersampling equalizes the classes before training, forcing the model to learn actual spam patterns rather than exploit class frequency.

```python
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)
```

**Text cleaning:**
```python
# Remove punctuation
balanced_data['text'] = balanced_data['text'].apply(remove_punctuations)

# Remove stopwords (NLTK)
balanced_data['text'] = balanced_data['text'].apply(remove_stopwords)
```

**WordCloud inspection:** Visualizes the most frequent words in spam vs. ham — a sanity check before training.

### 2. Tokenization & Padding

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

train_sequences = tokenizer.texts_to_sequences(train_X)
train_sequences = pad_sequences(train_sequences, maxlen=100,
                                padding='post', truncating='post')
```

All sequences are padded/truncated to exactly 100 tokens.

### 3. Model

```python
model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=100),
    LSTM(16),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss=BinaryCrossentropy(from_logits=True),
              optimizer='adam', metrics=['accuracy'])
```

### 4. Callbacks

```python
es = EarlyStopping(patience=3, monitor='val_accuracy',
                   restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5)

history = model.fit(train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=20, batch_size=32, callbacks=[lr, es])
```

`EarlyStopping` halts training when validation accuracy plateaus, restoring the best weights. `ReduceLROnPlateau` halves the learning rate when the loss stagnates.

---

## 💡 Key Concepts

### Undersampling vs Oversampling
Undersampling reduces the majority class to match the minority. This is simple and avoids generating synthetic data, but discards real samples. Oversampling (e.g., SMOTE) creates synthetic minority samples but can introduce noise.

### Padding Strategy
`padding='post'` adds zeros at the end of short sequences; `truncating='post'` removes tokens from the end of long ones. This keeps the most informative content (the beginning of an email) intact.

### Training Curve
```python
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
```
A growing gap between training and validation accuracy signals **overfitting** — the model is memorizing, not generalizing.

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
pandas
numpy
nltk
wordcloud
scikit-learn
matplotlib
seaborn
```
