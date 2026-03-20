# 📊 Handwritten Digit Classifier from CSV (TensorFlow/Keras)

Classifies handwritten digits from the **MNIST dataset in CSV format** — as commonly found in Kaggle competitions — applying a full tabular-to-image preprocessing pipeline before training a Dense Neural Network.

---

## 🎯 Objective

Build a complete preprocessing pipeline that handles a raw pixel CSV, applies defensive data cleaning, reshapes to image format, and trains a Dense NN with proper train/validation split and training curve visualization.

---

## 📁 Project Structure

```
neural_network_handwritten_digits/
├── neural_network_handwritten_digits.ipynb   # Main notebook
├── Train.csv                                 # Labeled dataset (label + 784 pixel columns)
└── test.csv                                  # Unlabeled test set (Kaggle submission format)
```

---

## 🔬 Approach

### 1. Load and Parse CSV

```python
train_data = pd.read_csv('Train.csv')

X = train_data.iloc[:, 1:]   # 784 pixel columns (28×28)
Y = train_data.iloc[:, 0]    # Label column: digit 0–9
```

### 2. Defensive Cleaning

```python
# Coerce non-numeric values to NaN, fill with 0
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
```

### 3. Normalize and Reshape

```python
X = X.values / 255.0          # [0, 255] → [0, 1]
X = X.reshape(-1, 28, 28, 1)  # (n_samples, height, width, channels)
```

The `-1` in reshape is inferred automatically from the total number of samples. The trailing `1` represents the grayscale channel count.

### 4. One-Hot Encode Labels

```python
y = to_categorical(Y, num_classes=10)
# Example: label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

### 5. Train/Validation Split

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. Model

```python
model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_val, y_val))
```

### 7. Inference on Unlabeled Test Set

```python
X_test = pd.read_csv('test.csv').values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
predictions    = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
```

---

## 💡 Key Concepts

### `to_categorical` vs `sparse_categorical_crossentropy`

Two equivalent approaches for multiclass classification:

| Approach | Label format | Loss function |
|----------|-------------|---------------|
| `to_categorical` (used here) | One-hot vector `[0,0,1,0,...]` | `categorical_crossentropy` |
| Raw integers | Integer `2` | `sparse_categorical_crossentropy` |

Both produce the same results. One-hot is more explicit about what the model outputs; sparse is simpler to set up.

### Defensive Cleaning with `errors='coerce'`
`pd.to_numeric(errors='coerce')` converts any non-numeric value to `NaN` instead of raising an error — a robust pattern for real-world CSV data that may contain corrupted cells.

### Training Curve Visualization

```python
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

A growing gap between training and validation accuracy signals overfitting. A flat validation curve from early on signals underfitting or a learning rate problem.

### Kaggle Format: Two Separate Files
`Train.csv` has labels (first column); `test.csv` has only pixel columns (no labels). This is the standard Kaggle competition format — the model is evaluated by submitting predicted labels for the unlabeled test set.

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
pandas
numpy
scikit-learn
matplotlib
```
