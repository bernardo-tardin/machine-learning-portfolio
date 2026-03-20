# ✍️ Handwritten Digit Classifier (MNIST)

Classifies handwritten digits (0–9) from the MNIST dataset using TensorFlow/Keras. Compares two architectures — a minimal baseline and a deeper normalized network — to illustrate the impact of depth and data normalization.

---

## 🎯 Objective

Build and compare two neural network models on MNIST to understand the effect of:
- Network depth (1 layer vs 3 layers)
- Data normalization (raw pixels vs normalized 0–1)
- Optimizer choice (SGD vs Adam)

---

## 📁 Project Structure

```
classify_handwritten_digits/
├── classify_handwritten_digits.ipynb   # Main notebook
└── epic_num_reader.h5                  # Saved deep model
```

---

## 🔬 Approach

### Model 1 — Baseline (SGD, no normalization)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=10, epochs=10)
```

### Model 2 — Deep Network (Adam + normalization)

```python
# Normalize pixels from [0, 255] → [0, 1]
x_train_dl = tf.keras.utils.normalize(x_train, axis=1)

model_dl = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_dl.fit(x_train_dl, y_train, epochs=3)
```

---

## 💡 Key Concepts

### Why Normalize?
Neural networks train via gradient descent. Pixel values between 0–255 produce large gradients, causing unstable or slow convergence. Normalizing to 0–1 ensures gradients flow evenly across all layers.

### Softmax Output
The final layer uses `softmax`, which distributes output as probabilities summing to 1.0 across all 10 digit classes. `np.argmax(prediction)` extracts the predicted digit.

### Sparse Categorical Crossentropy
Used when class labels are integers (0, 1, 2... 9) rather than one-hot encoded vectors.

---

## 📊 Comparison

| Feature | Model 1 (Baseline) | Model 2 (Deep) |
|---------|-------------------|----------------|
| Layers | Flatten → Dense(10) | Flatten → Dense(128)×2 → Dense(10) |
| Optimizer | SGD | Adam |
| Normalization | None | `tf.keras.utils.normalize` |
| Epochs | 10 | 3 |

---

## 💾 Model Persistence

```python
# Save
model_dl.save('epic_num_reader.h5')

# Load and predict
new_model = tf.keras.models.load_model('epic_num_reader.h5')
predictions = new_model.predict([x_test_dl])
print(np.argmax(predictions[3]))  # Predicted digit
```

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
numpy
matplotlib
```
