# 🍷 Wine Type Classification (Keras DNN)

Classifies wine variants (Red vs. White) based on their physicochemical properties using a **Deep Neural Network (DNN)** built with Keras/TensorFlow. Introduces binary classification, data merging, exploratory data analysis (EDA), and sequential model building.

---

## 🎯 Objective

Given a set of chemical attributes (like acidity, residual sugar, and alcohol content), predict whether a wine is red or white. This serves as a foundational exercise in structuring tabular data for deep learning and configuring a neural network for binary classification.

---

## 📁 Project Structure

```
wine_prediction/
├── wine_prediction.ipynb       # Main notebook
├── redwinequality.csv          # Dataset containing red wine samples
└── whitewinequality.csv        # Dataset containing white wine samples

```

---

## 🔬 Approach

### Pipeline

```
CSV Merging & Labeling → EDA (Histograms) → Train/Test Split 
    → Keras Sequential DNN (ReLU → ReLU → Sigmoid) → Binary Inference

```

### 1. Data Preparation & Labeling

```
red['type'] = 1
white['type'] = 0

wines = pd.concat([red, white], ignore_index=True)
wines.dropna(inplace=True)

```

The data comes from two separate files. We engineer a new target column `type` (1 for Red, 0 for White), concatenate them into a single continuous DataFrame, and clean missing values to prepare for training.

### 2. Exploratory Data Analysis (EDA)

Before building the model, the data is visualized using Matplotlib to compare distributions. For example, comparing the `alcohol` content between red and white wines helps verify that features have overlapping but distinct distributions, validating the need for a non-linear classifier.

### 3. Model Architecture

```
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, activation='relu', input_dim=12)) # Input layer + Hidden layer
model.add(Dense(9, activation='relu'))                # Hidden layer
model.add(Dense(1, activation='sigmoid'))             # Output layer

```

A feed-forward neural network is constructed:

* **Hidden Layers:** Uses **ReLU** (Rectified Linear Unit) activation to capture non-linear relationships in the chemical data.
* **Output Layer:** Uses a single neuron with a **Sigmoid** activation function, squeezing the output into a probability range of `[0, 1]`.

### 4. Compilation and Training

```
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)

```

* **Loss Function:** `binary_crossentropy` is the standard for 2-class problems.
* **Optimizer:** `adam` dynamically adjusts the learning rate during training.
* **Batch Size:** Set to `1`, meaning the model updates its weights after evaluating *every single sample* (Pure Stochastic Gradient Descent).

### 5. Binary Inference

```
y_pred = model.predict(X_test)
y_pred_labels = (y_pred >= 0.5).astype(int)

```

Because the Sigmoid output is a continuous probability, a threshold of `0.5` is applied. Probabilities ≥0.5 are cast to `1` (Red wine), and <0.5 are cast to `0` (White wine).

---

## 💡 Key Concepts

### Sigmoid + Binary Crossentropy

In Keras, binary classification requires a specific configuration. The final layer must have exactly `1` unit with a `sigmoid` activation, and the model must be compiled with `binary_crossentropy` loss. If you were doing multi-class classification, you would switch to `softmax` and `categorical_crossentropy`.

### The Impact of `batch_size=1`

Training with a batch size of 1 means the gradients are calculated and applied sample-by-sample. While this can lead to noisy gradient updates and slower epoch times (as it doesn't leverage matrix multiplication optimizations fully), it can sometimes help the model escape local minima. In practice, a mini-batch size (e.g., 32 or 64) is usually preferred.

### ⚠️ Notes on This Implementation

* **Feature Scaling:** The notebook passes raw chemical data directly into the neural network. Neural networks are highly sensitive to unscaled data. Applying a `StandardScaler` to normalize the inputs (mean=0, variance=1) before training would likely result in faster convergence and a more stable training process.
* **Input Shape Warning:** The `input_dim=12` argument in the first `Dense` layer is considered legacy in modern Keras versions. The updated approach is to use an explicit `Input(shape=(12,))` layer first.

---

## 🛠️ Requirements

```
pandas
numpy
matplotlib
scikit-learn
keras

```

---

