# 🧠 Digit Recognition with MLPClassifier (Scikit-learn)

Classifies handwritten digits using Scikit-learn's built-in **MLPClassifier** (Multi-Layer Perceptron). Uses the compact `load_digits` dataset (8×8 images) and tracks the training loss curve to evaluate convergence.

---

## 🎯 Objective

Train a shallow MLP on Scikit-learn's built-in digit dataset and visualize the loss curve across iterations to understand how the learning rate and architecture affect convergence.

---

## 📁 Project Structure

```
scikit_handwritten/
└── scikit_handwritten.ipynb   # Main notebook
```

---

## 🔬 Approach

```python
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load built-in digit dataset (1,797 images, 8×8 = 64 features each)
digits = datasets.load_digits()

# Flatten images
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Manual split: first 1,000 for training, rest for testing
X_train, y_train = X[:1000], y[:1000]
X_test,  y_test  = X[1000:], y[1000:]

# Train MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(15,),     # 1 hidden layer, 15 neurons
    activation='logistic',        # Sigmoid activation
    solver='sgd',                 # Stochastic Gradient Descent
    learning_rate_init=1,         # Initial learning rate
    alpha=1e-4,                   # L2 regularization
    tol=1e-4,
    random_state=1,
    verbose=True
)
mlp.fit(X_train, y_train)

# Evaluate
print(accuracy_score(y_test, mlp.predict(X_test)))
```

---

## 💡 Key Concepts

### MLPClassifier vs Keras Dense
Both are fully-connected neural networks. Scikit-learn's `MLPClassifier` is simpler to configure and integrates with the sklearn pipeline ecosystem, but lacks GPU support and advanced architectures. Keras is more flexible for production models.

### Loss Curve
```python
axes.plot(mlp.loss_curve_, 'o-')
```
Plotting `mlp.loss_curve_` reveals how the model converges over training iterations.

> ⚠️ **`learning_rate_init=1` is unusually high.** A standard starting point is `0.001`. A high learning rate can cause the optimizer to overshoot the loss minimum, producing oscillation in the loss curve rather than smooth convergence. Monitor the curve to verify stability.

### `activation='logistic'`
This is the Sigmoid function (`f(x) = 1 / (1 + e^(-x))`). It maps any input to (0, 1). Note: in the Keras ecosystem, this is called `'sigmoid'`.

### L2 Regularization (`alpha`)
`alpha=1e-4` applies a weight decay penalty that discourages overfitting by keeping weights small.

---

## 📊 Dataset Comparison

| Dataset | Images | Resolution | Features |
|---------|--------|------------|----------|
| `load_digits` (used here) | 1,797 | 8×8 | 64 |
| MNIST (other projects) | 70,000 | 28×28 | 784 |

---

## 🛠️ Requirements

```
scikit-learn
matplotlib
```
