# 🔥 Handwritten Digit Classifier (PyTorch)

Classifies handwritten digits (0–9) from MNIST using **PyTorch** — implementing a Logistic Regression model from scratch as an `nn.Module` class. Introduces the core PyTorch training paradigm: explicit forward pass, manual backpropagation loop, and DataLoader-based batching.

---

## 🎯 Objective

Build a first PyTorch model for digit classification, understanding the fundamental differences between the PyTorch and TensorFlow/Keras workflows.

---

## 📁 Project Structure

```
pytorch_handwritten_digits/
└── pytorch_handwritten_digits.ipynb   # Main notebook
```

---

## 🔬 Approach

### Data Loading with DataLoader

```python
train_dataset = dsets.MNIST(root='./data', train=True,
                            transform=transforms.ToTensor(), download=True)
train_loader  = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
```

`transforms.ToTensor()` converts PIL images to tensors and normalizes pixels to [0, 1] automatically.

### Model Definition

```python
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)  # No softmax — CrossEntropyLoss includes it
```

### Manual Training Loop

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = Variable(images.view(-1, 28 * 28))  # Flatten: (N, 1, 28, 28) → (N, 784)

        optimizer.zero_grad()       # Clear gradients from previous step
        outputs = model(images)     # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()             # Compute gradients
        optimizer.step()            # Update weights
```

### Evaluation

```python
correct, total = 0, 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)  # Argmax → predicted class
    total   += labels.size(0)
    correct += (predicted == labels).sum()

print(f'Accuracy: {100 * correct / total}%')  # ~84%
```

---

## 💡 Key Concepts

### PyTorch vs TensorFlow/Keras

| Aspect | TensorFlow/Keras | PyTorch |
|--------|-----------------|---------|
| Model definition | `Sequential([layers])` | Class inheriting `nn.Module` with `forward()` |
| Training | `model.fit()` automatic | Manual loop: `zero_grad → forward → loss → backward → step` |
| Data loading | Built-in generators | `DataLoader` with explicit `batch_size`, `shuffle` |
| Evaluation | `model.predict()` | `torch.no_grad()` context manager |

### `nn.CrossEntropyLoss` includes Softmax
Do NOT add a softmax activation to the last layer when using `CrossEntropyLoss` — it applies softmax internally. Adding it manually would apply softmax twice, distorting the probability distribution.

### `optimizer.zero_grad()`
PyTorch **accumulates** gradients by default (useful for RNNs). In standard training, call `zero_grad()` at the start of each batch to prevent gradient accumulation across steps.

### `torch.max(outputs, 1)`
Returns a tuple of `(values, indices)`. The indices are the predicted class labels — equivalent to `np.argmax(..., axis=1)` in NumPy.

### `Variable` (legacy)
In modern PyTorch (≥0.4), all tensors support autograd directly. `Variable` still works but is redundant — you'll see it in older tutorials.

---

## 📊 Results

| Model | Framework | Accuracy |
|-------|-----------|----------|
| Logistic Regression (linear) | PyTorch + SGD | ~84% |
| Dense NN (2 hidden layers) | TensorFlow + Adam | ~97%+ |

The gap is expected: a single linear layer without hidden units cannot learn non-linear patterns in image data. This is the baseline — the starting point for deeper PyTorch architectures.

---

## 🛠️ Requirements

```
torch
torchvision
```
