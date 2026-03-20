# ✋ ASL Sign Language Digit Classifier

Classifies hand sign images for digits 0–9 in **American Sign Language (ASL)** using both **TensorFlow/Keras** and **PyTorch** on the same task — the only project in this portfolio that implements and compares two deep learning frameworks side by side.

---

## 🎯 Objective

Train a classifier for real-world hand gesture images (not pixel CSVs), and directly compare how TensorFlow and PyTorch handle the full pipeline: data loading, preprocessing, model definition, training loop, and evaluation.

---

## 📁 Project Structure

```
sign_language/
├── sign_language.ipynb          # Main notebook
└── data/
    ├── asl_dataset_digits/      # Training data — one subfolder per digit class
    └── test/                    # Test data — one subfolder per digit class
```

---

## 🔬 Approach

### TensorFlow Pipeline

**Data loading from directory:**

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/asl_dataset_digits',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(64, 64),
    batch_size=32,
    color_mode='grayscale'   # 1 channel instead of 3 (RGB)
)
```

Each subfolder name becomes a class label automatically.

**Model with Rescaling layer:**

```python
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 1)),  # Normalize inside the graph
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

**Evaluation:**

```python
for images, labels in test_ds:
    preds  = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

accuracy = accuracy_score(y_true, y_pred)
```

---

### PyTorch Pipeline

**Transforms as preprocessing pipeline:**

```python
transformations = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),   # Converts to tensor and normalizes to [0, 1]
])

dataset = datasets.ImageFolder('data/asl_dataset_digits', transform=transformations)

# Manual split
size_train = int(0.8 * len(dataset))
size_val   = len(dataset) - size_train
train_data, val_data = random_split(dataset, [size_train, size_val])
```

**Model definition:**

```python
class SignsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers  = nn.Sequential(
            nn.Linear(64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)   # No softmax — CrossEntropyLoss includes it
        )

    def forward(self, x):
        return self.layers(self.flatten(x))
```

**Evaluation loop:**

```python
model.eval()
with torch.no_grad():   # Disable gradient computation for inference
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
```

---

## 💡 Key Concepts

### Framework Comparison on the Same Task

| Aspect | TensorFlow | PyTorch |
|--------|------------|---------|
| Load from directory | `image_dataset_from_directory` | `datasets.ImageFolder` |
| Normalization | `Rescaling(1./255)` as a layer | `transforms.ToTensor()` |
| Validation split | `validation_split=0.2` (automatic) | `random_split` (manual) |
| Model definition | `Sequential([...])` | `nn.Module` class with `forward()` |
| Evaluation | `model.predict()` | `torch.no_grad()` context |

### `image_dataset_from_directory` vs `ImageFolder`
Both infer class labels from subfolder names. The key difference: TensorFlow's version supports `validation_split` directly, while PyTorch requires manually splitting with `random_split` after loading.

### `model.eval()` + `torch.no_grad()`
Always pair these for PyTorch evaluation:
- `model.eval()` disables Dropout and BatchNorm (sets them to inference mode)
- `torch.no_grad()` disables the autograd graph, saving memory and speeding up inference

### ⚠️ Bug in the Notebook
`Rescaling(1./225)` should be `Rescaling(1./255)`. The typo (`225` instead of `255`) means pixel values are normalized to a range slightly above 1.0 (max ≈ 1.13 instead of 1.0). The model still trains because it compensates through weight learning, but this is a subtle source of inconsistency.

### `transforms.ToTensor()`
Does three things in one call: converts a PIL Image to a PyTorch tensor, reorders axes from HWC (Height, Width, Channels) to CHW (Channels, Height, Width), and normalizes pixel values to [0.0, 1.0].

---

## 🛠️ Requirements

```
tensorflow>=2.15.0
torch
torchvision
opencv-python
scikit-learn
matplotlib
seaborn
numpy
```
