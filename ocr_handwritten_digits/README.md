# 🔍 OCR with k-Nearest Neighbors (OpenCV)

Recognizes handwritten digits from a raw image file using **OpenCV's built-in kNN classifier**. The entire pipeline — from image slicing to training and evaluation — runs inside OpenCV without TensorFlow or Scikit-learn.

---

## 🎯 Objective

Starting from a single PNG image (`digits1.png`) containing 5,000 handwritten digits arranged in a 50×100 grid, slice the image into individual samples, train a kNN classifier, and evaluate its accuracy.

---

## 📁 Project Structure

```
ocr_handwritten_digits/
├── ocr_handwritten_digits.ipynb   # Main notebook
└── digits1.png                    # Source image (50 rows × 100 columns of digits)
```

---

## 🔬 Approach

### Full Pipeline

```python
import numpy as np
import cv2

# 1. Load image and convert to grayscale
image    = cv2.imread('digits1.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Slice into 5,000 individual digit images (20×20 each)
division = list(np.hsplit(i, 100) for i in np.vsplit(gray_img, 50))
NP_array = np.array(division)  # shape: (50, 100, 20, 20)

# 3. Split: left half = train, right half = test
train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)
test_data  = NP_array[:, 50:].reshape(-1, 400).astype(np.float32)

# 4. Create labels: 250 samples per digit (0–9)
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels  = np.repeat(k, 250)[:, np.newaxis]

# 5. Train kNN and evaluate
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
ret, output, neighbours, distance = knn.findNearest(test_data, k=3)

# 6. Calculate accuracy
accuracy = (np.count_nonzero(output == test_labels) * 100) / output.size
print(accuracy)  # ~91.64%
```

---

## 💡 Key Concepts

### k-Nearest Neighbors (kNN)
Unlike neural networks, kNN does not learn weights. It memorizes the training data and classifies new samples by finding the `k` most similar training examples (nearest neighbors by Euclidean distance) and taking a majority vote.

With `k=3`: if the 3 nearest neighbors are digits `[7, 7, 1]`, the prediction is `7`.

### Why `float32`?
kNN computes Euclidean distances across 400-dimensional vectors (20×20 pixels flattened). These operations require floating-point numbers. OpenCV will raise an error if integer pixel values (`uint8`) are passed.

### `np.vsplit` / `np.hsplit`
- `np.vsplit(image, 50)` — 50 horizontal cuts → 50 rows
- `np.hsplit(row, 100)` — 100 vertical cuts per row → 100 digit images per row
- Total: 50 × 100 = **5,000 digit images**

### `np.newaxis`
Transforms a flat array `[0, 0, 0, 1, 1, 1, ...]` into a column vector required by OpenCV's training API.

---

## 📊 Result

| Classifier | k | Accuracy |
|------------|---|----------|
| kNN (OpenCV) | 3 | ~91.64% |

---

## 🛠️ Requirements

```
numpy
opencv-python
```
