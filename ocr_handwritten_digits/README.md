# 🔢 OCR: Handwritten Digit Recognition (kNN Implementation)

This project demonstrates a classical approach to **Optical Character Recognition (OCR)** using the **k-Nearest Neighbors (kNN)** algorithm. Unlike deep learning models that learn abstract weights, this system utilizes geometric proximity to classify digits from a raw image grid.

## 📌 Project Overview

The objective of this project was to extract and classify 5,000 handwritten digits from a single large image file (`digits1.png`). The project focuses on high-performance image decomposition and the implementation of distance-based classification using the **OpenCV** machine learning module.

## 🛠️ Tech Stack

* **Language**: Python 3.12+
* **Computer Vision**: OpenCV (`cv2`) — used for image reading, grayscale conversion, and the kNN engine.
* **Data Science**: NumPy — used for efficient matrix slicing, data reshaping, and vectorized label generation.

## ⚙️ Core Pipeline

1. **Image Slicing & Decomposition**:
The original 1000×2000 pixel image was divided into 5,000 individual squares (20×20 pixels) using `np.vsplit` and `np.hsplit`.
2. **Feature Engineering (Flattening)**:
Each 2D digit square was reshaped into a 1D vector of 400 pixels (`reshape(-1, 400)`) and converted to `float32` to meet the algorithm's mathematical requirements.
3. **Supervised Learning (kNN)**:
Initialized `cv2.ml.KNearest_create()` and trained it on a dataset of 2,500 samples. The model "memorizes" the pixel intensity patterns for each digit (0-9).
4. **Inference (k=3)**:
The model identifies the 3 most similar training samples (neighbors) for each test input. The final classification is determined by a majority vote among these neighbors.

## 🧠 Key Technical Learning

* **Vectorized Operations**: By avoiding Python loops and using NumPy's slicing capabilities, the pipeline can process thousands of images in milliseconds.
* **Shape Matching & Debugging**: Addressed a critical `ValueError` involving broadcast shapes (50,1) vs (2500,1), highlighting the necessity of aligning sample size with label counts.
* **Dimensionality Management**: Learned to flatten high-dimensional image data while maintaining the statistical signal required for classification.

## 📂 Project Structure

* `ocr_handwritten_digits.ipynb`: The primary notebook containing image processing and kNN logic.
* `digits1.png`: The source dataset consisting of 5,000 handwritten samples.

## 🚀 How to Run

1. Ensure you have the required libraries installed:

```
pip install numpy opencv-python

```
2. Open the Jupyter Notebook and execute the cells to slice the image and train the classifier.

---