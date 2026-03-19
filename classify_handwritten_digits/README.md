# 🔢 MNIST Handwritten Digit Classification

This project explores the application of Machine Learning and Deep Learning to recognize handwritten digits using the **MNIST** dataset. It demonstrates the evolution from a simple linear classifier to a Deep Neural Network (DNN), highlighting fundamental concepts of computer vision and tensor processing.

## 📌 Project Overview

The MNIST dataset consists of 70,000 grayscale images of digits from 0 to 9, each with a resolution of 28x28 pixels. This project implements a complete model development lifecycle:

1. **Data Loading and Exploration**: Using NumPy to manage raw image arrays.
2. **Preprocessing**: Pixel normalization to optimize gradient descent.
3. **Architectural Benchmarking**: Comparing linear baseline models against multi-layer neural networks.
4. **Inference and Visualization**: Testing predictions on individual samples with visual feedback.

## 🛠️ Tech Stack

* **Language**: Python 3.12
* **Deep Learning**: TensorFlow and Keras
* **Data Manipulation**: NumPy (array slicing and `np.argmax` logic)
* **Visualization**: Matplotlib (image rendering and prediction labeling)

## 🏗️ Model Architectures

### 1. Linear Baseline

A direct approach using a **Flatten** layer followed by a single **Dense** layer with **Softmax** activation.

* **Input**: 28x28 pixels (flattened to 784 neurons).
* **Output**: 10 neurons representing multiclass probability distribution.

### 2. Deep Learning Model (DNN)

To capture complex patterns like curves and intersections, a deep architecture was implemented:

* **Hidden Layers**: Two Dense layers with **128 neurons** each.
* **ReLU Activation**: Introduced to handle non-linearity and allow the network to learn abstract shapes.
* **Optimizer**: Adam.
* **Loss Function**: Sparse Categorical Crossentropy.

## 🧠 Key Technical Insights

* **Normalization**: Pixel values (0-255) were scaled to a **0-1 range** using `tf.keras.utils.normalize`, which significantly accelerates model convergence.
* **Flattening**: Converting the 2D image matrix into a 1D vector is required for processing by fully connected (Dense) layers.
* **Softmax vs. Argmax**: While the model outputs 10 probabilities via Softmax, `np.argmax` is used during inference to select the digit with the highest confidence.
* **Model Portability**: The trained deep learning model is exported in `.h5` format for production readiness.

## 📂 How to Run

1. Clone this repository.
2. Install the required dependencies:

```
pip install numpy matplotlib tensorflow

```
3. Launch the Jupyter Notebook `classify_handwritten_digits.ipynb` and execute the cells in sequence.

## 💾 Model Export

The final model is saved as follows:

```
model_dl.save('epic_num_reader.h5')

```

---