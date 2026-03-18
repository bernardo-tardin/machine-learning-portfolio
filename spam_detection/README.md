# Spam Email Detection with NLP & LSTM

This repository contains a Deep Learning project focused on binary classification of emails. Using Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks, the model identifies whether an email is **Spam** or **Ham** (legitimate) with high accuracy.

## 📌 Project Overview

The core objective is to build an automated system to filter unwanted emails. The project covers the entire machine learning lifecycle: from data balancing and cleaning to the development of a recurrent neural network architecture.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** `Pandas`, `Numpy`
* **Visualization:** `Matplotlib`, `Seaborn`, `WordCloud`
* **NLP:** `NLTK` (Natural Language Toolkit)
* **Deep Learning Framework:** `TensorFlow` / `Keras`
* **Machine Learning Utilities:** `Scikit-Learn`

## 📉 Methodology & Pipeline

### 1. Data Analysis & Balancing

The dataset initially contained a class imbalance (more "Ham" than "Spam"). To ensure the model doesn't become biased, I performed **Undersampling** on the majority class to achieve a perfect 50/50 distribution.

### 2. Text Preprocessing (NLP)

To convert raw text into a format understandable by a neural network, the following steps were implemented:

* **Cleaning:** Removal of headers (like "Subject:") and punctuation.
* **Stopwords Removal:** Filtering out common words (e.g., 'the', 'is', 'at') that do not carry semantic weight.
* **Tokenization:** Converting words into unique integer indices.
* **Padding:** Ensuring all sequences have a uniform length of 100 tokens to feed into the model.

### 3. Model Architecture

I built a Sequential Deep Learning model designed to understand context in sequences:

* **Embedding Layer:** Creates dense vector representations of words, capturing semantic similarities.
* **LSTM Layer:** A recurrent layer that maintains "memory" of previous tokens, crucial for understanding sentence structure.
* **Dense (ReLU):** Fully connected layer for feature processing.
* **Dense (Sigmoid):** Output layer that generates a probability between 0 and 1.

### 4. Training Optimization

* **Loss Function:** Binary Crossentropy.
* **Optimizer:** Adam.
* **Callbacks:** \* `EarlyStopping`: Prevents overfitting by stopping training when validation accuracy plateaus.

* `ReduceLROnPlateau`: Fine-tunes the learning rate during training to find the global minimum of the loss function.

## 📊 Results

The model demonstrates excellent performance on unseen data.

* **Test Accuracy:** ~98%
* **Loss:** ~0.05

## 📂 How to Run

1. Clone this repository:

```
git clone https://github.com/berrnardo-tardin/spam-detection-lstm.git

```
2. Install dependencies:

```
pip install -r requirements.txt

```
3. Run the Jupyter Notebook:

```
jupyter notebook spam_detection.ipynb

```
