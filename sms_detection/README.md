# Comparative Spam Detection: Dense vs. Bi-LSTM vs. Transfer Learning

This repository contains a comprehensive Deep Learning project focused on binary classification of emails. Unlike traditional single-model approaches, this project benchmarks three distinct architectures—**Dense Neural Networks**, **Bidirectional LSTMs**, and **Transfer Learning (Universal Sentence Encoder)**—to identify the most effective method for detecting Spam.

## 📌 Project Overview

The core objective is to evaluate how different complexity levels in model architecture affect classification performance. The project covers the end-to-end Machine Learning lifecycle, from data cleaning and encoding to comparative performance visualization using standard NLP metrics.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** `Pandas`, `Numpy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **NLP & Preprocessing:** `Keras TextVectorization`, `TensorFlow Hub`
* **Deep Learning Framework:** `TensorFlow 2.x` / `Keras`
* **Evaluation Metrics:** `Scikit-Learn` (Accuracy, Precision, Recall, F1-Score)

## 📉 Methodology & Pipeline

### 1. Data Cleaning & Encoding

* **Handling Encoding:** Loaded data using `latin-1` to manage special characters common in email datasets.
* **Feature Selection:** Dropped unnecessary columns and renamed features for clarity (`label` and `Text`).
* **Label Encoding:** Mapped categorical labels ("ham"/"spam") to binary integers (0 and 1).

### 2. Text Preprocessing

Two distinct pipelines were used depending on the model:

* **Custom Tokenization:** Used `TextVectorization` to standardize (lower and strip punctuation), tokenize, and pad sequences based on the average message length.
* **Pre-trained Embeddings:** For the Transfer Learning model, raw strings were fed directly into the Universal Sentence Encoder (USE) via `TensorFlow Hub`.

## 🏗️ Model Architectures

I implemented and compared three specific models:

### 1. Dense Model (Baseline)

A lightweight model using a custom **Embedding layer** followed by **GlobalAveragePooling1D**. It serves as a fast and efficient baseline for text classification.

### 2. Bidirectional LSTM (Deep Sequence Model)

A complex architecture designed to capture sequential context.

* **Bidirectional Layers:** Processes text in both forward and backward directions to understand full sentence context.
* **Dropout:** Implemented to prevent overfitting during the learning process.

### 3. Transfer Learning (Universal Sentence Encoder)

Leverages Google's **Universal Sentence Encoder (USE)**. This model uses pre-trained weights from billions of words, allowing for high-level semantic extraction even with a limited training set.

## 📊 Results

The models were evaluated based on their ability to minimize False Positives (legitimate emails marked as spam).

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Dense Embedding** | \~98% | High | Moderate | High |
| **Bi-LSTM** | \~98% | High | High | High |
| **Transfer Learning (USE)** | **\~99%** | **Highest** | **Highest** | **Highest** |

Exportar para Sheets*Note: Results are visualized in the repository via bar charts and line graphs comparing performance across all metrics.*

## 📂 How to Run

1. **Clone this repository:**

```
git clone https://github.com/bernardo-tardin/sms-detection.git

```
2. **Install dependencies:**

```
pip install -r requirements.txt

```
3. **Run the analysis:**
Open the provided notebook in Google Colab or Jupyter:

```
jupyter notebook sms_detection.ipynb

```
