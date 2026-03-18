# 🛡️ TruthGuard: Fake News Detection with NLP & Deep Learning

This repository contains **TruthGuard**, a Deep Learning project designed to classify news articles as **Real** or **Fake**. The project explores the predictive power of different news components by benchmarking models trained on **Full Article Text** versus **Article Titles**.

## 📌 Project Overview

The objective is to build a robust classifier to combat disinformation. The project implements a complete machine learning pipeline, including data engineering, text preprocessing (NLP), and a comparative analysis of three distinct neural network architectures.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** `Pandas`, `Numpy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **NLP:** `NLTK` (Stopwords), `Keras TextVectorization`
* **Deep Learning:** `TensorFlow 2.x`, `Keras`, `TensorFlow Hub` (Universal Sentence Encoder)
* **Metrics:** `Scikit-Learn` (Accuracy, Precision, Recall, F1-Score)

## 📉 Methodology & Pipeline

### 1. Data Engineering & Balancing

* **Cleaning:** Dropped metadata (`subject`, `date`) to focus strictly on textual content.
* **Balancing:** Performed **Undersampling** on the majority class to achieve a perfect 50/50 distribution between "Real" and "Fake" news, preventing model bias.
* **Labeling:** Encoded Real news as `1` and Fake news as `0`.

### 2. Text Preprocessing (NLP)

* **Stopwords Removal:** Leveraged `NLTK` to filter out common words that lack semantic importance.
* **Vectorization:** Implemented a `TextVectorization` layer to tokenize text into integer indices, using the average length of the input for padding.
* **Padding:** Standardized all input sequences to ensure uniform matrix dimensions for the neural network.

### 3. Model Architectures

I benchmarked three different approaches to find the optimal balance between speed and performance:

* **Dense Model:** A baseline model using Embeddings and `GlobalAveragePooling1D`.
* **Bi-LSTM Model:** A complex architecture using **Bidirectional LSTMs** to capture long-range dependencies and context from both directions.
* **Transfer Learning (USE):** Utilizing Google's **Universal Sentence Encoder** to extract high-level semantic embeddings.

---

## 📊 Performance Results

The project compared performance based on two different inputs. Interestingly, the models performed exceptionally well on full text body, suggesting that stylistic patterns in disinformation are more evident in longer sequences.

### **Analysis by Full Text (Body)**

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Dense Embedding** | 99.46% | 99.32% | 99.60% | 99.46% |
| **Bi-LSTM** | **99.80%** | **100.0%** | 99.60% | **99.80%** |
| **Transfer Learning (USE)** | 91.47% | 91.08% | 91.97% | 91.52% |

**Analysis by Title**

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Dense Embedding** | 94.81% | 94.69% | 94.96% | 94.83% |
| **Bi-LSTM** | **95.45%** | **94.70%** | **96.31%** | **95.50%** |
| **Transfer Learning (USE)** | 91.94% | 91.29% | 92.74% | 92.01% |


## 📂 How to Run

1. **Clone the repository:**

```
git clone https://github.com/bernardo-tardin/truthguard-fake-news.git

```
2. **Install dependencies:**

```
pip install -r requirements.txt

```
3. **Run the analysis:**
Execute the Jupyter Notebook to train and evaluate the models:

```
jupyter notebook truthguard_analysis.ipynb

```