# 🤖 SupportBot: Multiclass Ticket Routing & Data Quality Analysis

SupportBot is a Natural Language Processing (NLP) project designed to automate the routing of customer support tickets to specific departments (e.g., Billing, Technical Issue, Refund). This project serves as a **comparative study** between statistical baselines and deep learning architectures in the context of synthetic, high-noise datasets.

## 📌 Project Overview

The goal was to classify tickets into 4-5 distinct categories. The project implements a full pipeline, from heavy text preprocessing and class balancing (Undersampling) to model benchmarking.

### The "Data Quality" Challenge

A key finding of this project is the **Impact of Synthetic Data**. Despite using advanced architectures like Bidirectional LSTMs, the accuracy remained at baseline levels (\~25% for 4 classes).

* **Diagnosis:** The dataset utilizes overlapping templates for different labels, creating a "Semantic Vacuum" where classes are mathematically inseparable.
* **Value:** This project demonstrates the principle of **Garbage In, Garbage Out (GIGO)** and the importance of data audit before model deployment.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** `Scikit-Learn` (Naive Bayes, CountVectorizer, LabelEncoding)
* **Deep Learning:** `TensorFlow/Keras` (Bidirectional LSTM, GlobalAveragePooling1D)
* **NLP Tools:** `NLTK` (Stopwords), `Regex` (Pattern Cleaning)
* **Visualization:** `Seaborn`, `Matplotlib` (Confusion Matrix Heatmaps)

## 🏗️ Model Architectures

### 1. Statistical Baseline (Multinomial Naive Bayes)

* **Approach:** Bag-of-Words (BoW) frequency analysis.
* **Purpose:** To establish a low-cost computational benchmark.
* **Finding:** On synthetic data, the probabilistic approach performed similarly to complex neural networks, highlighting that when data lacks clear signals, complexity does not yield marginal gains.

### 2. Deep Learning (Bidirectional LSTM)

* **Approach:** Sequence modeling with recurrent layers.
* **Architecture:**

* **Embedding Layer:** 128-dimensional vectors.
* **Stacked Bi-LSTMs:** To capture bidirectional context.
* **GlobalAveragePooling1D:** Used instead of Flatten to reduce parameters and combat overfitting on noisy templates.
* **Softmax Output:** For multiclass probability distribution.

## 📊 Key Lessons Learned

* **Feature Sensitivity:** In multiclass support routing, the **Subject Line** often carries more "weight" than the body description.
* **Overfitting Identification:** Documented the divergence between Training Loss and Validation Loss, identifying the point where the model stops "learning" and starts "memorizing" noise.
* **Metrics over Accuracy:** Used **Confusion Matrices** to identify class confusion, proving that the model struggled most with categories sharing the same boilerplate text.

## 📂 How to Run

1. **Clone the repository:**

```
git clone https://github.com/bernardo-tardin/supportbot-router.git

```
2. **Install dependencies:**

```
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn nltk

```
3. **Run the Notebook:** Open `supportbot.ipynb` in Jupyter or Google Colab and run all cells.

---