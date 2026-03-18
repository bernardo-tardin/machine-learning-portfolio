# 📈 Classic Text Categorization: Naive Bayes Baseline

This project implements a highly efficient **Classic Machine Learning** pipeline for text categorization. Unlike Deep Learning approaches (LSTMs or Transformers), this project leverages **Probabilistic Classification** and the **Bag-of-Words (BoW)** model to establish a high-performance baseline for text data.

## 📌 Project Overview

The core objective of this project is to demonstrate the power of statistical machine learning in NLP. By using **Multinomial Naive Bayes**, we create a "Speed Demon" classifier that can process large amounts of text in milliseconds with minimal computational overhead.

This serves as a crucial **Baseline Model** to evaluate whether more complex neural architectures provide enough marginal improvement to justify their increased resource consumption.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** `Scikit-Learn` (CountVectorizer, MultinomialNB)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn` (Heatmaps)

## 📉 Methodology & Pipeline

### 1. Feature Extraction: Bag-of-Words (BoW)

Instead of dense embeddings, we use the `CountVectorizer` to transform raw text into a sparse matrix of token counts. This approach:

* Learns a vocabulary from the training set.
* Represents each document as a frequency vector.
* Ignores word order to focus on word distribution/occurrence.

### 2. The Algorithm: Multinomial Naive Bayes

We chose **MultinomialNB** because of its proven track record in text classification.

* **Independence Assumption:** It assumes word features are independent, making the math incredibly fast.
* **Scalability:** It performs exceptionally well even with small datasets and high-dimensional sparse data.

### 3. Visual Evaluation: Confusion Matrix

To move beyond a simple "Accuracy" score, we implemented a **Confusion Matrix Heatmap**. This allows us to visualize:

* **Correct Classifications:** The diagonal line.
* **Model Confusion:** Off-diagonal values that show exactly which categories are being mistaken for one another.

---

## 📊 Results & Insights

* **Accuracy:** \~98% (on synthetic text data).
* **Performance:** Training and inference occur nearly instantaneously compared to LSTM/USE models.
* **Inference Example:**

> *Input:* "I love artificial intelligence and machine learning"
> 
> *Predicted Label:* **Technology**
> 
>

## 📂 How to Run

1. **Clone the repository:**

```
git clone https://github.com/bernardo-tardin/classic-text-classification.git

```
2. **Install dependencies:**

```
pip install pandas numpy matplotlib seaborn scikit-learn

```
3. **Run the script:**

```
python text_classification.py

```

---

## 📘 Comparison: Classic ML vs. Deep Learning

| Feature | Naive Bayes (This Project) | LSTM / USE (Previous Projects) |
| --- | --- | --- |
| **Speed** | ⚡ Extremely Fast | 🐢 Slower (needs GPU/CPU cycles) |
| **Data Needed** | 📉 Low (works on small sets) | 📈 High (needs large datasets) |
| **Context** | ❌ None (Bag-of-Words) | ✅ Captures sequence and meaning |
| **Use Case** | First Baseline / High-speed API | State-of-the-Art / Complex Nuance |