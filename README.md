# 🤖 Machine Learning & NLP Portfolio

### Developed by Bernardo Moraes

Welcome to my Machine Learning portfolio. This repository serves as a comprehensive collection of my work in **Deep Learning**, **Natural Language Processing (NLP)**, and **Data Engineering**. Each project is focused on solving real-world classification problems using state-of-the-art architectures and rigorous evaluation metrics.

---

## 🚀 Projects Overview

| Project | Brief Description | Tech Stack | Key Result |
| --- | --- | --- | --- |
| **🛡️ TruthGuard** | Fake News classification comparing Headlines vs. Full-Text. | `Bi-LSTM`, `TF-Hub (USE)` | **99.8% Accuracy** |
| **🤖 SupportBot** | Multiclass ticket routing & Data Quality (GIGO) audit. | `Bi-LSTM`, `Naive Bayes`, `Softmax` | **Case Study on Noise** |
| **📧 Multi-Arch Spam** | Benchmarking 3 architectures for email filtering. | `Dense`, `LSTM`, `USE` | **F1-Score Optimized** |
| **📈 Classic Categorization** | High-speed baseline using statistical ML. | `Naive Bayes`, `Sklearn` | **98% Accuracy** |
| **📩 Basic Spam** | Initial NLP pipeline and LSTM baseline. | `LSTM`, `NLTK`, `Keras` | **98% Accuracy** |


## 📂 Project Highlights

### 1. 🛡️ TruthGuard: Fake News Detection

A sensitivity analysis project that identifies misinformation by comparing the predictive power of headlines against the full article body.

* **Technical Highlight:** Conducted a comparative study between short-form input (Titles) and long-form sequences (Body).
* **Winning Model:** A **Bidirectional LSTM** reached 99.8% accuracy on full-text data, proving that stylistic patterns in disinformation are most evident in narrative structure.
* [🔗 View Project Folder](./truthguard)

### 2. 🤖 SupportBot: Multiclass Ticket Routing

A multiclass classification project designed to route customer tickets to departments, focusing on architectural trade-offs and **Data Quality Audit**.

* **Technical Highlight:** Implemented a **GIGO (Garbage In, Garbage Out)** diagnostic. Identified that model convergence was limited by the semantic vacuum of synthetic templates rather than architectural flaws.
* **Architecture:** Utilized **Stacked Bi-LSTMs** with **GlobalAveragePooling1D** to reduce parameters and combat overfitting in high-noise environments.
* [🔗 View Project Folder](./supportbot)

### 3. 📧 Multi-Architecture Spam Classification

A deep dive into architectural trade-offs, comparing traditional Dense networks, Recurrent Neural Networks (LSTM), and Transfer Learning.

* **Technical Highlight:** Benchmarked model latency vs. classification precision to determine the most viable model for a production environment.
* **Key Insight:** Google’s **Universal Sentence Encoder (USE)** provided the most robust semantic understanding for complex, unstructured messages.
* [🔗 View Project Folder](./sms_detection)

### 4. 📈 Classic Text Categorization (Naive Bayes)

A project focused on computational efficiency and establishing a statistical baseline for NLP tasks.

* **Technical Highlight:** Implemented a **Bag-of-Words (BoW)** model using `CountVectorizer` to transform text into sparse frequency matrices.
* **Algorithm:** Leveraged **Multinomial Naive Bayes** to create a lightweight, high-speed classifier that rivals deep learning models on structured datasets.
* [🔗 View Project Folder](./classify_files)

### 5. 📩 Spam Email Classifier (NLP/LSTM)

The foundational project used to master the machine learning life-cycle for text data.

* **Technical Highlight:** Implemented **Undersampling** techniques to resolve class imbalance, ensuring the model remained unbiased toward legitimate emails.
* **Core Concepts:** Tokenization, Padding, and the use of Dropout layers to prevent overfitting.
* [🔗 View Project Folder](./spam_detection)

---

## 🛠️ Tech Stack & Skills

* **Languages:** Python 3.10+
* **Deep Learning:** TensorFlow 2.x, Keras, TensorFlow Hub.
* **NLP Tools:** NLTK, WordCloud, TextVectorization, CountVectorizer, Word Embeddings, LabelEncoding.
* **Data Science:** Pandas, NumPy, Scikit-Learn.
* **Visualization:** Matplotlib, Seaborn (Confusion Matrix Heatmaps).
* **Architectures:** Dense Networks, LSTMs, Bidirectional LSTMs, Transfer Learning, Naive Bayes, GlobalAveragePooling1D.
* **Problem Domains:** Binary Classification, Multiclass Classification, Data Quality Auditing.
