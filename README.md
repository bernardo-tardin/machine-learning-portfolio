# 🤖 Machine Learning Portfolio

A collection of hands-on Machine Learning projects built to develop practical skills across Computer Vision, Natural Language Processing, and Classical ML. Each project is self-contained and progressively explores more advanced techniques — spanning two frameworks (TensorFlow and PyTorch).

---

## 📂 Projects

| Folder | Title | Area | Techniques |
|--------|-------|------|------------|
| [`classify_files`](./classify_files) | Text Category Classifier | NLP | CountVectorizer, Multinomial Naive Bayes |
| [`classify_handwritten_digits`](./classify_handwritten_digits) | Handwritten Digit Classifier | Computer Vision | TensorFlow/Keras, Dense Neural Networks, MNIST |
| [`ocr_handwritten_digits`](./ocr_handwritten_digits) | OCR with kNN | Computer Vision | OpenCV, k-Nearest Neighbors, Image Slicing |
| [`scikit_handwritten`](./scikit_handwritten) | Digit Recognition with MLP | Computer Vision | Scikit-learn, MLPClassifier, SGD |
| [`sms_detection`](./sms_detection) | SMS Spam Detector | NLP | Embeddings, BiLSTM, Transfer Learning (USE) |
| [`spam_detection`](./spam_detection) | Email Spam Detector | NLP | Tokenizer, LSTM, EarlyStopping, WordCloud |
| [`supportbot`](./supportbot) | Support Ticket Classifier | NLP | Regex, Naive Bayes, Stacked BiLSTM, Multiclass |
| [`truthguard_by_text`](./truthguard_by_text) | Fake News Detector | NLP | Embeddings, BiLSTM, Universal Sentence Encoder |
| [`pytorch_handwritten_digits`](./pytorch_handwritten_digits) | Digit Classifier in PyTorch | Computer Vision | PyTorch, nn.Module, DataLoader, SGD |
| [`neural_network_handwritten_digits`](./neural_network_handwritten_digits) | Digit Classifier from CSV | Computer Vision | TensorFlow/Keras, one-hot encoding, CSV pipeline |
| [`sign_language`](./sign_language) | ASL Sign Language Classifier | Computer Vision | TensorFlow + PyTorch, image_dataset_from_directory, ImageFolder |

---

## 🗺️ Learning Path

```
Classical ML              Deep Learning (TF)            PyTorch              Cross-Framework
────────────              ──────────────────            ───────              ──────────────
CountVectorizer     →     Embeddings + Dense      →     nn.Module      →     Same task,
+ Naive Bayes             Embeddings + LSTM              DataLoader           two frameworks
kNN (OpenCV)              Embeddings + BiLSTM            manual loop          (sign_language)
MLPClassifier             Transfer Learning (USE)
                          CSV → image pipeline
```

---

## 🛠️ Tech Stack

- **Languages:** Python 3.12
- **Deep Learning:** TensorFlow / Keras · PyTorch · TensorFlow Hub
- **Classical ML:** Scikit-learn · OpenCV (`cv2`)
- **Data:** Pandas · NumPy
- **NLP:** NLTK · WordCloud
- **Visualization:** Matplotlib · Seaborn

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/machine_learning_portfolio.git
   cd machine_learning_portfolio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to a project and open the notebook**
   ```bash
   cd sign_language
   jupyter notebook
   ```

---

## 📊 Results Overview

| Project | Model | Framework | Accuracy |
|---------|-------|-----------|----------|
| `ocr_handwritten_digits` | kNN (k=3) | OpenCV | ~91.6% |
| `pytorch_handwritten_digits` | Logistic Regression | PyTorch | ~84% |
| `classify_handwritten_digits` | Dense NN (Adam) | TensorFlow | ~97%+ |
| `neural_network_handwritten_digits` | Dense NN (Adam) | TensorFlow | ~97%+ |
| `scikit_handwritten` | MLPClassifier (SGD) | Scikit-learn | ~95%+ |
| `sign_language` | Dense NN | TF + PyTorch | — |
| `sms_detection` | BiLSTM / USE | TensorFlow | ~98%+ |
| `spam_detection` | LSTM + EarlyStopping | TensorFlow | ~97%+ |
| `supportbot` | Stacked BiLSTM | TensorFlow | ~90%+ |
| `truthguard_by_text` | BiLSTM / USE | TensorFlow | ~98%+ |
| `classify_files` | Multinomial Naive Bayes | Scikit-learn | dataset-dependent |

---

## 📚 Key Concepts Covered

- **Data Engineering:** Encoding, regex cleaning, class imbalance, feature selection, defensive CSV parsing
- **Computer Vision:** Image slicing, grayscale conversion, kNN, DataLoader batching, `image_dataset_from_directory`
- **NLP:** Stopword removal, CountVectorizer, TextVectorization, padding, word embeddings
- **Architectures:** Dense, LSTM, BiLSTM, Stacked BiLSTM, Transfer Learning, PyTorch `nn.Module`
- **Framework Comparison:** TensorFlow vs PyTorch on the same task (`sign_language`)
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Best Practices:** `random_state`, `fit` vs `transform`, callbacks, `model.eval()` + `torch.no_grad()`

---

*This portfolio is actively updated as new projects are completed.*
