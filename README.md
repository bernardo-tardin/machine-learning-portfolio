# 🤖 Machine Learning Portfolio

A collection of hands-on Machine Learning projects built to develop practical skills across Computer Vision, Natural Language Processing, Healthcare ML, and Classical ML. Each project is self-contained and progressively explores more advanced techniques — spanning two frameworks (TensorFlow and PyTorch).

---

## 📂 Projects

| Folder | Title | Area | Techniques |
| --- | --- | --- | --- |
| [`classify_files`](https://www.google.com/search?q=./classify_files) | Text Category Classifier | NLP | CountVectorizer, Multinomial Naive Bayes |
| [`classify_handwritten_digits`](https://www.google.com/search?q=./classify_handwritten_digits) | Handwritten Digit Classifier | Computer Vision | TensorFlow/Keras, Dense Neural Networks, MNIST |
| [`ocr_handwritten_digits`](https://www.google.com/search?q=./ocr_handwritten_digits) | OCR with kNN | Computer Vision | OpenCV, k-Nearest Neighbors, Image Slicing |
| [`scikit_handwritten`](https://www.google.com/search?q=./scikit_handwritten) | Digit Recognition with MLP | Computer Vision | Scikit-learn, MLPClassifier, SGD |
| [`sms_detection`](https://www.google.com/search?q=./sms_detection) | SMS Spam Detector | NLP | Embeddings, BiLSTM, Transfer Learning (USE) |
| [`spam_detection`](https://www.google.com/search?q=./spam_detection) | Email Spam Detector | NLP | Tokenizer, LSTM, EarlyStopping, WordCloud |
| [`supportbot`](https://www.google.com/search?q=./supportbot) | Support Ticket Classifier | NLP | Regex, Naive Bayes, Stacked BiLSTM, Multiclass |
| [`truthguard_by_text`](https://www.google.com/search?q=./truthguard_by_text) | Fake News Detector | NLP | Embeddings, BiLSTM, Universal Sentence Encoder |
| [`pytorch_handwritten_digits`](https://www.google.com/search?q=./pytorch_handwritten_digits) | Digit Classifier in PyTorch | Computer Vision | PyTorch, nn.Module, DataLoader, SGD |
| [`neural_network_handwritten_digits`](https://www.google.com/search?q=./neural_network_handwritten_digits) | Digit Classifier from CSV | Computer Vision | TensorFlow/Keras, one-hot encoding, CSV pipeline |
| [`sign_language`](https://www.google.com/search?q=./sign_language) | ASL Sign Language Classifier | Computer Vision | TensorFlow + PyTorch, image\_dataset\_from\_directory, ImageFolder |
| [`disease_prediction`](https://www.google.com/search?q=./disease_prediction) | Disease Prediction from Symptoms | Healthcare ML | SVM, GaussianNB, Random Forest, Voting Ensemble, StratifiedKFold, RandomOverSampler |
| [`heart_disease_lg`](https://www.google.com/search?q=./heart_disease_lg) | 10-Year CHD Risk Prediction | Healthcare ML | Logistic Regression, SMOTE, StandardScaler, Confusion Matrix |
| [`heart_disease_predictor`](https://www.google.com/search?q=./heart_disease_predictor) | Heart Disease Predictor | Healthcare ML | Voting Ensemble (SVM, Naive Bayes, Random Forest), SMOTE, StratifiedKFold |
| [`wine_prediction`](https://www.google.com/search?q=./wine_prediction) | Wine Type Classification | Classical ML | Keras DNN, Binary Classification, EDA, Pandas merging |

---

## 🗺️ Learning Path

```
Classical ML          Deep Learning (TF)          PyTorch           Healthcare ML
────────────          ──────────────────          ───────           ─────────────
CountVectorizer  →    Embeddings + Dense    →     nn.Module    →    Voting Ensemble
+ Naive Bayes         Embeddings + LSTM            DataLoader        SVM + NB + RF
kNN (OpenCV)          Embeddings + BiLSTM          manual loop       StratifiedKFold
MLPClassifier         Transfer Learning (USE)      ImageFolder       Oversampling (SMOTE)
SVM                   CSV → image pipeline         framework         Logistic Regression
Keras DNN             Binary Classification                          Feature Scaling

```

---

## 🛠️ Tech Stack

* **Languages:** Python 3.12
* **Deep Learning:** TensorFlow / Keras · PyTorch · TensorFlow Hub
* **Classical ML:** Scikit-learn · OpenCV (`cv2`) · imbalanced-learn
* **Data:** Pandas · NumPy
* **NLP:** NLTK · WordCloud
* **Visualization:** Matplotlib · Seaborn

---

## 🚀 Getting Started

1. **Clone the repository**

```
git clone https://github.com/<your-username>/machine_learning_portfolio.git
cd machine_learning_portfolio

```
2. **Install dependencies**

```
pip install -r requirements.txt

```
3. **Navigate to a project and open the notebook**

```
cd disease_prediction
jupyter notebook

```

---

## 📊 Results Overview

| Project | Model | Framework | Accuracy |
| --- | --- | --- | --- |
| `ocr_handwritten_digits` | kNN (k=3) | OpenCV | \~91.6% |
| `pytorch_handwritten_digits` | Logistic Regression | PyTorch | \~84% |
| `classify_handwritten_digits` | Dense NN (Adam) | TensorFlow | \~97%+ |
| `neural_network_handwritten_digits` | Dense NN (Adam) | TensorFlow | \~97%+ |
| `scikit_handwritten` | MLPClassifier (SGD) | Scikit-learn | \~95%+ |
| `sign_language` | Dense NN | TF + PyTorch | — |
| `disease_prediction` | Voting Ensemble (SVM+NB+RF) | Scikit-learn | \~99%\* |
| `heart_disease_lg` | Logistic Regression | Scikit-learn | \~67% |
| `heart_disease_predictor` | Voting Ensemble (SVM+NB+RF) | Scikit-learn | \~88% |
| `wine_prediction` | Keras DNN | TensorFlow | \~94%+ |
| `sms_detection` | BiLSTM / USE | TensorFlow | \~98%+ |
| `spam_detection` | LSTM + EarlyStopping | TensorFlow | \~97%+ |
| `supportbot` | Stacked BiLSTM | TensorFlow | \~90%+ |
| `truthguard_by_text` | BiLSTM / USE | TensorFlow | \~98%+ |
| `classify_files` | Multinomial Naive Bayes | Scikit-learn | dataset-dependent |

\*Evaluated on resampled training data — not a held-out test set.

---

## 📚 Key Concepts Covered

* **Data Engineering:** Encoding, regex cleaning, class imbalance (undersampling & oversampling/SMOTE), feature selection, defensive CSV parsing, data merging
* **Computer Vision:** Image slicing, grayscale conversion, kNN, DataLoader batching, `image_dataset_from_directory`
* **NLP:** Stopword removal, CountVectorizer, TextVectorization, padding, word embeddings
* **Healthcare ML:** Symptom-based disease prediction, CHD risk prediction, voting ensembles, stratified cross-validation, Logistic Regression
* **Architectures:** Dense, LSTM, BiLSTM, Stacked BiLSTM, Transfer Learning, PyTorch `nn.Module`, SVM, Random Forest, Keras DNN
* **Framework Comparison:** TensorFlow vs PyTorch on the same task (`sign_language`)
* **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Cross-Validation

---

*This portfolio is actively updated as new projects are completed.*

---

