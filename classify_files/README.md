# 📄 Text Category Classifier

A text classification pipeline using **CountVectorizer** and **Multinomial Naive Bayes** to categorize documents into predefined topics. Includes interactive inference on custom text inputs.

---

## 🎯 Objective

Given a text document, classify it into the correct category using classical NLP techniques — no deep learning required.

---

## 📁 Project Structure

```
classify_files/
├── classify_files.ipynb       # Main notebook
└── synthetic_text_data.csv    # Dataset
```

---

## 🔬 Approach

### Pipeline
```
Raw Text → CountVectorizer → MultinomialNB → Predicted Category
```

### Key Steps
1. **Load dataset** from `synthetic_text_data.csv`
2. **Split** into 80% train / 20% test (`train_test_split`, `random_state=42`)
3. **Vectorize** with `CountVectorizer` — `fit_transform` on train, `transform` on test
4. **Train** Multinomial Naive Bayes
5. **Evaluate** with accuracy score and confusion matrix heatmap
6. **Inference** on custom user input

---

## 💡 Key Concepts

### CountVectorizer
Converts text into a sparse matrix of word counts. Each column represents a vocabulary word; each row represents a document.

> ⚠️ **Data Leakage:** Always use `fit_transform` on training data and `transform` only on test data. Using `fit_transform` on test data would let the model "see" future vocabulary, inflating metrics.

### Multinomial Naive Bayes
A probabilistic classifier that uses Bayes' theorem with the "naive" assumption that each word is statistically independent. Ideal for word-count vectors.

```
P(class | text) ∝ P(class) × P(word₁|class) × P(word₂|class) × ...
```

**Laplace Smoothing (+1)** prevents zero probabilities for unseen words.

---

## 📊 Evaluation

```python
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
```

---

## 🖥️ Inference Example

```python
user_input = "I love artificial intelligence and machine learning"
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"Category: '{predicted_label[0]}'")
```

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
