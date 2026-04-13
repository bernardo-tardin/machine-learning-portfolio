# 🦠 Disease Prediction from Symptoms

Predicts diseases based on symptom input using a **Voting Ensemble** of three classical ML classifiers: Support Vector Machine, Gaussian Naive Bayes, and Random Forest. Introduces ensemble methods, imbalanced dataset handling with oversampling, and stratified cross-validation.

---

## 🎯 Objective

Given a set of patient symptoms as input, predict the most likely disease by combining the outputs of three independently trained classifiers through majority voting.

---

## 📁 Project Structure

```
disease_prediction/
├── disease_prediction.ipynb        # Main notebook
└── improved_disease_dataset.csv    # Dataset (symptoms + disease label)
```

---

## 🔬 Approach

### Pipeline

```
CSV → LabelEncoder → RandomOverSampler → StratifiedKFold CV
    → SVM + GaussianNB + RandomForest → Voting Ensemble → Inference
```

### 1. Preprocessing

**Label Encoding:**
```python
encoder = LabelEncoder()
data['disease'] = encoder.fit_transform(data['disease'])
# encoder.classes_ saves the original disease names for inference
```

**Class Balancing with RandomOverSampler:**
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

RandomOverSampler duplicates samples from minority classes until all classes have equal representation. Without balancing, a model trained on skewed data learns to ignore rare diseases — maximising accuracy by simply predicting the majority class.

### 2. Stratified Cross-Validation

```python
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_resampled, y_resampled, cv=stratified_kfold)
```

`StratifiedKFold` preserves class proportions in each fold — essential for multiclass problems with many disease categories.

### 3. The Three Models

```python
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}
```

| Model | Mechanism | Strength |
|-------|-----------|----------|
| **SVM** | Finds maximum-margin hyperplane between classes | Effective in high-dimensional spaces (many symptoms) |
| **Gaussian Naive Bayes** | Probabilistic, assumes feature independence | Fast, works well for medical classification |
| **Random Forest** | Ensemble of decision trees with bagging | Robust to overfitting, handles noise well |

### 4. Voting Ensemble

```python
from statistics import mode

final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
```

For each sample, all three models vote. The final prediction is the most common answer. If all three disagree, `mode` returns the first value.

**Why combine models?** Each model has different failure modes. SVM may fail where Naive Bayes succeeds. Combining reduces prediction variance and generally improves robustness.

### 5. Inference Function

```python
def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)            # Zero vector

    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1   # Activate present symptoms

    input_df = pd.DataFrame([input_data], columns=symptoms)

    rf_pred  = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred  = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction":   nb_pred,
        "SVM Prediction":           svm_pred,
        "Final Prediction":         mode([rf_pred, nb_pred, svm_pred])
    }

print(predict_disease("skin_rash,fever,headache"))
```

Input symptoms are converted into a binary vector (1 = present, 0 = absent) matching the training format. `encoder.classes_[index]` converts numeric predictions back to readable disease names.

---

## 💡 Key Concepts

### RandomOverSampler vs SMOTE
- **RandomOverSampler** (used here): Duplicates existing minority samples randomly. Simple and fast.
- **SMOTE**: Generates *synthetic* minority samples by interpolating between nearest neighbours. More sophisticated, less prone to exact duplicates.

### `StratifiedKFold` vs `KFold`
Standard `KFold` splits data randomly. `StratifiedKFold` ensures each fold contains the same class distribution as the full dataset — critical when some diseases have few samples.

### Voting Ensemble
Combining classifiers that make *different types of errors* reduces overall error. SVM is geometric (margins), NB is probabilistic (statistics), RF is structural (decision trees) — their errors tend not to overlap.

### ⚠️ Notes on This Implementation
- The Random Forest confusion matrix cell incorrectly uses `cf_matrix_nb` instead of `cf_matrix_rf` — showing the wrong matrix visually.
- All models are evaluated on the resampled training data, not a held-out test set. For production evaluation, oversampling should be applied *inside* each CV fold to avoid leakage.

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
scipy
```
