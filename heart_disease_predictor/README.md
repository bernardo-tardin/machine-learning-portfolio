# ❤️ Heart Disease Predictor (Ensemble Learning)

Predicts the presence of heart disease based on clinical test results using an **Ensemble** of three classical ML classifiers: Support Vector Machine (SVM), Gaussian Naive Bayes, and Random Forest. Introduces class balancing with SMOTE, feature scaling, and stratified cross-validation.

---

## 🎯 Objective

Given a set of clinical patient metrics (like age, cholesterol, maximum heart rate, etc.), predict whether the patient has a heart condition by combining the outputs of three independently trained models through majority voting.

---

## 📁 Project Structure

```
heart_disease_predictor/
├── heart_disease_predictor.ipynb    # Main notebook
└── heart_cleveland_upload.csv       # Dataset (13 clinical features + condition label)

```

---

## 🔬 Approach

### Pipeline

```
CSV → Train/Test Split (Stratified) → SMOTE Oversampling → StandardScaler 
    → SVM + GaussianNB + RandomForest → Voting Ensemble → Evaluation

```

### 1. Preprocessing & Balancing

**Train/Test Split & Stratification:**

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

```

Stratification ensures that the train and test sets have the same proportion of heart disease cases as the original dataset.

**Class Balancing with SMOTE:**

```
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

```

Unlike random duplication, SMOTE synthesizes new examples for the minority class, helping the models learn better decision boundaries without overfitting on exact duplicates.

**Feature Scaling:**

```
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

```

Models like SVM are highly sensitive to the scale of the input features. `StandardScaler` standardizes features by removing the mean and scaling to unit variance.

### 2. Model Training & Cross-Validation

```
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Evaluated using F1-score across 5 folds for SVM, Naive Bayes, and Random Forest

```

| Model | Average F1-Score (CV) | Strength |
| --- | --- | --- |
| **SVM** | \~0.7969 | Effective in finding hyperplanes in scaled continuous data |
| **Gaussian Naive Bayes** | \~0.8105 | Fast, probabilistic approach assuming feature independence |
| **Random Forest** | \~0.8358 | Non-linear ensemble of decision trees, robust to complex interactions |

Exportar para Sheets### 3. Voting Ensemble

```
from scipy.stats import mode

ensemble_preds = mode([svm_preds, nb_preds, rf_preds], axis=0).mode.flatten()

```

The final prediction uses a majority vote (`mode` from SciPy). If the models disagree, the most frequent prediction across the three classifiers is chosen.

### 4. Feature Importance

The notebook extracts `feature_importances_` from the Random Forest model and plots them using Seaborn, allowing us to interpret which clinical metrics (e.g., maximum heart rate, age, or cholesterol) drive the predictions the most.

---

## 💡 Key Concepts

### Proper Leakage Prevention

Notice how `train_test_split` is performed *before* applying SMOTE and `StandardScaler`. This is a crucial best practice. If you apply SMOTE or scaling to the entire dataset before splitting, information from the test set "leaks" into the training set, leading to falsely inflated performance metrics.

### Ensemble Performance

While the individual models perform well, the Ensemble matrix shows an overall robust performance, achieving an F1-Score of **\~0.88** and an accuracy of **88%** on the test set, proving that combining models can effectively balance out individual algorithmic weaknesses.

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
