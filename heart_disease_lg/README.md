# 🫀 10-Year CHD Risk Prediction (Logistic Regression)

Predicts the 10-year risk of developing Coronary Heart Disease (CHD) using **Logistic Regression**. Introduces data cleaning, handling imbalanced medical datasets with SMOTE, and evaluating probabilistic classifiers.

---

## 🎯 Objective

Using the famous Framingham Heart Study dataset, build a baseline Logistic Regression model to predict whether a patient will develop CHD within 10 years (`TenYearCHD`) based on their current demographics, behavioral, and medical risk factors.

---

## 📁 Project Structure

```
chd_risk_prediction/
├── heart_disease_lg.ipynb          # Main notebook
└── framingham.csv                  # Dataset (Framingham Heart Study)

```

---

## 🔬 Approach

### Pipeline

```
CSV → Data Cleaning (Drop/Rename) → EDA → StandardScaler → Train/Test Split
    → SMOTE → Logistic Regression → Confusion Matrix Evaluation

```

### 1. Data Cleaning

```
disease_df.drop(columns=['education'], inplace=True, axis=1)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

```

Irrelevant features (`education`) are removed, columns are renamed for clarity, and rows with missing values (`NaN`) are dropped to ensure a clean dataset for the algorithm.

### 2. Exploratory Data Analysis (EDA)

```
sns.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")

```

Visualizing the target variable reveals a heavy class imbalance: \~3179 negative cases (no CHD) vs. only \~572 positive cases (CHD).

### 3. Preprocessing & Balancing

```
X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

```

The data is standardized first, split into training and testing sets (70/30), and then the training data is balanced using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the Logistic Regression model doesn't become biased toward predicting the negative class.

### 4. Model Training & Evaluation

```
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_res, y_train_res)

```

The model is evaluated using accuracy, a classification report, and a Seaborn heatmap of the confusion matrix.

---

## 💡 Key Concepts

### Logistic Regression for Medical Data

Despite its name, Logistic Regression is a *classification* algorithm. It uses a logistic (sigmoid) function to model the probability of a binary outcome (e.g., CHD vs. No CHD). It is widely used in medical fields because the model's coefficients are easily interpretable as log-odds.

### Interpreting the Results

The model achieves an overall accuracy of **\~67%**. However, looking at the classification report:

* The recall for the positive class (1) is **0.71**.
* This means the model successfully identifies 71% of the patients who *actually* end up getting CHD. In medical screening, having a higher recall is often preferred even if precision drops, because missing a sick patient (False Negative) is more dangerous than a false alarm (False Positive).

### ⚠️ Notes on This Implementation

* **Scaling before splitting:** The notebook applies `StandardScaler` to the entire dataset `X` *before* the `train_test_split`. This causes **data leakage**, as the mean and variance of the test set influence the scaling of the training set. Standard scaling should ideally be fit *only* on the training data.

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn

```