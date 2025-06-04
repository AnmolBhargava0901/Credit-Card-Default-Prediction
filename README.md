![image](https://github.com/user-attachments/assets/1a579ba3-514e-4ea1-b474-3c6ac0882fd3)# 🧠 Credit Card Default Prediction

This project uses machine learning techniques to predict whether a customer will default on their credit card payment next month based on various demographic and financial features.

---

## 📂 Dataset

- **Source:** UCI Machine Learning Repository  
- **File Used:** `UCI_Credit_Card.csv`  
- **Target Column:** `default.payment.next.month`  

This dataset contains information on 30,000 credit card clients in Taiwan from April to September 2005.

---

## 🧪 Objective

To build and evaluate machine learning models that predict customer default behavior using demographic features, past bill amounts, and payment history.

---

## 🔧 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## 🔍 Data Preprocessing

1. **Dropped Unnecessary Columns:**
   - `ID`: Irrelevant for prediction

2. **One-Hot Encoding:**
   - `EDUCATION` → `EDU_1`, `EDU_2`, ...
   - `MARRIAGE` → `MAR_1`, `MAR_2`, ...

3. **Feature Scaling:**
   - StandardScaler used to normalize all numerical features

4. **Train-Test Split:**
   - 70% test, 30% train using `train_test_split`

---

## 📊 Correlation Analysis

A heatmap was generated to identify the strongest relationships with the target feature (`default.payment.next.month`):

### 🔹 Strongest Correlations:
| Feature | Correlation |
|--------|-------------|
| `PAY_0` | **+0.32** |
| `PAY_2` | +0.26 |
| `PAY_3` | +0.24 |
| `LIMIT_BAL` | -0.15 |

> Payment delay features (`PAY_0` to `PAY_6`) are most predictive. Credit limit has a slight negative correlation with default.

### 🔸 Highly Redundant Features:
- Strong correlations among:
  - `BILL_AMT1`–`BILL_AMT6`
  - `PAY_AMT1`–`PAY_AMT6`

These could be considered for dimensionality reduction or feature selection to avoid multicollinearity.

### 🖼 Heatmap:

![image](https://github.com/user-attachments/assets/7fea6338-5089-48af-a71b-1b7b1d85f4c2)


---

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | A linear baseline classifier |
| Support Vector Machine (SVC) | A margin-based classifier for complex boundaries |
| Multi-Layer Perceptron (MLP) | A simple neural network classifier |

Each model is trained on the training set and evaluated on the test set.

---

## 🧪 Model Training and Evaluation

```python
models = {
    LogisticRegression(): "Logistic Regression",
    SVC(): "Support Vector Machine",
    MLPClassifier(): "Neural Network",
}

for model, name in models.items(): 
    model.fit(X_train, y_train)
    print(f"{name} Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

