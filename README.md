# 📈 Term Deposit Subscription Prediction Project

---

## 📅 Project Title
**Predicting Customer Subscription to Term Deposits using Machine Learning**

---

## 💡 Project Overview
This project predicts whether a customer will subscribe to a term deposit based on a marketing campaign dataset from a Portuguese bank.

The task is framed as a **binary classification problem**:
- **1**: Customer subscribes
- **0**: Customer does not subscribe

The project builds:
- A **Baseline (Naïve) Model** using simple rule-based logic
- Two **Novel Machine Learning Models**:
  - Logistic Regression
  - Random Forest Classifier

---

## 📂 Dataset Link
- [Bank Marketing Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## 📚 Dataset Description
- **Source**: Portuguese bank marketing campaign
- **Observations**: 45,000+ rows
- **Features**: Demographics, financial, contact details, campaign interaction, etc.

Major columns:
- Demographics: `age`, `marital`, `education`, `job`
- Financial: `salary`, `balance`, `housing`, `loan`
- Marketing Campaign: `contact`, `previous outcome`, `duration`
- Target: `response` (yes/no)

---

## 🔄 Project Pipeline

### 1. Data Preprocessing
- Created `response_flag` (1 for 'yes', 0 for 'no')
- Binary encoding and One-Hot Encoding
- Missing value imputation (`pdays`)
- Feature engineering (`was_contacted_before`, `duration_min`)
- Dropped unnecessary columns (`customerid`, `duration`)

### 2. Exploratory Data Analysis (EDA)
- Univariate and Bivariate Analysis
- Multivariate Heatmaps
- Violin Plots, Boxplots
- ROC Curve visualization

### 3. Handling Class Imbalance
- Used **class_weight='balanced'** during model building

### 4. Model Building
- **Baseline Naïve Model** (Simple Rule Based)
- **Logistic Regression** with cross-validation and tuning
- **Random Forest Classifier** with hyperparameter tuning (GridSearchCV)

### 5. Model Evaluation
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score
- Feature Importance Plot

---

## 📈 Model Performance Summary

| Model | Accuracy | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | ROC-AUC |
|:------|:---------|:-----------------|:-------------------|:------------------|:--------|
| Baseline Rule | 88% | 6% | 36% | 11% | 0.52 |
| Logistic Regression (Tuned) | 79% | 73% | 33% | 45% | 0.765 |
| Random Forest (Default) | 90% | 31% | 68% | 42% | 0.645 |
| Random Forest (Tuned) | 89% | 69% | 54% | 61% | 0.81 |

---

## 📋 Key Observations
- Strong class imbalance: ~88% 'no' and ~12% 'yes'
- **Duration of call** was the strongest predictor
- Customers with **tertiary education**, **single marital status**, and **previous contact** had higher subscription chances
- Random Forest after tuning became highly competitive

---

## 🚀 Future Work
- Explore advanced models (XGBoost, LightGBM)
- Feature interactions and automated feature engineering
- Threshold tuning for better marketing decision cutoffs
- Deployment using Streamlit for live predictions

---


## 📦 Requirements
Python 3.8+ recommended with:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install using:
```bash
pip install -r requirements.txt
****
