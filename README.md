Term Deposit Subscription Prediction Project

📅 Project Title

Predicting Customer Subscription to Term Deposits using Machine Learning

💡 Project Overview

This project aims to predict whether a customer will subscribe to a term deposit based on a marketing campaign dataset from a Portuguese bank.

The task is framed as a binary classification problem, where:

1: Customer subscribes

0: Customer does not subscribe

The project builds:

A Baseline (Naïve) Model using simple rule-based logic.

Two Novel Models using machine learning:

Logistic Regression

Random Forest Classifier

We address real-world challenges like class imbalance, data preprocessing, feature engineering, and model tuning.

📚 Dataset Description

Source: Portuguese bank marketing campaign dataset

Observations: 45,000+ rows

Features: 20+ features (age, job, marital status, education, balance, etc.)

Major columns include:

Demographics: age, marital, education, job

Financial Info: salary, balance, loan, housing

Marketing Campaign Details: contact type, previous contact outcome, number of contacts, campaign duration

Target Variable: response (yes/no)

🔄 Project Pipeline

1. 🌐 Data Preprocessing

Created response_flag (1 for 'yes', 0 for 'no')

Dropped unnecessary columns like customerid, duration

Encoded:

Binary columns (loan, housing, default, targeted) using 0/1

Multiclass columns (education, marital, job, etc.) using One-Hot Encoding

Imputed missing values for pdays by filling with 999

2. 📊 Exploratory Data Analysis (EDA)

Univariate analysis (distribution plots for categorical & continuous variables)

Bivariate analysis (boxplots, violin plots, pairplots)

Multivariate heatmaps (Education vs Marital vs Response, Job vs Marital, etc.)

ROC Curve plotting

3. 🔀 Feature Engineering

Created was_contacted_before binary feature

Created duration_min (duration in minutes)

Dropped derived columns after transformation

4. 🎓 Handling Class Imbalance

Used Class Weights instead of oversampling

Weighted minority class more during model fitting

Smarter alternative to SMOTE when dataset is not huge

5. 📊 Train-Test Split

Training set: 70%

Test set: 30%

Stratified sampling to maintain class balance

6. 🔄 Baseline Model (Naïve Rule-Based)

Rules based on education, marital status, contact history, loan status

Evaluation metrics: Precision, Recall, F1-score, ROC AUC

7. 👩‍💻 Novel Models

Logistic Regression with class weights

Cross-validated Grid Search for best C and penalty

Final evaluation on Test Set

Random Forest Classifier

Grid Search for n_estimators, max_depth, min_samples_split

Evaluation using ROC AUC, Precision, Recall, F1-score

8. 🔢 Model Interpretability

Feature Importance plot (Top 10 features)

Predicted subscription for a new customer profile

📐 Key Observations

Strong class imbalance exists: ~88% 'no' and 12% 'yes'

Contact duration is the strongest predictor: longer calls ➔ higher subscription likelihood

Customers with "tertiary education", "single" marital status, and previous positive interactions were more likely to subscribe

Random Forest outperformed Logistic Regression in AUC and recall for minority class

📊 Model Performance Summary

Model

ROC AUC Score

F1-Score (Minority Class)

Baseline (Rule-Based)

0.52

0.11

Logistic Regression

0.77

0.51

Tuned Random Forest

0.81

0.61

🛠️ Future Enhancements

Try XGBoost or LightGBM for further improvement

Fine-tune probability thresholds instead of 0.5 cutoff

Create more interaction features (education * contact type, etc.)

Use SMOTE or ADASYN if dataset was smaller

Create a Streamlit app for interactive predictions

📈 Key Skills Demonstrated

End-to-end Machine Learning Project

Data Cleaning and Feature Engineering

Class Imbalance Handling

Model Building, Hyperparameter Tuning

Cross-Validation and Grid Search

Interpretability and Business Communication

📚 Conclusion

We successfully built a robust predictive pipeline for identifying potential term deposit subscribers. This project combines solid EDA, modeling, tuning, and evaluation workflows suitable for a graduate-level data science project and potential industry applications.
