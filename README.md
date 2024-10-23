# Fraud Detection in Credit Card Trasactions

## Introduction

This repository contains the code for a project focused on detecting credit card fraud using machine learning models. The goal of the project is to predict fraudulent credit card transactions by building and evaluating multiple machine learning models, including Logistic Regression, Decision Trees, Random Forests, Boosted Trees, and XGBoost.

## Repository Contents

1. **data_explore.ipynb**: Code for exploratory data analysis (EDA), including summary statistics, visualizations, and initial insights into the credit card transaction data.
   
2. **clean_make_variables.ipynb**: Code for cleaning the dataset, handling missing values, and creating new variables (feature engineering). This notebook includes steps like imputing missing data, generating derived variables, and encoding categorical fields.

3. **feature_selection.ipynb**: Code for selecting the most relevant features using techniques like Kolmogorov-Smirnov (KS) distance and sequential forward selection. The goal is to reduce dimensionality and improve model performance.

4. **models.ipynb**: Code for building, tuning, and evaluating various machine learning models, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Boosted Tree (LightGBM)
   - Neural Network
   - XGBoost

## Dataset Description

The dataset used in this project consists of credit card transaction records from 2010, with 97,852 transactions in total, of which 2.09% are labeled as fraudulent. Key fields include transaction amounts, merchant details, transaction types, and the fraud label (binary: 1 for fraud, 0 for non-fraud). Data cleaning, variable creation, and feature selection were applied to optimize the model performance for fraud detection.

## Key Findings

1. **Exploratory Data Analysis**: Initial analysis revealed that fraudulent transactions account for only 2.09% of the dataset, highlighting the class imbalance issue that needs to be addressed during model building.
   
2. **Feature Engineering**: Several new variables were created to improve fraud detection, such as transaction frequency, transaction amounts, and day-of-week risk tables. Benford’s law was also applied to detect anomalies in transaction amounts.

3. **Feature Selection**: Variables were selected based on their ability to differentiate between fraud and non-fraud cases, with techniques like KS distance and forward selection ensuring only the most important features were used.

4. **Model Performance**: XGBoost was the best-performing model, resulting in a 74.08% FDR at 3% for the testing dataset and a 55.31% FDR at 3% for the OOT dataset.

## Conclusion

This project successfully demonstrates how machine learning can be applied to detect fraudulent credit card transactions. The XGBoost model provides the best balance of fraud detection rate and stability. Moving forward, future improvements could include gathering more data and experimenting with alternative sampling techniques to handle class imbalance.

---

For more details on the project, please refer to the individual notebooks in this repository.
