# Predicting Hospital Readmission for Diabetic Patients

## Project Overview

This project predicts **hospital readmission within 30 days** for diabetic patients using machine learning techniques. Hospital readmissions increase healthcare costs and indicate gaps in patient care. Predicting high-risk patients helps hospitals take preventive actions and improve patient outcomes.

## Project Objectives

1. Identify the **most important predictors** of hospital readmission.
2. Develop accurate **machine learning models**.
3. Compare multiple models to select the **best-performing model**.
4. Deploy predictions using **Flask API**.
5. Visualize insights using **Tableau dashboards**.

Three machine learning models were implemented:

* Decision Tree
* XGBoost
* LightGBM

**Final Selected Model:** XGBoost

---

# Dataset Description

Dataset Source:
**UCI Machine Learning Repository**
**Diabetes 130-US Hospitals Dataset (1999–2008)**

Link:
[https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

## Dataset Summary

* **Total Records:** 101,766
* **Original Features:** 50
* **Selected Features:** **30**
* **Hospitals:** 130 US hospitals
* **Target Variable:** Readmission within 30 days

Each row represents a **hospital encounter** of a diabetic patient.

---

# Target Variable

**readmitted**

Converted into binary:

| Value | Meaning                   |
| ----- | ------------------------- |
| 0     | Not Readmitted            |
| 1     | Readmitted within 30 days |

---

# Data Preprocessing

Several preprocessing steps were performed to clean and prepare the dataset.

## Missing Value Handling

* Replaced `'?'` with `'Unknown'`
* Dropped columns with excessive missing values:

  * weight
  * payer_code
  * medical_specialty

## Encoding

Categorical variables were converted into numeric format using:

* Label Encoding
* One-Hot Encoding

Examples:

* Gender
* Race
* Admission Type
* Diagnosis Categories

---

# Feature Engineering

New features were created to improve prediction performance.

## Engineered Features

* **service_utilization**
  Sum of outpatient, emergency, and inpatient visits.

* **numchange**
  Number of medication changes.

* **meds_per_diag**
  Ratio of medications to diagnoses.

* **Log-transformed Features**

  * time_in_hospital_log
  * num_medications_log

## Diagnosis Grouping

ICD diagnosis codes were grouped into:

* Diabetes
* Circulatory
* Respiratory
* Injury
* Other

---

# Feature Selection

Feature selection was performed using:

**Recursive Feature Elimination (RFE)**
with **RandomForest**

## Selected Features

* **Top 30 most important features**
* Reduced dimensionality
* Improved model efficiency
* Reduced training time

---

# Data Balancing

The dataset was highly imbalanced:

* **Non-readmitted:** ~88%
* **Readmitted:** ~12%

To handle imbalance:

## SMOTE (Synthetic Minority Over-sampling Technique) was applied.

SMOTE generates synthetic samples of the minority class.

## Benefits of SMOTE

* Improves minority class learning
* Reduces model bias
* Improves Recall performance
* Balances class distribution

---

# Model Development

Three machine learning models were trained.

## Models Used

1. Decision Tree
2. XGBoost
3. LightGBM

---

# Hyperparameter Tuning

Hyperparameters were optimized using:

**GridSearchCV**

With:

* Cross-validation
* Parallel processing
* Recall-based optimization

---

# Model Evaluation Metrics

Models were evaluated using:

* Recall
* Precision
* F1-Score
* ROC-AUC
* PR-AUC
* Cross-Validation Recall
* Confusion Matrix

Recall was prioritized because:

**Missing high-risk patients is dangerous in healthcare systems.**

---

# Model Performance Results

## Final Model Comparison

| Model         | Recall   | F1       | ROC-AUC  | CV Recall       |
| ------------- | -------- | -------- | -------- | --------------- |
| Decision Tree | 0.69     | 0.54     | 0.84     | 0.83 ± 0.03     |
| **XGBoost**   | **0.76** | 0.53     | **0.85** | **0.93 ± 0.05** |
| LightGBM      | 0.70     | **0.55** | 0.85     | 0.88 ± 0.11     |

---

# Final Model Selection

**Selected Model:** XGBoost

## Reason for Selection

* Highest Recall (0.76)
* Highest Cross-Validation Recall (0.93)
* Strong ROC-AUC performance
* Stable across validation folds

XGBoost provided the **best balance between performance and reliability**.

---

# Threshold Optimization

Prediction threshold was tuned to:

**Threshold = 0.714**

## Goal

* Maintain Recall ≥ 0.70
* Improve Precision

This ensures better detection of high-risk patients.

---

# Confusion Matrix (XGBoost)

| Actual \ Predicted | 0     | 1    |
| ------------------ | ----- | ---- |
| 0                  | 10341 | 1989 |
| 1                  | 419   | 1340 |

---

# Key Predictors Identified

Top predictors influencing readmission:

1. Number of Inpatient Visits
2. Service Utilization
3. Time in Hospital
4. Number of Medications
5. Medication Changes

These predictors significantly impact readmission risk.

---

# Visualization

Visualizations were created using:

* Matplotlib
* Seaborn
* Tableau

## Key Visualizations

* Readmission Distribution
* Service Utilization vs Readmission
* Medication Changes Analysis
* Time in Hospital Distribution
* Feature Correlation Heatmap

---

# Industry Impact

This model helps:

* Identify high-risk patients early
* Reduce hospital readmissions
* Improve patient outcomes
* Reduce healthcare costs

## Industry Benchmark Comparison

| Metric  | Achieved | Benchmark |
| ------- | -------- | --------- |
| Recall  | 0.76     | ≥ 0.70    |
| ROC-AUC | 0.85     | ≥ 0.80    |
| PR-AUC  | 0.54     | ≥ 0.50    |

The model meets key healthcare prediction standards.

---

# Tools & Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn

---

# Future Work

* Improve Precision ≥ 0.50
* Add additional clinical variables
* Deploy real-time prediction system
* Integrate with Electronic Health Records (EHR)
* Expand dataset coverage
* Develop advanced dashboards

---

# Acknowledgments

* UCI Machine Learning Repository
* Healthcare Readmission Research Community
* Open-source Python Libraries
