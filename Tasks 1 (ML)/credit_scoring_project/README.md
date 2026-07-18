# Credit Scoring Model
### Industry-Standard Machine Learning Project


---

## Project Overview

A complete, end-to-end credit scoring pipeline that predicts whether a bank customer will default on a loan. Built to industry standards using classical machine learning with scikit-learn.

---

## Business Problem

Banks need to assess the creditworthiness of loan applicants to minimise financial losses from bad debt. A reliable credit scoring model automates this assessment, enabling faster decisions and consistent risk evaluation at scale.

---

## Objective

| Value | Meaning |
|-------|---------|
| `0` | No Default — customer repaid the loan |
| `1` | Default — customer failed to repay |

---

## Dataset Description

File: `credit_data.csv` — 3,000 rows, 11 columns

| Column | Description |
|--------|-------------|
| `income` | Annual income (USD) |
| `debt` | Total outstanding debt (USD) |
| `age` | Customer age (years) |
| `loan_amount` | Loan amount requested (USD) |
| `payment_history` | Historical payment behaviour score (1–10) |
| `credit_score` | Credit bureau score (300–850) |
| `credit_utilization` | Credit used ÷ credit available (0–1) |
| `num_credit_lines` | Number of active credit accounts |
| `late_payments` | Count of past late payments |
| `employment_length` | Years of continuous employment |
| `default` | **Target**: 0 = No Default, 1 = Default |

---

## Folder Structure

```
credit_scoring_project/
│
├── credit_scoring.py            ← Main script
├── credit_data.csv              ← Dataset
├── requirements.txt             ← Dependencies
├── README.md                    ← This file
│
├── reports/
│   ├── figures/
│   │   ├── target_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── feature_histograms.png
│   │   ├── boxplots_by_target.png
│   │   ├── roc_pr_curves.png
│   │   └── feature_importance.png
│   └── metrics/
│       ├── eda_summary.txt
│       ├── model_comparison.csv
│       ├── feature_importance.csv
│       ├── logistic_regression_report.txt
│       ├── decision_tree_report.txt
│       └── random_forest_report.txt
│
├── models/
│   ├── credit_scoring_model.pkl ← Best trained model
│   └── scaler.pkl               ← Fitted StandardScaler
│
└── outputs/
    └── predictions.csv          ← Test predictions with probabilities
```

---

## Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 1.5 | Data loading and manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | Visualisations and plots |
| `scikit-learn` | ≥ 1.2 | ML models, preprocessing, metrics |
| `joblib` | ≥ 1.2 | Model serialisation |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python credit_scoring.py
```

Place `credit_data.csv` in the same folder before running.

---

## Data Cleaning

| Step | Method | Reason |
|------|--------|--------|
| Duplicate removal | `drop_duplicates()` | Prevent model bias from repeated rows |
| Missing value imputation | Column mean | Simple, effective for normally distributed features |
| Outlier treatment | IQR capping (P1–P99) | Preserves data size, reduces skew influence |

---

## Feature Engineering

Six domain-driven financial features are created:

| Feature | Formula | Business Rationale |
|---------|---------|-------------------|
| `debt_to_income` | debt / income | Standard bank underwriting metric — high ratio = over-leveraged |
| `loan_to_income` | loan_amount / income | Measures affordability of the new loan |
| `late_payment_ratio` | late_payments / num_credit_lines | Normalised measure of payment reliability |
| `utilization_risk` | Binned credit_utilization | >30% utilisation is a known risk signal |
| `income_per_credit_line` | income / num_credit_lines | Financial capacity per credit obligation |
| `risk_score_index` | Weighted composite | Single-number summary of key risk factors |

---

## Machine Learning Models

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| **Logistic Regression** | Linear probabilistic classifier | `C=1.0`, `class_weight=balanced`, `max_iter=1000` |
| **Decision Tree** | Rule-based threshold splits | `max_depth=8`, `min_samples_split=20`, `class_weight=balanced` |
| **Random Forest** | Ensemble of 200 decision trees | `n_estimators=200`, `max_depth=10`, `class_weight=balanced` |

`class_weight=balanced` compensates for the 78/22 class imbalance automatically.

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | % of all predictions that were correct |
| **Precision** | Of predicted defaults, how many were actual defaults |
| **Recall** | Of all actual defaults, how many did the model detect |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Ability to distinguish defaulters vs non-defaulters (1.0 = perfect) |
| **Confusion Matrix** | TP, TN, FP, FN breakdown |

> **Why ROC-AUC for ranking?** Unlike accuracy, ROC-AUC is robust to class imbalance. It measures ranking quality — can the model score actual defaulters higher than non-defaulters? This is the most meaningful metric for credit risk.

---

## Results

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|-------|----------|-----------|--------|----|---------|
| 1 | **Logistic Regression** | 0.81 | 0.63 | 0.36 | 0.46 | **0.83** |
| 2 | Random Forest | 0.80 | 0.58 | 0.25 | 0.35 | 0.80 |
| 3 | Decision Tree | 0.78 | 0.49 | 0.29 | 0.36 | 0.67 |

---

## Feature Importance

Top predictors of loan default (Logistic Regression coefficients):

1. Late Payments
2. Credit Utilization
3. Income
4. Credit Score
5. Debt

---

## Business Insights

- **Credit score and utilization** are the two strongest predictors — customers with low scores and high utilization are highest risk.
- **Late payment history** is a powerful signal — even one or two late payments significantly increases default probability.
- **Income alone is insufficient** — debt-to-income ratio is a far better predictor than raw income.
- At a default rate of 22%, the model achieves ROC-AUC of 0.83, meaning it correctly ranks a defaulter above a non-defaulter 83% of the time.

---

## Limitations

- Dataset is synthetic — real-world performance may vary with production data
- Recall of 36% means ~64% of actual defaulters are missed; threshold tuning could improve this
- No temporal validation — in production, walk-forward validation on time-ordered data is required

---

## Future Improvements

- Apply SMOTE or class resampling to improve recall on the minority class
- Add threshold optimisation to tune Precision/Recall trade-off for business requirements
- Implement XGBoost or Gradient Boosting for potentially higher AUC
- Add time-series validation (walk-forward) for production deployment
- Build a REST API with FastAPI for real-time scoring
- Integrate MLflow for experiment tracking and model registry
