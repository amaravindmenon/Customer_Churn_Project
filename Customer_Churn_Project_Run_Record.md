# Customer Churn Prediction Project -- Project Run Guide

## 1. Purpose

This document records the project structure, scripts, trained models,
and generated outputs that are present in the project.

**This guide is based only on the project files/runs shown in the
provided project-folder screenshot and the two model filenames confirmed
in the conversation.**

------------------------------------------------------------------------

# 2. Project Structure

The project contains the following main folders and files:

``` text
Customer_Churn_Project/
│
├── data/
│   └── client_customer_data.xlsx
│
├── models/
│   ├── logistic_regression_churn_model.joblib
│   └── random_forest_churn_model.joblib
│
├── outputs/
│   ├── customer_churn_predictions.csv
│   ├── decision_support.csv
│   ├── high_risk_retention_priorities.csv
│   ├── local_explanations_high_risk.csv
│   ├── logistic_feature_importance_current.csv
│   ├── model_comparison.csv
│   ├── rf_feature_importance_current.csv
│   └── threshold_analysis.csv
│
├── src/
│   ├── check_model.py
│   ├── compare_models.py
│   ├── decision_support.py
│   ├── feature_importance.py
│   ├── local_explanations.py
│   ├── predict_churn.py
│   ├── threshold_analysis.py
│   ├── train_model.py
│   └── train_random_forest.py
│
├── Customer_Churn_Project_Execution_Guide.md
└── requirements.txt
```

------------------------------------------------------------------------

# 3. Dataset

The input dataset is:

``` text
data/client_customer_data.xlsx
```

This is the source customer dataset used by the project scripts.

The dataset is kept separately inside the `data` folder so that the
machine-learning scripts can load it as the project input.

------------------------------------------------------------------------

# 4. Trained Models

The project contains exactly **two trained model files**.

## Model 1 -- Logistic Regression

``` text
models/logistic_regression_churn_model.joblib
```

This is the Logistic Regression churn model.

The model is used by the prediction, feature-importance,
local-explanation, threshold-analysis, and decision-support workflow
where applicable.

------------------------------------------------------------------------

## Model 2 -- Random Forest

``` text
models/random_forest_churn_model.joblib
```

This is the Random Forest churn model.

It is used as the second model for comparison with Logistic Regression.

------------------------------------------------------------------------

# 5. Source Code / Python Scripts

The `src` folder contains the scripts used during the project.

## 5.1 `train_model.py`

``` text
src/train_model.py
```

Purpose:

``` text
Train the Logistic Regression churn model.
```

The resulting model is:

``` text
models/logistic_regression_churn_model.joblib
```

Run:

``` bat
python src  rain_model.py
```

------------------------------------------------------------------------

## 5.2 `train_random_forest.py`

``` text
src/train_random_forest.py
```

Purpose:

``` text
Train the Random Forest churn model.
```

The resulting model is:

``` text
models/random_forest_churn_model.joblib
```

Run:

``` bat
python src  rain_random_forest.py
```

------------------------------------------------------------------------

# 6. Model Checking

## `check_model.py`

``` text
src/check_model.py
```

Purpose:

``` text
Inspect/check the saved model structure.
```

Run:

``` bat
python src\check_model.py
```

This script was part of the project workflow for checking the trained
model objects before using them in later analysis scripts.

------------------------------------------------------------------------

# 7. Model Comparison

## `compare_models.py`

``` text
src/compare_models.py
```

Purpose:

``` text
Compare the Logistic Regression and Random Forest models.
```

The comparison output is stored in:

``` text
outputs/model_comparison.csv
```

Run:

``` bat
python src\compare_models.py
```

The comparison is based on the two trained models:

``` text
logistic_regression_churn_model.joblib
random_forest_churn_model.joblib
```

------------------------------------------------------------------------

# 8. Churn Prediction

## `predict_churn.py`

``` text
src/predict_churn.py
```

Purpose:

``` text
Generate customer churn predictions using the trained model.
```

The generated prediction file is:

``` text
outputs/customer_churn_predictions.csv
```

Run:

``` bat
python src\predict_churn.py
```

The prediction output provides the model-generated churn prediction
information for customers.

------------------------------------------------------------------------

# 9. Feature Importance

## `feature_importance.py`

``` text
src/feature_importance.py
```

Purpose:

``` text
Identify and save important model features/signals.
```

The project contains separate feature-importance output files:

``` text
outputs/logistic_feature_importance_current.csv
outputs/rf_feature_importance_current.csv
```

Run:

``` bat
python srceature_importance.py
```

The two output files correspond to the two trained models:

``` text
Logistic Regression
Random Forest
```

------------------------------------------------------------------------

# 10. Threshold Analysis

## `threshold_analysis.py`

``` text
src/threshold_analysis.py
```

Purpose:

``` text
Evaluate different prediction thresholds and their effect on classification performance.
```

Output:

``` text
outputs/threshold_analysis.csv
```

Run:

``` bat
python src  hreshold_analysis.py
```

This step is important because a churn-risk model produces a prediction
score/probability, and the threshold determines how that score is
converted into a classification or risk decision.

------------------------------------------------------------------------

# 11. Local Explanations

## `local_explanations.py`

``` text
src/local_explanations.py
```

Purpose:

``` text
Generate explanations for individual high-risk customer predictions.
```

Output:

``` text
outputs/local_explanations_high_risk.csv
```

Run:

``` bat
python src\local_explanations.py
```

This provides customer-level explanation evidence rather than only
giving an overall model result.

------------------------------------------------------------------------

# 12. Decision Support

## `decision_support.py`

``` text
src/decision_support.py
```

Purpose:

``` text
Convert churn-risk predictions into decision-support information.
```

Outputs:

``` text
outputs/decision_support.csv
outputs/high_risk_retention_priorities.csv
```

Run:

``` bat
python src\decision_support.py
```

The output is intended to help organize customers according to their
predicted churn risk and corresponding retention priorities.

------------------------------------------------------------------------

# 13. Complete Project Run Sequence

A logical execution sequence using the scripts present in the project
is:

``` text
1. Prepare the project environment
              ↓
2. Load customer dataset
              ↓
3. Train Logistic Regression
              ↓
4. Train Random Forest
              ↓
5. Check saved models
              ↓
6. Compare models
              ↓
7. Generate churn predictions
              ↓
8. Analyse feature importance
              ↓
9. Analyse prediction thresholds
              ↓
10. Generate local explanations
              ↓
11. Generate decision-support outputs
```

Corresponding commands:

``` bat
python src  rain_model.py
python src  rain_random_forest.py
python src\check_model.py
python src\compare_models.py
python src\predict_churn.py
python srceature_importance.py
python src  hreshold_analysis.py
python src\local_explanations.py
python src\decision_support.py
```

------------------------------------------------------------------------

# 14. Generated Output Files

The project contains the following analysis outputs.

## Customer predictions

``` text
outputs/customer_churn_predictions.csv
```

Contains the generated customer churn prediction results.

------------------------------------------------------------------------

## Model comparison

``` text
outputs/model_comparison.csv
```

Contains the comparison of the two trained models.

------------------------------------------------------------------------

## Logistic Regression feature importance

``` text
outputs/logistic_feature_importance_current.csv
```

Contains feature-importance information for the Logistic Regression
model.

------------------------------------------------------------------------

## Random Forest feature importance

``` text
outputs/rf_feature_importance_current.csv
```

Contains feature-importance information for the Random Forest model.

------------------------------------------------------------------------

## Threshold analysis

``` text
outputs/threshold_analysis.csv
```

Contains the results of testing prediction thresholds.

------------------------------------------------------------------------

## Local explanations

``` text
outputs/local_explanations_high_risk.csv
```

Contains customer-level explanation information for high-risk
predictions.

------------------------------------------------------------------------

## Decision support

``` text
outputs/decision_support.csv
```

Contains the decision-support results generated from churn predictions.

------------------------------------------------------------------------

## High-risk retention priorities

``` text
outputs/high_risk_retention_priorities.csv
```

Contains the high-risk customer retention-priority output.

------------------------------------------------------------------------

# 15. Requirements

The project has:

``` text
requirements.txt
```

Install the project dependencies with:

``` bat
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 16. Project Execution Record

The project therefore contains three main components:

## Input

``` text
data/
└── client_customer_data.xlsx
```

## Machine-learning models

``` text
models/
├── logistic_regression_churn_model.joblib
└── random_forest_churn_model.joblib
```

## Analysis and decision-support outputs

``` text
outputs/
├── customer_churn_predictions.csv
├── decision_support.csv
├── high_risk_retention_priorities.csv
├── local_explanations_high_risk.csv
├── logistic_feature_importance_current.csv
├── model_comparison.csv
├── rf_feature_importance_current.csv
└── threshold_analysis.csv
```

The Python implementation is contained in:

``` text
src/
```

------------------------------------------------------------------------

# 17. Relationship Between the Files

The overall workflow can be represented as:

``` text
client_customer_data.xlsx
          |
          v
    Training Scripts
          |
     +----+----+
     |         |
     v         v
Logistic     Random
Regression   Forest
     |         |
     +----+----+
          |
          v
    compare_models.py
          |
          v
 model_comparison.csv
          |
          +------------------+
          |                  |
          v                  v
   predict_churn.py    feature_importance.py
          |                  |
          v                  +-------------------+
customer_churn_predictions   |                   |
                             v                   v
               logistic_feature...     rf_feature...

          |
          v
 threshold_analysis.py
          |
          v
 threshold_analysis.csv

          |
          v
 local_explanations.py
          |
          v
 local_explanations_high_risk.csv

          |
          v
 decision_support.py
          |
          +-------------------------------+
          |                               |
          v                               v
 decision_support.csv       high_risk_retention_priorities.csv
```

------------------------------------------------------------------------

# 18. What Was Actually Created

Based on the project folder shown, the completed implementation
contains:

### One input dataset

``` text
client_customer_data.xlsx
```

### Two trained models

``` text
logistic_regression_churn_model.joblib
random_forest_churn_model.joblib
```

### Nine project Python scripts

``` text
check_model.py
compare_models.py
decision_support.py
feature_importance.py
local_explanations.py
predict_churn.py
threshold_analysis.py
train_model.py
train_random_forest.py
```

### Eight visible analysis outputs

``` text
customer_churn_predictions.csv
decision_support.csv
high_risk_retention_priorities.csv
local_explanations_high_risk.csv
logistic_feature_importance_current.csv
model_comparison.csv
rf_feature_importance_current.csv
threshold_analysis.csv
```

### Supporting project files

``` text
requirements.txt
Customer_Churn_Project_Execution_Guide.md
```

------------------------------------------------------------------------

# 19. Important Note About Results

This document intentionally records the project based on the files shown
in the project folder.

It does **not** invent additional scripts, models, datasets, output
files, metrics, or project stages that are not shown or confirmed.

The two trained models are:

``` text
logistic_regression_churn_model.joblib
random_forest_churn_model.joblib
```

No additional trained model files are included in this project record.

------------------------------------------------------------------------

# 20. Final Project Workflow

``` text
DATA
  |
  v
client_customer_data.xlsx
  |
  v
MODEL TRAINING
  |
  +---------------------------+
  |                           |
  v                           v
Logistic Regression       Random Forest
  |                           |
  +-------------+-------------+
                |
                v
        MODEL COMPARISON
                |
                v
        CHURN PREDICTIONS
                |
       +--------+---------+
       |        |         |
       v        v         v
 Feature    Threshold   Local
Importance  Analysis  Explanations
       |        |         |
       +--------+---------+
                |
                v
        DECISION SUPPORT
                |
       +--------+---------+
       |                  |
       v                  v
decision_support.csv   high_risk_retention_priorities.csv
```

------------------------------------------------------------------------

## Final Note

This Markdown file is a **project execution/run record**. It documents
what is present in the project and how the visible scripts and outputs
fit together.

It should be kept with the project as supporting documentation alongside
the workbook submission.
