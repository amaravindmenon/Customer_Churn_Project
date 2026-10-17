import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_excel("data/client_customer_data.xlsx")

y = df["Churned"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churned", "Customer_ID"])


# --------------------------------------------------
# 2. Same train/test split used by both models
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# --------------------------------------------------
# 3. Load trained models
# --------------------------------------------------

logistic_model = joblib.load(
    "models/logistic_regression_churn_model.joblib"
)

random_forest_model = joblib.load(
    "models/random_forest_churn_model.joblib"
)


# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 5. Evaluation function
# --------------------------------------------------

def evaluate_model(name, y_true, predictions, probabilities):

    cm = confusion_matrix(y_true, predictions)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy_score(y_true, predictions), 4))
    print("Precision:", round(precision_score(y_true, predictions), 4))
    print("Recall   :", round(recall_score(y_true, predictions), 4))
    print("F1 Score :", round(f1_score(y_true, predictions), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_true, probabilities), 4))

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions),
        "Recall": recall_score(y_true, predictions),
        "F1": f1_score(y_true, predictions),
        "ROC_AUC": roc_auc_score(y_true, probabilities)
    }


# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


# --------------------------------------------------
# 7. Save comparison
# --------------------------------------------------

results = pd.DataFrame([
    logistic_results,
    rf_results
])

results.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\n")
print(results)

print("\nComparison saved to:")
print("outputs/model_comparison.csv")