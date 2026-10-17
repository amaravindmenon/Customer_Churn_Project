import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# 2. Same test split used for both models
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

models = {
    "Logistic Regression": joblib.load(
        "models/logistic_regression_churn_model.joblib"
    ),

    "Random Forest": joblib.load(
        "models/random_forest_churn_model.joblib"
    )
}


# --------------------------------------------------
# 4. Thresholds to test
# --------------------------------------------------

thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]


all_results = []


# --------------------------------------------------
# 5. Test every threshold
# --------------------------------------------------

for model_name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        all_results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        print(
            f"Threshold {threshold:.2f} | "
            f"Accuracy {accuracy:.3f} | "
            f"Precision {precision:.3f} | "
            f"Recall {recall:.3f} | "
            f"F1 {f1:.3f}"
        )


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results = pd.DataFrame(all_results)

results.to_csv(
    "outputs/threshold_analysis.csv",
    index=False
)


# --------------------------------------------------
# 7. Show best threshold by F1
# --------------------------------------------------

print("\nThreshold analysis saved to:")
print("outputs/threshold_analysis.csv")

print("\nBest threshold by F1:")

best = results.loc[
    results.groupby("Model")["F1"].idxmax()
]

print(
    best.to_string(index=False)
)