import os
import joblib
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

DATA_PATH = "data/client_customer_data.xlsx"
MODEL_PATH = "models/logistic_regression_churn_model.joblib"
OUTPUT_PATH = "outputs/local_explanations_high_risk.csv"

os.makedirs("outputs", exist_ok=True)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df = pd.read_excel(DATA_PATH)

model = joblib.load(MODEL_PATH)


# =========================================================
# PREPARE INPUT DATA
# =========================================================

# Keep Customer_ID separately so we can identify customers
customer_ids = df["Customer_ID"].copy()

# Remove target and ID from model input
X = df.drop(columns=["Churned", "Customer_ID"])


# =========================================================
# GENERATE CHURN PROBABILITIES
# =========================================================

probabilities = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Churn_Probability": probabilities
})


# =========================================================
# SELECT HIGH-RISK CUSTOMERS
# =========================================================

# We use 0.60 because our threshold analysis showed
# that Logistic Regression achieved its best tested F1
# score at this threshold.

HIGH_RISK_THRESHOLD = 0.60

high_risk_mask = results["Churn_Probability"] >= HIGH_RISK_THRESHOLD

high_risk_indices = np.where(high_risk_mask)[0]


# =========================================================
# GET LOGISTIC REGRESSION COMPONENTS
# =========================================================

# Our Logistic Regression pipeline uses:
#
# preprocessing
# model

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["model"]


# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Logistic Regression coefficients
coefficients = classifier.coef_[0]


# =========================================================
# TRANSFORM CUSTOMER DATA
# =========================================================

X_transformed = preprocessor.transform(X)

# Convert sparse matrix if necessary
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()


# =========================================================
# CREATE LOCAL EXPLANATIONS
# =========================================================

explanation_rows = []


for index in high_risk_indices:

    customer_id = customer_ids.iloc[index]

    probability = probabilities[index]

    # Contribution of every transformed feature
    contributions = X_transformed[index] * coefficients

    # Positive contribution = pushes prediction toward churn
    # Negative contribution = pushes prediction away from churn

    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    # Sort positive contributors
    positive = contribution_df[
        contribution_df["Contribution"] > 0
    ].sort_values(
        by="Contribution",
        ascending=False
    )

    # Sort protective contributors
    negative = contribution_df[
        contribution_df["Contribution"] < 0
    ].sort_values(
        by="Contribution",
        ascending=True
    )


    # -----------------------------------------------------
    # Top 3 risk factors
    # -----------------------------------------------------

    top_risk = positive.head(3)

    risk_factors = []

    for _, row in top_risk.iterrows():
        risk_factors.append(
            f"{row['Feature']} ({row['Contribution']:.3f})"
        )


    # -----------------------------------------------------
    # Top 3 protective factors
    # -----------------------------------------------------

    top_protective = negative.head(3)

    protective_factors = []

    for _, row in top_protective.iterrows():
        protective_factors.append(
            f"{row['Feature']} ({row['Contribution']:.3f})"
        )


    # -----------------------------------------------------
    # Convert lists to readable text
    # -----------------------------------------------------

    risk_text = "; ".join(risk_factors)

    protective_text = "; ".join(protective_factors)


    # -----------------------------------------------------
    # Save explanation
    # -----------------------------------------------------

    explanation_rows.append({
        "Customer_ID": customer_id,
        "Churn_Probability": round(float(probability), 6),
        "Risk_Level": "High",
        "Top_Risk_Factors": risk_text,
        "Top_Protective_Factors": protective_text
    })


# =========================================================
# CREATE OUTPUT DATAFRAME
# =========================================================

explanations = pd.DataFrame(explanation_rows)


# Sort highest-risk customers first
explanations = explanations.sort_values(
    by="Churn_Probability",
    ascending=False
)


# =========================================================
# SAVE RESULTS
# =========================================================

explanations.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n==============================================")
print("LOCAL EXPLANATIONS - HIGH RISK CUSTOMERS")
print("==============================================")

print(
    explanations.head(10).to_string(index=False)
)


print("\n==============================================")
print("SUMMARY")
print("==============================================")

print(
    f"Total customers analysed: {len(df)}"
)

print(
    f"High-risk customers: {len(explanations)}"
)

print(
    f"High-risk threshold: {HIGH_RISK_THRESHOLD}"
)

print(
    f"Output file: {OUTPUT_PATH}"
)

print("\nLocal explanation analysis completed successfully.")